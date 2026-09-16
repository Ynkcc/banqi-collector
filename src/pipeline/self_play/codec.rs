// src/pipeline/self_play/codec.rs — 训练数据记录的二进制编码
//
// 取代原 serialize.rs（逐字段 JSON）：契约由 scheduler.proto 的
// EpisodeBatch/EpisodeRecord/NnueEpisodeRecord 唯一声明（字段号 + schema_version），
// 训练端（banqi_training/episode_codec.py）按同一份 schema 零拷贝解码。
//
// 编码约定：
// - 张量一律「稠密小端缓冲区」（f32/u32/u8 顺序拼接），训练端 np.frombuffer 还原，
//   不做逐元素文本解析；
// - 0/1 特征（棋盘位平面、合法动作掩码）位打包：字节内高位对应较小下标
//   （与 numpy.unpackbits(bitorder='big') 一致），尾部补 0。棋盘布局
//   [步][通道][位置字节]（每通道独立字节对齐），掩码布局 [步][动作字节]；
// - 维度与取值范围在编码时严格校验，不符即报错——宁可丢一批，也不产出
//   形状错位的训练数据。

use anyhow::{bail, Context, Result};
use prost::Message;

use banqi_core::core::env::ResNetObservation;

use crate::pb::{EpisodeBatch, EpisodeRecord, NnueEpisodeRecord, NnueFeatures, NnueMeta};
use crate::pipeline::self_play::{GameEpisode, NnueEpisode, NnueEpisodeMeta, NnueStepFeatures};

/// 记录格式版本。字段语义或布局破坏性变更时递增；训练端遇到不认识的版本直接拒绝。
pub const SCHEMA_VERSION: u32 = 1;

/// 编码一批自对弈记录为 `EpisodeBatch` 二进制载荷（调用方负责再 gzip）。
///
/// `variant` 为变体标识（4x8 / 4x4 / 4x2），训练端据此校验数据未串变体。
/// 无有效样本的局直接跳过（不计入记录），因为它不携带任何训练信号。
pub fn encode_episode_batch(
    variant: &str,
    episodes: &[GameEpisode],
    nnue_episodes: &[NnueEpisode],
) -> Result<Vec<u8>> {
    if variant.is_empty() {
        bail!("变体标识为空，无法标注训练数据来源");
    }
    let mut batch = EpisodeBatch {
        schema_version: SCHEMA_VERSION,
        variant: variant.to_string(),
        episodes: Vec::with_capacity(episodes.len()),
        nnue_episodes: Vec::with_capacity(nnue_episodes.len()),
    };

    for (i, ep) in episodes.iter().enumerate() {
        match encode_episode(ep).with_context(|| format!("编码第 {i} 局 ResNet episode 失败"))? {
            Some(rec) => batch.episodes.push(rec),
            None => eprintln!("⚠️ codec: 第 {i} 局 ResNet episode 无有效样本，跳过"),
        }
    }
    for (i, ep) in nnue_episodes.iter().enumerate() {
        match encode_nnue_episode(ep).with_context(|| format!("编码第 {i} 局 NNUE episode 失败"))? {
            Some(rec) => batch.nnue_episodes.push(rec),
            None => eprintln!("⚠️ codec: 第 {i} 局 NNUE episode 无有效样本，跳过"),
        }
    }
    Ok(batch.encode_to_vec())
}

// ============================================================================
// ResNet（Gumbel MCTS）episode
// ============================================================================

fn encode_episode(ep: &GameEpisode) -> Result<Option<EpisodeRecord>> {
    let steps = ep.samples.len();
    if steps == 0 {
        return Ok(None);
    }

    // 形状从首步样本推导（各变体维度不同，不硬编码常量）
    let (board_channels, board_rows, board_cols) = shape_of(&ep.samples[0].0)?;
    let positions = board_rows * board_cols;
    let scalar_count = ep.samples[0].0.scalars.len();
    let action_space = ep.samples[0].1.len();
    if scalar_count == 0 || action_space == 0 {
        bail!("标量维度={scalar_count}、动作空间={action_space} 不能为 0");
    }

    let board_bytes = board_channels * positions.div_ceil(8);
    let mask_bytes = action_space.div_ceil(8);
    let mut boards_bits = vec![0u8; steps * board_bytes];
    let mut action_masks_bits = vec![0u8; steps * mask_bytes];
    let mut scalars = Vec::with_capacity(steps * scalar_count * 4);
    let mut policies = Vec::with_capacity(steps * action_space * 4);
    let mut mcts_values = Vec::with_capacity(steps * 4);
    let mut completed_qs = Vec::with_capacity(steps * 4);
    let mut game_results = Vec::with_capacity(steps * 4);
    let mut health_diffs = Vec::with_capacity(steps * 4);
    let mut root_visits = Vec::with_capacity(steps * 4);
    let mut actions = Vec::with_capacity(steps * 4);
    let mut is_full_search = Vec::with_capacity(steps);

    for (step, sample) in ep.samples.iter().enumerate() {
        let (obs, policy, mcts_value, completed_q, root_visit, game_result, mask, action,
            health_diff, is_full) = sample;

        // ---- 维度一致性：任一步形状不符即契约破裂，直接失败 ----
        let (c, r, w) = shape_of(obs)?;
        if (c, r, w) != (board_channels, board_rows, board_cols) {
            bail!(
                "第 {step} 步棋盘形状 ({c},{r},{w}) 与首步 ({board_channels},{board_rows},{board_cols}) 不一致"
            );
        }
        if obs.scalars.len() != scalar_count {
            bail!("第 {step} 步标量维度 {} 与首步 {scalar_count} 不一致", obs.scalars.len());
        }
        if policy.len() != action_space {
            bail!("第 {step} 步策略长度 {} 与首步 {action_space} 不一致", policy.len());
        }
        if mask.len() != action_space {
            bail!("第 {step} 步动作掩码长度 {} 与首步 {action_space} 不一致", mask.len());
        }

        // ---- 棋盘：0/1 位平面 ----
        let board = obs
            .board
            .as_slice()
            .with_context(|| format!("第 {step} 步棋盘张量非标准行主序布局，无法取连续切片"))?;
        if board.len() != board_channels * positions {
            bail!("第 {step} 步棋盘元素数 {} 与形状推导值 {} 不一致", board.len(), board_channels * positions);
        }
        let base = step * board_bytes;
        pack_board_frame(
            &mut boards_bits[base..base + board_bytes],
            board,
            board_channels,
            positions,
            step,
        )?;

        // ---- 标量 / 策略：稠密 f32 ----
        let scalar_slice = obs
            .scalars
            .as_slice()
            .with_context(|| format!("第 {step} 步标量张量非连续布局"))?;
        push_f32(&mut scalars, scalar_slice);
        push_f32(&mut policies, policy);

        // ---- 动作掩码：0/1 位图 ----
        let mask_base = step * mask_bytes;
        let mask_frame = &mut action_masks_bits[mask_base..mask_base + mask_bytes];
        for (i, &v) in mask.iter().enumerate() {
            set_bit(mask_frame, i, mask_flag(v, step, i)?);
        }

        // ---- 逐步标量 ----
        push_f32(&mut mcts_values, std::slice::from_ref(mcts_value));
        push_f32(&mut completed_qs, std::slice::from_ref(completed_q));
        push_f32(&mut game_results, std::slice::from_ref(game_result));
        push_f32(&mut health_diffs, std::slice::from_ref(health_diff));
        push_u32(&mut root_visits, *root_visit);
        push_u32(&mut actions, u32_of(*action, "动作索引")?);
        is_full_search.push(u8::from(*is_full));
    }

    let nnue = match &ep.nnue {
        Some((meta, feats)) => {
            if feats.len() != steps {
                bail!("NNUE 特征步数 {} 与样本步数 {steps} 不一致", feats.len());
            }
            Some(encode_nnue_features(meta, feats)?)
        }
        None => None,
    };

    Ok(Some(EpisodeRecord {
        steps: u32_of(steps, "样本步数")?,
        board_channels: u32_of(board_channels, "棋盘通道数")?,
        board_rows: u32_of(board_rows, "棋盘行数")?,
        board_cols: u32_of(board_cols, "棋盘列数")?,
        scalar_count: u32_of(scalar_count, "标量维度")?,
        action_space: u32_of(action_space, "动作空间")?,
        boards_bits,
        scalars,
        policies,
        action_masks_bits,
        mcts_values,
        completed_qs,
        game_results,
        health_diffs,
        root_visits,
        actions,
        is_full_search,
        game_length: u32_of(ep.game_length, "对局步数")?,
        winner: ep.winner,
        health_diff_red: ep.health_diff_red,
        nnue,
    }))
}

// ============================================================================
// NNUE（Expectimax 强自对弈）episode
// ============================================================================

fn encode_nnue_episode(ep: &NnueEpisode) -> Result<Option<NnueEpisodeRecord>> {
    let steps = ep.num_steps();
    if steps == 0 {
        return Ok(None);
    }
    if ep.search_values.len() != steps || ep.players.len() != steps || ep.actions.len() != steps {
        bail!(
            "NNUE episode 各字段步数不一致: features={steps} search_values={} players={} actions={}",
            ep.search_values.len(),
            ep.players.len(),
            ep.actions.len()
        );
    }

    let features = encode_nnue_features(&ep.meta, &ep.features)?;
    let mut search_values = Vec::with_capacity(steps * 4);
    push_f32(&mut search_values, &ep.search_values);
    let mut players = Vec::with_capacity(steps * 4);
    let mut actions = Vec::with_capacity(steps * 4);
    for (i, (&player, &action)) in ep.players.iter().zip(ep.actions.iter()).enumerate() {
        if player != 1 && player != -1 {
            bail!("第 {i} 步行棋方标记 {player} 非法（仅允许 1 红 / -1 黑）");
        }
        push_i32(&mut players, player);
        push_u32(&mut actions, u32_of(action, "动作索引")?);
    }

    Ok(Some(NnueEpisodeRecord {
        meta: Some(meta_of(&ep.meta)),
        steps: u32_of(steps, "样本步数")?,
        features_indices: features.indices,
        features_offsets: features.offsets,
        search_values,
        players,
        actions,
        game_length: u32_of(ep.game_length, "对局步数")?,
        winner: ep.winner,
    }))
}

/// 双视角稀疏特征索引：mover/opponent 段按步顺序拼进 indices，
/// offsets 为前缀和（长度 2*steps+1），训练端据此零拷贝切片。
fn encode_nnue_features(meta: &NnueEpisodeMeta, feats: &[NnueStepFeatures]) -> Result<NnueFeatures> {
    let steps = feats.len();
    let mut indices: Vec<u8> = Vec::with_capacity(steps * 32 * 4);
    let mut offsets: Vec<u8> = Vec::with_capacity((2 * steps + 1) * 4);
    let mut cursor: u32 = 0;
    push_u32(&mut offsets, 0);
    for (step, f) in feats.iter().enumerate() {
        for (side, list) in [("mover", &f.mover), ("opponent", &f.opponent)] {
            for &idx in list.iter() {
                if usize::from(idx) >= meta.feature_dim {
                    bail!(
                        "NNUE {side} 特征第 {step} 步索引 {idx} 超出 feature_dim={}",
                        meta.feature_dim
                    );
                }
                push_u32(&mut indices, u32::from(idx));
            }
            cursor += u32_of(list.len(), "特征索引数")?;
            push_u32(&mut offsets, cursor);
        }
    }
    Ok(NnueFeatures {
        meta: Some(meta_of(meta)),
        indices,
        offsets,
    })
}

fn meta_of(meta: &NnueEpisodeMeta) -> NnueMeta {
    NnueMeta {
        feature_dim: meta.feature_dim as u32,
        states_per_square: meta.states_per_square as u32,
        bag_stride: meta.bag_stride as u32,
        num_active: meta.num_active as u32,
        total_positions: meta.total_positions as u32,
    }
}

// ============================================================================
// 工具
// ============================================================================

fn shape_of(obs: &ResNetObservation) -> Result<(usize, usize, usize)> {
    let s = obs.board.shape();
    if s.len() != 3 || s[0] == 0 || s[1] == 0 || s[2] == 0 {
        bail!("棋盘张量形状非法: {s:?}");
    }
    Ok((s[0], s[1], s[2]))
}

/// 把一个 0/1 棋盘帧打包成位平面：布局 [通道][位置字节]，每通道独立字节对齐
/// （训练端 reshape 为 (channels, ceil(positions/8)) 后逐行 unpackbits 即可还原）。
fn pack_board_frame(
    frame: &mut [u8],
    board: &[f32],
    channels: usize,
    positions: usize,
    step: usize,
) -> Result<()> {
    let chan_bytes = positions.div_ceil(8);
    for c in 0..channels {
        let src = &board[c * positions..(c + 1) * positions];
        let dst = &mut frame[c * chan_bytes..(c + 1) * chan_bytes];
        for (p, &v) in src.iter().enumerate() {
            set_bit(dst, p, binary_flag(v, "棋盘特征", step, c * positions + p)?);
        }
    }
    Ok(())
}

/// 位打包写入：字节内高位对应较小下标（MSB 优先）。
#[inline]
fn set_bit(buf: &mut [u8], index: usize, set: bool) {
    if set {
        buf[index / 8] |= 1 << (7 - (index % 8));
    }
}

#[inline]
fn binary_flag(value: f32, what: &str, step: usize, index: usize) -> Result<bool> {
    if value == 1.0 {
        Ok(true)
    } else if value == 0.0 {
        Ok(false)
    } else {
        bail!("{what} 第 {step} 步下标 {index} 的值 {value} 不是 0/1，无法位打包")
    }
}

#[inline]
fn mask_flag(value: i32, step: usize, index: usize) -> Result<bool> {
    match value {
        1 => Ok(true),
        0 => Ok(false),
        other => bail!("动作掩码第 {step} 步下标 {index} 的值 {other} 不是 0/1"),
    }
}

fn push_f32(out: &mut Vec<u8>, values: &[f32]) {
    out.reserve(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// usize → u32：越界即报错，避免静默截断出错的形状/长度。
fn u32_of(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{what}={value} 超出 u32 范围"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::EpisodeBatch;
    use banqi_core::core::env::ResNetObservation;

    /// 2 步、2 通道 2x2 棋盘、3 维标量、4 动作的最小样本，验证编码后的
    /// 位打包长度与张量布局（Python 侧解码契约的 Rust 侧镜像断言）。
    ///
    /// 棋盘按「[通道][位置]」展开（每步 8 个 0/1），两步图案刻意不同，
    /// 以便同时卡住通道顺序与字节序：
    ///   第 0 步 ch0=[0,1,0,0] ch1=[0,0,0,0] → 0x40, 0x00
    ///   第 1 步 ch0=[1,1,1,1] ch1=[1,0,1,0] → 0xF0, 0xA0
    fn sample_episode() -> GameEpisode {
        let boards: [[f32; 8]; 2] = [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        ];
        let masks: [[i32; 4]; 2] = [[1, 1, 0, 0], [0, 1, 1, 0]];
        let mut samples = Vec::new();
        for step in 0..2usize {
            let obs = ResNetObservation::from_flat(
                boards[step].to_vec(),
                vec![0.5, -0.5, 1.0],
                (2, 2, 2),
            );
            samples.push((
                obs,
                vec![0.25f32, 0.5, 0.25, 0.0],
                0.1f32,
                -0.2f32,
                7u32,
                1.0f32,
                masks[step].to_vec(),
                2usize,
                0.3f32,
                step == 0,
            ));
        }
        GameEpisode {
            samples,
            game_length: 2,
            winner: Some(1),
            health_diff_red: Some(0.3),
            nnue: None,
        }
    }

    #[test]
    fn encode_episode_layout() {
        let ep = sample_episode();
        let raw = encode_episode_batch("4x4", std::slice::from_ref(&ep), &[]).unwrap();
        let batch = EpisodeBatch::decode(raw.as_slice()).unwrap();

        assert_eq!(batch.schema_version, SCHEMA_VERSION);
        assert_eq!(batch.variant, "4x4");
        assert_eq!(batch.episodes.len(), 1);
        let rec = &batch.episodes[0];

        assert_eq!((rec.steps, rec.board_channels, rec.board_rows, rec.board_cols), (2, 2, 2, 2));
        assert_eq!(rec.scalar_count, 3);
        assert_eq!(rec.action_space, 4);
        // 每步每通道 ceil(4/8)=1 字节；2 步 × 2 通道 × 1 字节
        assert_eq!(rec.boards_bits.len(), 4);
        // 位在字节内 MSB 优先：4 个位置只占该字节高 4 位
        assert_eq!(rec.boards_bits, vec![0x40u8, 0x00, 0xF0, 0xA0]);
        // 每步 ceil(4/8)=1 字节掩码：第 0 步前两位 → 0b1100_0000；第 1 步中间两位 → 0b0110_0000
        assert_eq!(rec.action_masks_bits, vec![0xC0u8, 0x60u8]);
        assert_eq!(rec.scalars.len(), 2 * 3 * 4);
        assert_eq!(rec.policies.len(), 2 * 4 * 4);
        assert_eq!(rec.root_visits.len(), 2 * 4);
        assert_eq!(rec.is_full_search, vec![1u8, 0u8]);
        assert_eq!(rec.game_length, 2);
        assert_eq!(rec.winner, Some(1));
    }



    #[test]
    fn reject_non_binary_board() {
        let mut ep = sample_episode();
        ep.samples[0].0.board[[0, 0, 0]] = 0.5;
        let err = encode_episode_batch("4x4", &[ep], &[]).unwrap_err();
        assert!(format!("{err:#}").contains("不是 0/1"), "实际错误: {err:#}");
    }

    #[test]
    fn skip_empty_episodes() {
        let ep = GameEpisode {
            samples: Vec::new(),
            game_length: 12,
            winner: None,
            health_diff_red: None,
            nnue: None,
        };
        let raw = encode_episode_batch("4x4", &[ep], &[]).unwrap();
        let batch = EpisodeBatch::decode(raw.as_slice()).unwrap();
        assert!(batch.episodes.is_empty());
    }
}
