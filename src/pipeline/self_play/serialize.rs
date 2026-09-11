// src/pipeline/self_play/serialize.rs
//
// 自 rust_4x8/src/pipeline/self_play/serialize.rs 迁入（banqi-collector 拆分）。
//
// 将 `GameEpisode` 序列化为与 Python 侧 `episode_to_samples` / PyO3
// `episode_to_dict` 字段契约一致的 JSON 对象，用于 gRPC 跨语言传输。
//
// 关键约定（与原 bridge/python/episode.rs 的 episode_to_dict_with_shapes 对齐）：
// - boards      : 每步棋盘特征，展平为一维数组 [C*H*W]（Python 侧再 reshape）
// - scalars     : 每步标量特征 [scalar_count]
// - policies    : 每步策略概率 [action_space]
// - action_masks: 每步动作掩码 [action_space]
// - 其余标量字段均为每步对齐的并列数组。

use serde_json::{json, Value};

use super::{GameEpisode, NnueEpisode};

/// 将 `GameEpisode` 序列化为与 PyO3 `episode_to_dict` 字段一致的 JSON 对象。
///
/// 从首个样本的观测动态推导 shape（board/scalar），从 policy 长度推导 action_space，
/// 从而无需硬编码变体常量，可服务任意 `GameEnv`。空样本局（无任何中间状态）
/// 时返回带空数组的 dict，不会 panic。
pub fn episode_to_dict_json(episode: &GameEpisode) -> Value {
    let n = episode.samples.len();

    // 从首个样本推导 shape（空局时回退为全 0，Python 侧无样本可消费，不影响）
    let (board_shape, scalar_shape, action_space) = match episode.samples.first() {
        Some((obs, policy, ..)) => {
            let bs: Vec<usize> = obs.board.shape().to_vec();
            let ss = vec![obs.scalars.len()];
            let ac = policy.len();
            (bs, ss, ac)
        }
        None => (vec![0, 0, 0], vec![0], 0),
    };

    let mut boards: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut scalars: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut policies: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut mcts_values: Vec<f32> = Vec::with_capacity(n);
    let mut completed_qs: Vec<f32> = Vec::with_capacity(n);
    let mut root_visits: Vec<u32> = Vec::with_capacity(n);
    let mut game_results: Vec<f32> = Vec::with_capacity(n);
    let mut action_masks: Vec<Vec<i32>> = Vec::with_capacity(n);
    let mut actions: Vec<usize> = Vec::with_capacity(n);
    let mut health_diffs: Vec<f32> = Vec::with_capacity(n);
    let mut is_full_searches: Vec<bool> = Vec::with_capacity(n);

    for (obs, policy, mcts_val, completed_q, root_visit, game_result, mask, action, health_diff, is_full_search) in &episode.samples {
        boards.push(obs.board.as_slice().unwrap().to_vec());
        scalars.push(obs.scalars.as_slice().unwrap().to_vec());
        policies.push(policy.clone());
        mcts_values.push(*mcts_val);
        completed_qs.push(*completed_q);
        root_visits.push(*root_visit);
        game_results.push(*game_result);
        action_masks.push(mask.clone());
        actions.push(*action);
        health_diffs.push(*health_diff);
        is_full_searches.push(*is_full_search);
    }

    let mut dict = json!({
        "game_length": episode.game_length,
        "winner": episode.winner,
        "num_samples": n,
        "boards": boards,
        "scalars": scalars,
        "policies": policies,
        "mcts_values": mcts_values,
        "completed_qs": completed_qs,
        "root_visits": root_visits,
        "game_results": game_results,
        "health_diffs": health_diffs,
        "action_masks": action_masks,
        "actions": actions,
        "is_full_search": is_full_searches,
        "health_diff_red": episode.health_diff_red,
        "board_shape": board_shape,
        "scalar_shape": scalar_shape,
        "action_space": action_space,
    });

    // NNUE 稀疏特征（可选字段：仅在收集开启时存在，旧数据流不受影响）
    if let Some((meta, feats)) = &episode.nnue {
        dict["nnue_meta"] = json!({
            "feature_dim": meta.feature_dim,
            "states_per_square": meta.states_per_square,
            "bag_stride": meta.bag_stride,
            "num_active": meta.num_active,
            "total_positions": meta.total_positions,
        });
        dict["nnue_features"] = json!({
            "mover": feats.iter().map(|f| &f.mover).collect::<Vec<_>>(),
            "opponent": feats.iter().map(|f| &f.opponent).collect::<Vec<_>>(),
        });
    }

    dict
}

/// 将 `NnueEpisode` 序列化为与 `NnueSampleDataset`（python/banqi/nnue/train.py）
/// 消费契约一致的 JSON 对象：字段名沿用 episode dict（`nnue_meta` /
/// `nnue_features` / `completed_qs` / `game_results` / `is_full_search`），
/// Expectimax 自对弈数据因此可被现有训练脚本直接加载。
pub fn nnue_episode_to_dict_json(episode: &NnueEpisode) -> Value {
    let n = episode.num_steps();
    let results = episode.game_results();
    json!({
        "game_length": episode.game_length,
        "winner": episode.winner,
        "num_samples": n,
        "nnue_meta": {
            "feature_dim": episode.meta.feature_dim,
            "states_per_square": episode.meta.states_per_square,
            "bag_stride": episode.meta.bag_stride,
            "num_active": episode.meta.num_active,
            "total_positions": episode.meta.total_positions,
        },
        "nnue_features": {
            "mover": episode.features.iter().map(|f| &f.mover).collect::<Vec<_>>(),
            "opponent": episode.features.iter().map(|f| &f.opponent).collect::<Vec<_>>(),
        },
        // Expectimax 根值作为搜索价值来源（兼容训练脚本 value_source=completed_q）
        "completed_qs": episode.search_values,
        "mcts_values": episode.search_values,
        "game_results": results,
        "actions": episode.actions,
        "is_full_search": vec![true; n],
    })
}

/// 序列化为单行 JSONL 文本（不含换行符）。
pub fn nnue_episode_to_jsonl(episode: &NnueEpisode) -> String {
    nnue_episode_to_dict_json(episode).to_string()
}
