// src/pipeline/reanalysis.rs — 跨进程局面重搜（reanalysis）执行端
//
// 用**当前（更强的）网络**对历史局面重跑 MCTS，产出新的策略 / 价值目标。这是「样本效率」
// 类改造：同一条自对弈记录随网络变强被反复榨取，且目标质量单调上升（KataGo 的 reanalysis）。
//
// 数据流（跨进程）：
//   自对弈（collect_positions=true）→ episode 携带每步局面快照
//   → 训练侧保留一批历史局面 → 经调度器下发重搜任务 → 本模块执行 → 结果走既有 episode 上报通道
//
// 目标语义（与训练端零改动对齐）：
//   - policy / mcts_value / completed_q 用**本次重搜**的产出（这是重搜的全部价值）；
//   - game_result / health_diff **沿用原局真值**（载荷携带），因此 `VALUE_TARGET_MODE`
//     无论取 game / game_hp / mcts / completed_q 都不需要为这批数据做特殊处理。
//
// 产物形态：**一个局面一条 episode（1 样本）**。原因是契约限制 ——
// `finalize_episode` 的 winner 是整局一个，多个不同对局的局面混在一局里无法各自回填结果。

use rayon::prelude::*;

use banqi_core::core::env::{GameEnv, PositionSnapshot, SnapshotEnv};
use banqi_core::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};

use super::SelfPlayConfig;
use super::finalize_episode;
use super::types::GameEpisode;

/// 日志中最多打印多少条失败原因（避免一批脏载荷刷屏）。
const MAX_LOGGED_ERRORS: usize = 5;

/// 一条待重搜项：局面快照 + 该局面所属对局的终局信息。
#[derive(Debug, Clone)]
pub struct ReanalysisItem {
    /// 局面快照（`PositionSnapshot::encode` 的字节串）。
    pub snapshot: Vec<u8>,
    /// 原局结果（红方视角：1 红胜 / -1 黑胜 / 0 平 / None 作废）。用于回填样本的 game_result。
    pub winner: Option<i32>,
    /// 原局终局归一化血量差（红方视角）。用于回填样本的 health_diff。
    pub health_diff_red: Option<f32>,
}

/// 重搜结果。
pub struct ReanalysisReport {
    /// 产出的一局面一 episode（顺序与输入对应，被跳过/失败的项不出现）。
    pub episodes: Vec<GameEpisode>,
    /// 载荷非法或推理失败的数量（不产出数据，计入日志）。
    pub failed: usize,
    /// 局面本身无可走动作（已终局）而跳过的数量。
    pub skipped: usize,
}

/// 对 `items` 逐个用 `evaluator` 重跑 MCTS（`config.mcts_sims` 次模拟），返回可上报的 episode。
///
/// - `G` 由调用方按任务变体分派（`G::from_snapshot` 会校验快照变体与 G 一致）；
/// - 局面之间互不依赖，故按 `thread_pool` 并行；每局面一棵新树（不复用）。
pub fn run_reanalysis<G, E>(
    items: &[ReanalysisItem],
    evaluator: &E,
    config: &SelfPlayConfig,
    thread_pool: Option<&rayon::ThreadPool>,
    tag: &str,
) -> ReanalysisReport
where
    G: GameEnv + SnapshotEnv + Send + Sync + 'static,
    E: Evaluator<G> + Sync,
{
    let gumbel_cfg = GumbelConfig {
        num_simulations: config.mcts_sims,
        max_considered_actions: config.max_considered_actions,
        c_scale: config.c_scale,
        gumbel_scale: config.gumbel_scale,
        health_enabled: config.health_enabled,
        health_weight: config.health_weight,
        health_confidence_exp: config.health_confidence_exp,
    };

    // Ok(Some(ep)) 有产出 / Ok(None) 局面已终局 / Err 载荷或推理失败
    let run_one = |item: &ReanalysisItem| -> Result<Option<GameEpisode>, String> {
        let snapshot = PositionSnapshot::decode(&item.snapshot)
            .map_err(|e| format!("快照解码失败: {e}"))?;
        let env = G::from_snapshot(&snapshot).map_err(|e| format!("快照重建失败: {e}"))?;

        let mut mcts = GumbelMCTS::new(&env, evaluator, gumbel_cfg.clone());
        match mcts.run() {
            Ok(Some(r)) => {
                let data = vec![(
                    r.state,
                    r.improved_policy,
                    r.mcts_value,
                    r.completed_q,
                    r.root_visit_count,
                    r.player,
                    r.action_mask,
                    r.action,
                    true, // 重搜一律为 Full Search，训练侧据此参与 loss
                )];
                // winner / health_diff 沿用原局真值；positions 留空（无需再采集）
                Ok(Some(finalize_episode(data, item.winner, item.health_diff_red, None)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(format!("推理失败: {e}")),
        }
    };

    let results: Vec<Result<Option<GameEpisode>, String>> = match thread_pool {
        Some(pool) => pool.install(|| items.par_iter().map(run_one).collect()),
        None => items.iter().map(run_one).collect(),
    };

    let mut episodes = Vec::with_capacity(results.len());
    let mut failed = 0usize;
    let mut skipped = 0usize;
    let mut logged = 0usize;
    for r in results {
        match r {
            Ok(Some(ep)) => episodes.push(ep),
            Ok(None) => skipped += 1,
            Err(e) => {
                failed += 1;
                if logged < MAX_LOGGED_ERRORS {
                    logged += 1;
                    eprintln!("⚠️ [{tag}] 重搜失败: {e}");
                }
            }
        }
    }
    if failed > MAX_LOGGED_ERRORS {
        eprintln!("⚠️ [{tag}] 另有 {} 条重搜失败未列出", failed - MAX_LOGGED_ERRORS);
    }

    ReanalysisReport {
        episodes,
        failed,
        skipped,
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use banqi_core::core::env::seed::SeedableEnv;
    use banqi_core::core::env::variants::MINI_RESNET_BOARD_CHANNELS;
    use banqi_core::core::env::{MiniDarkChessEnv, Player};
    use banqi_core::core::mcts::{EvaluatorError, EvaluatorOutput};

    /// 均匀先验评估器（价值 0）：只验证重搜流程与产物契约，不依赖任何模型。
    struct UniformEval;

    impl<G: GameEnv> Evaluator<G> for UniformEval {
        fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
            Ok(EvaluatorOutput {
                logits: envs.iter().map(|e| vec![0.0f32; e.action_space_size()]).collect(),
                values: vec![0.0; envs.len()],
                health: None,
            })
        }
    }

    fn config() -> SelfPlayConfig {
        SelfPlayConfig {
            mcts_sims: 16,
            max_considered_actions: 4,
            ..Default::default()
        }
    }

    /// 走一局 4x2（固定种子）并收集每步快照。
    fn collect_positions() -> Vec<Vec<u8>> {
        let mut env = MiniDarkChessEnv::new();
        env.set_seed(20260916);
        let mut masks = vec![0i32; env.action_space_size()];
        let mut snapshots = Vec::new();
        for step in 0..6 {
            snapshots.push(SnapshotEnv::to_snapshot(&env).encode());
            env.action_masks_into(&mut masks);
            let legal: Vec<usize> = masks
                .iter()
                .enumerate()
                .filter(|(_, m)| **m == 1)
                .map(|(i, _)| i)
                .collect();
            if legal.is_empty() {
                break;
            }
            if env.step(legal[step % legal.len()]).is_err() {
                break;
            }
        }
        snapshots
    }

    #[test]
    fn reanalysis_produces_one_sample_episode_per_position() {
        let snapshots = collect_positions();
        assert!(!snapshots.is_empty(), "未收集到局面快照");

        // 假设原局红胜、终局血量差 +0.5：产物的 game_result / health_diff 必须按行棋方视角回填
        let items: Vec<ReanalysisItem> = snapshots
            .iter()
            .map(|s| ReanalysisItem {
                snapshot: s.clone(),
                winner: Some(1),
                health_diff_red: Some(0.5),
            })
            .collect();

        let report = run_reanalysis::<MiniDarkChessEnv, _>(
            &items,
            &UniformEval,
            &config(),
            None,
            "test",
        );

        assert_eq!(report.failed, 0, "均匀评估器不应产生失败");
        assert_eq!(report.episodes.len(), items.len(), "每个局面应产出一条 episode");
        assert_eq!(report.skipped, 0, "开局若干步不应出现无合法动作的局面");

        for (ep, item) in report.episodes.iter().zip(&items) {
            assert_eq!(ep.samples.len(), 1, "重搜 episode 应为 1 样本");
            assert_eq!(ep.game_length, 1);
            assert_eq!(ep.winner, item.winner, "winner 应沿用原局真值");
            assert_eq!(ep.health_diff_red, item.health_diff_red);
            assert!(ep.positions.is_none(), "重搜产物无需再采集快照");

            let (obs, policy, _mcts_value, _completed_q, _root_visit, game_result, mask, action, health_diff, is_full) =
                &ep.samples[0];
            assert!(*is_full, "重搜样本应标记为 Full Search");
            assert_eq!(mask.len(), policy.len(), "动作掩码长度应等于动作空间");
            assert_eq!(
                obs.board.shape()[0],
                MINI_RESNET_BOARD_CHANNELS,
                "棋盘通道数应为 4x2 变体的特征通道数"
            );

            // 行棋方视角回填：红方行棋 → +1 / +0.5；黑方行棋 → -1 / -0.5
            let snap = PositionSnapshot::decode(&item.snapshot).expect("快照解码失败");
            let sign = if snap.current_player == Player::Red { 1.0 } else { -1.0 };
            assert_eq!(*game_result, sign, "game_result 视角不符");
            assert_eq!(*health_diff, sign * 0.5, "health_diff 视角不符");

            // 策略分布合法（重搜产出的软分布）
            let sum: f32 = policy.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "策略概率和 {sum} 不为 1");
            assert!(*action < policy.len(), "动作索引越界");
        }
    }

    #[test]
    fn invalid_payload_is_counted_not_panicking() {
        let items = vec![
            ReanalysisItem {
                snapshot: vec![0xff; 8], // 版本/变体/长度全非法
                winner: Some(0),
                health_diff_red: None,
            },
            ReanalysisItem {
                snapshot: Vec::new(),
                winner: None,
                health_diff_red: None,
            },
        ];
        let report = run_reanalysis::<MiniDarkChessEnv, _>(
            &items,
            &UniformEval,
            &config(),
            None,
            "test",
        );
        assert_eq!(report.failed, 2, "非法载荷应全部计入 failed");
        assert!(report.episodes.is_empty());
    }
}
