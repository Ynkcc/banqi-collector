// src/pipeline/self_play/finalize.rs - 自对弈辅助函数
//
// 自 rust_4x8/src/pipeline/self_play/finalize.rs 迁入（banqi-collector 拆分），
// 样本回填、动作选择等纯计算工具。

use banqi_core::core::env::{GameEnv, ResNetObservation, Player};
use banqi_core::core::mcts::{Evaluator, GumbelMCTS};

/// 按 winner 统一回填 episode_data 并构造 GameEpisode。
///
/// - reward_red：winner=Some(1) → 1.0，Some(-1) → -1.0，None/Some(0) → 0.0；
/// - 每个样本按该样本玩家的视角换算 game_result（红方视角为正）；
/// - game_length 统一为「已完成步数」= 样本数，消除各终止路径语义差 1 的不一致。
///
/// 该函数同时被三条终止路径调用：MCTS None 分支（无合法走法判负）、
/// 终局分支（terminated/truncated）、步数上限分支。
pub fn finalize_episode(
    episode_data: Vec<(ResNetObservation, Vec<f32>, f32, f32, u32, Player, Vec<i32>, usize, bool)>,
    winner: Option<i32>,
    health_diff_red: Option<f32>,
) -> crate::pipeline::self_play::GameEpisode {
    let game_length = episode_data.len();
    let reward_red: f32 = match winner {
        Some(1) => 1.0,
        Some(-1) => -1.0,
        _ => 0.0,
    };
    let samples = episode_data
        .into_iter()
        .map(|(obs, p, mcts_val, completed_q, root_visit_count, player, mask, action, is_full_search)| {
            let game_result_val: f32 = if player.val() == 1 {
                reward_red
            } else {
                -reward_red
            };
            // 血量差与 game_result 一致：按该样本玩家的视角取号（红方视角为正）。
            let health_diff_val: f32 = match health_diff_red {
                Some(d) if player.val() == 1 => d,
                Some(d) => -d,
                None => 0.0,
            };
            (
                obs,
                p,
                mcts_val,
                completed_q,
                root_visit_count,
                game_result_val,
                mask,
                action,
                health_diff_val,
                is_full_search,
            )
        })
        .collect();
    crate::pipeline::self_play::GameEpisode {
        samples,
        game_length,
        winner,
        health_diff_red,
    }
}

/// 选择 completed_Q 最大的动作（确定性）
pub fn select_completed_q_action<G: GameEnv, E: Evaluator<G>>(
    mcts: &GumbelMCTS<G, E>,
    masks: &[i32],
) -> (usize, f32) {
    let mut best_action: Option<usize> = None;
    let mut best_completed_q = f32::NEG_INFINITY;

    for (action, &mask) in masks.iter().enumerate() {
        if mask != 1 {
            continue;
        }
        let completed_q = mcts.get_root_completed_q(action);
        if completed_q > best_completed_q {
            best_completed_q = completed_q;
            best_action = Some(action);
        }
    }

    let action = best_action.expect("无有效动作");
    (action, best_completed_q)
}

/// 获取 Top-K 动作 (用于调试)
pub fn get_top_k_actions(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
}
