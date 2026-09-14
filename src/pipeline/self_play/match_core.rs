// src/pipeline/self_play/match_core.rs
// 统一对局主干引擎（Unified Match Core Engine）。
//
// 自 rust_4x8/src/pipeline/self_play/match_core.rs 迁入（banqi-collector 拆分），
// 移除 PyO3（PyPredictor）分支——本 crate 仅服务分布式训练采集链路。
//
// 设计要点：
// - `PlayerSpec`：统一选手抽象（模型 / Expectimax / 随机）。
// - `run_match_core`：统一主干，支持固定 Seed、记录 Episode 或仅收集胜负统计。
// - `AsDarkChessRef` / `SeedableEnv` 从 banqi-core 重导出，供规则选手
//   取底层棋盘与设置种子。

use std::sync::Arc;

use rayon::prelude::*;

use banqi_core::core::env::{DarkChessEnv, GameEnv};
use banqi_core::core::expectimax::ExpectimaxEngine;
use banqi_core::core::mcts::{
    Evaluator, EvaluatorError, EvaluatorOutput, GumbelConfig, GumbelMCTS,
};

use super::{finalize_episode, GameEpisode, SelfPlayConfig};

// ============================================================================
// 辅助 Trait：提取 &DarkChessEnv 与种子设置
// （实现已随领域核心下沉至 banqi-core::core::env::seed，此处重导出保持路径不变）
// ============================================================================

pub use banqi_core::core::env::seed::{AsDarkChessRef, SeedableEnv};

/// 从收集到的每步特征与 env.config 组装 `GameEpisode.nnue` 字段。
/// 未启用收集时返回 None；env 仅用于读取 config 推导布局元信息。
fn nnue_meta_and_features(
    env: &DarkChessEnv,
    features: Option<Vec<super::NnueStepFeatures>>,
) -> Option<(super::NnueEpisodeMeta, Vec<super::NnueStepFeatures>)> {
    let feats = features?;
    let cfg = &env.config;
    let meta = super::NnueEpisodeMeta {
        feature_dim: cfg.nnue_feature_dim(),
        states_per_square: cfg.nnue_states_per_square(),
        bag_stride: cfg.nnue_bag_stride(),
        num_active: cfg.num_active,
        total_positions: cfg.total_positions,
    };
    Some((meta, feats))
}

// ============================================================================
// 统一选手抽象 PlayerSpec
// ============================================================================

/// 统一选手抽象：支持模型 / Expectimax+NNUE / 随机任意组合。
pub enum PlayerSpec<G: GameEnv> {
    /// Expectimax + NNUE 引擎（独立 DFS 决策，不经 Gumbel MCTS）。
    Expectimax(Arc<ExpectimaxEngine>),
    /// Rust 侧持有 .pt / .onnx 模型的评估器（经 Gumbel MCTS 决策）。
    ModelEval(Arc<dyn Evaluator<G> + Send + Sync>),
    /// 纯策略：直接取 policy head 在合法动作上的 argmax，不做任何搜索。
    PolicyArgmax(Arc<dyn Evaluator<G> + Send + Sync>),
    Random,
}

// ============================================================================
// 随机评估器（供记录 Episode 的 MCTS 自对弈路径使用）
// ============================================================================

/// 随机评估器：对所有合法动作给均匀先验。
struct RandomEval;

impl<G: GameEnv> Evaluator<G> for RandomEval {
    fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        let mut logits = Vec::with_capacity(envs.len());
        let mut values = Vec::with_capacity(envs.len());
        for env in envs {
            let action_space = env.action_space_size();
            let mut lg = vec![0.0f32; action_space];
            let mut masks = vec![0i32; action_space];
            env.action_masks_into(&mut masks);
            for (i, &m) in masks.iter().enumerate() {
                if m == 1 {
                    lg[i] = 1.0;
                }
            }
            logits.push(lg);
            values.push(0.0);
        }
        Ok(EvaluatorOutput { logits, values, health: None })
    }
}

/// 具体（Sized）评估器包装：把 `PlayerSpec` 统一为可驱动 Gumbel MCTS 的评估器。
enum PlayerEval<G: GameEnv + AsDarkChessRef> {
    Model(Arc<dyn Evaluator<G> + Send + Sync>),
    Random(RandomEval),
}

impl<G: GameEnv + AsDarkChessRef + Sync> Evaluator<G> for PlayerEval<G> {
    fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        match self {
            PlayerEval::Model(e) => e.evaluate(envs),
            PlayerEval::Random(e) => e.evaluate(envs),
        }
    }
}

/// 把 `PlayerSpec` 转成具体评估器（供记录 Episode 的 MCTS 路径）。
fn make_evaluator<G>(spec: &PlayerSpec<G>) -> PlayerEval<G>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    match spec {
        PlayerSpec::ModelEval(e) => PlayerEval::Model(e.clone()),
        PlayerSpec::PolicyArgmax(e) => PlayerEval::Model(e.clone()),
        PlayerSpec::Random => PlayerEval::Random(RandomEval),
        PlayerSpec::Expectimax(_) => {
            unreachable!("Expectimax 选手不经 make_evaluator / MCTS 路径（应在调用处分流）")
        }
    }
}

// ============================================================================
// 动作选择（非记录 / 评估路径）
// ============================================================================

/// 用 Gumbel MCTS 驱动的模型动作选择。
fn model_mcts_action<G>(env: &G, evaluator: &PlayerEval<G>, sims: usize) -> Option<usize>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    let config = GumbelConfig {
        num_simulations: sims,
        max_considered_actions: 16,
        c_scale: 0.25,
        gumbel_scale: 1.0,
        ..Default::default()
    };
    let mut mcts = GumbelMCTS::new(env, evaluator, config);
    match mcts.run() {
        Ok(Some(r)) => Some(r.action),
        Ok(None) => None,
        Err(e) => {
            eprintln!("❌ 评估失败（模型推理），本步无动作: {e}");
            None
        }
    }
}

/// 纯策略动作选择：对 policy head logits 做合法动作掩码后取 argmax，不做任何搜索。
/// 非法动作的 logits 一律忽略（不依赖评估器是否已置 -inf），不可用值同样跳过；
/// 全部非法/非有限时返回 None，由上层按对局终止处理。
fn policy_argmax_action<G>(env: &G, evaluator: &Arc<dyn Evaluator<G> + Send + Sync>) -> Option<usize>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    let out = match evaluator.evaluate(std::slice::from_ref(env)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("❌ 评估失败（policy argmax），本步无动作: {e}");
            return None;
        }
    };
    let logits = out.logits.first()?;
    let mut masks = vec![0i32; env.action_space_size()];
    env.action_masks_into(&mut masks);
    let mut best: Option<(usize, f32)> = None;
    for (i, &lg) in logits.iter().enumerate() {
        if masks.get(i).copied().unwrap_or(0) != 1 || !lg.is_finite() {
            continue;
        }
        if best.map_or(true, |(_, bv)| lg > bv) {
            best = Some((i, lg));
        }
    }
    best.map(|(a, _)| a)
}

fn random_action<G: GameEnv>(env: &G) -> Option<usize> {
    let mut masks = vec![0i32; env.action_space_size()];
    env.action_masks_into(&mut masks);
    let legal: Vec<usize> = masks
        .iter()
        .enumerate()
        .filter(|(_, m)| **m == 1)
        .map(|(i, _)| i)
        .collect();
    if legal.is_empty() {
        return None;
    }
    use rand::Rng;
    Some(legal[rand::thread_rng().gen_range(0..legal.len())])
}

/// 按选手类型选择一步动作（评估路径，不记录 Episode）。
fn get_player_action<G>(env: &G, spec: &PlayerSpec<G>, model_sims: usize) -> Option<usize>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    match spec {
        PlayerSpec::Expectimax(e) => {
            e.search_par(env.as_darkchess_ref()).map(|r| r.action)
        }
        PlayerSpec::ModelEval(e) => model_mcts_action(env, &PlayerEval::Model(e.clone()), model_sims),
        PlayerSpec::PolicyArgmax(e) => policy_argmax_action(env, e),
        PlayerSpec::Random => random_action(env),
    }
}

// ============================================================================
// 单局驱动
// ============================================================================

/// 单局结果（从选手 A 视角）。`episode` 仅在 MCTS 记录模式下 Some；
/// `nnue_episode` 仅在 Expectimax 记录模式下 Some。
pub(crate) struct GameOutcome {
    pub(crate) result: i32,
    pub(crate) moves: usize,
    pub(crate) episode: Option<GameEpisode>,
    pub(crate) nnue_episode: Option<super::NnueEpisode>,
}

/// 非记录（评估）路径：A vs B 直接推进，不生成 Episode。
#[allow(clippy::too_many_arguments)]
fn play_one_game<G>(
    player_a_spec: &PlayerSpec<G>,
    player_b_spec: &PlayerSpec<G>,
    player_a_is_red: bool,
    model_sims: usize,
    game_seed: Option<u64>,
    make_env: &Arc<dyn Fn() -> G + Send + Sync>,
) -> GameOutcome
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default,
{
    let mut env = make_env();
    if let Some(s) = game_seed {
        env.set_seed(s);
    }
    let mut moves = 0;
    let max_moves = env.max_steps();

    while !env.check_game_over_conditions().0 {
        if env.check_game_over_conditions().2.is_some() {
            break;
        }
        let cur = env.get_current_player().val();
        let is_a_turn = (cur == 1) == player_a_is_red;

        let action = if is_a_turn {
            get_player_action(&env, player_a_spec, model_sims)
        } else {
            get_player_action(&env, player_b_spec, model_sims)
        };

        let Some(a) = action else {
            break;
        };

        if GameEnv::step(&mut env, a).is_err() {
            break;
        }

        moves += 1;
        if moves >= max_moves {
            break;
        }
    }

    let winner = env.check_game_over_conditions().2;
    let r = match winner {
        Some(1) => {
            if player_a_is_red { 1 } else { -1 }
        }
        Some(-1) => {
            if player_a_is_red { -1 } else { 1 }
        }
        _ => 0,
    };
    GameOutcome {
        result: r,
        moves,
        episode: None,
        nnue_episode: None,
    }
}

/// Expectimax 记录（自对弈）路径：每步由 `engine.search_par` 直接决策，
/// 仅收集 NNUE 训练所需稀疏特征 + 搜索标量，产出 [`super::NnueEpisode`]。
pub(crate) fn play_one_game_expectimax<G>(
    engine: &ExpectimaxEngine,
    player_a_is_red: bool,
    game_seed: Option<u64>,
    make_env: &Arc<dyn Fn() -> G + Send + Sync>,
) -> GameOutcome
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default,
{
    let mut env = make_env();
    if let Some(s) = game_seed {
        env.set_seed(s);
    }

    let meta = {
        let cfg = &env.as_darkchess_ref().config;
        super::NnueEpisodeMeta {
            feature_dim: cfg.nnue_feature_dim(),
            states_per_square: cfg.nnue_states_per_square(),
            bag_stride: cfg.nnue_bag_stride(),
            num_active: cfg.num_active,
            total_positions: cfg.total_positions,
        }
    };

    let mut features = Vec::new();
    let mut search_values = Vec::new();
    let mut players = Vec::new();
    let mut actions = Vec::new();
    let mut step = 0usize;

    fn finish(
        meta: super::NnueEpisodeMeta,
        winner: Option<i32>,
        moves: usize,
        player_a_is_red: bool,
        features: Vec<super::NnueStepFeatures>,
        search_values: Vec<f32>,
        players: Vec<i32>,
        actions: Vec<usize>,
    ) -> (super::NnueEpisode, i32) {
        let ep = super::NnueEpisode {
            meta,
            features,
            search_values,
            players,
            actions,
            game_length: moves,
            winner,
        };
        let r = match winner {
            Some(1) => {
                if player_a_is_red {
                    1
                } else {
                    -1
                }
            }
            Some(-1) => {
                if player_a_is_red {
                    -1
                } else {
                    1
                }
            }
            _ => 0,
        };
        (ep, r)
    }

    loop {
        let cur = env.get_current_player().val();
        let d = env.as_darkchess_ref();

        // NNUE 特征收集：须在 step 前取当前局面的行棋方/对方双视角索引
        let mover_player = d.get_current_player();
        let mut mover = Vec::with_capacity(d.config.total_positions + 16);
        let mut opponent = Vec::with_capacity(d.config.total_positions + 16);
        d.nnue_active_features_for_player_into(mover_player, &mut mover);
        d.nnue_active_features_for_player_into(mover_player.opposite(), &mut opponent);
        features.push(super::NnueStepFeatures {
            mover: mover.into_iter().map(|i| i as u16).collect(),
            opponent: opponent.into_iter().map(|i| i as u16).collect(),
        });

        let Some(result) = engine.search_par(d) else {
            let (_, _, winner) = env.check_game_over_conditions();
            let (ep, r) = finish(meta, winner, step, player_a_is_red, features, search_values, players, actions);
            return GameOutcome { result: r, moves: step, episode: None, nnue_episode: Some(ep) };
        };

        search_values.push(result.value);
        players.push(cur);
        actions.push(result.action);

        match env.step(result.action) {
            Ok((_, terminated, truncated, winner)) => {
                if terminated || truncated {
                    let (ep, r) = finish(meta, winner, step + 1, player_a_is_red, features, search_values, players, actions);
                    return GameOutcome { result: r, moves: step + 1, episode: None, nnue_episode: Some(ep) };
                }
            }
            Err(e) => {
                eprintln!("⚠️ Expectimax 记录对局 step 错误: {e}");
                let (ep, r) = finish(meta, None, step, player_a_is_red, features, search_values, players, actions);
                return GameOutcome { result: r, moves: step, episode: None, nnue_episode: Some(ep) };
            }
        }

        step += 1;
        if step >= env.max_steps() {
            let (ep, r) = finish(meta, Some(0), step, player_a_is_red, features, search_values, players, actions);
            return GameOutcome { result: r, moves: step, episode: None, nnue_episode: Some(ep) };
        }
    }
}

/// 记录（自对弈）路径：每步用当前选手的评估器驱动 Gumbel MCTS，收集完整 Episode。
/// 双方均为 `PlayerSpec::Expectimax` 时走 Expectimax 专属 NNUE 记录循环。
fn play_one_game_recorded<G>(
    player_a_spec: &PlayerSpec<G>,
    player_b_spec: &PlayerSpec<G>,
    player_a_is_red: bool,
    config: &SelfPlayConfig,
    game_seed: Option<u64>,
    make_env: &Arc<dyn Fn() -> G + Send + Sync>,
) -> GameOutcome
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default,
{
    // 双方均为 Expectimax：走 NNUE 专属记录循环（Expectimax 为 DFS 串行搜索，
    // 无 improved_policy / completed_q / 访问计数等 MCTS 概念，不产出 GameEpisode）。
    if let (PlayerSpec::Expectimax(ea), PlayerSpec::Expectimax(eb)) = (player_a_spec, player_b_spec) {
        let engine: &ExpectimaxEngine = if player_a_is_red { ea } else { eb };
        return play_one_game_expectimax::<G>(engine, player_a_is_red, game_seed, make_env);
    }
    if matches!(player_a_spec, PlayerSpec::Expectimax(_)) || matches!(player_b_spec, PlayerSpec::Expectimax(_)) {
        eprintln!("❌ 记录（自对弈）路径仅支持 Expectimax vs Expectimax 或双方均非 Expectimax 的组合");
        return GameOutcome { result: 0, moves: 0, episode: None, nnue_episode: None };
    }

    let mut env = make_env();
    if let Some(s) = game_seed {
        env.set_seed(s);
    }

    let gumbel_cfg = GumbelConfig {
        num_simulations: config.mcts_sims,
        max_considered_actions: config.max_considered_actions,
        c_scale: config.c_scale,
        gumbel_scale: config.gumbel_scale,
        health_enabled: config.health_enabled,
        health_weight: config.health_weight,
        health_confidence_exp: config.health_confidence_exp,
    };

    let mut episode_data = Vec::new();
    let mut nnue_features = if config.collect_nnue_features {
        Some(Vec::new())
    } else {
        None
    };
    let mut step = 0;

    let eval_a = make_evaluator(player_a_spec);
    let eval_b = make_evaluator(player_b_spec);

    loop {
        let cur = env.get_current_player().val();
        let is_a_turn = (cur == 1) == player_a_is_red;
        let evaluator = if is_a_turn { &eval_a } else { &eval_b };

        let is_full_search = if config.playout_cap_random_enabled {
            if step == 0 {
                true
            } else {
                rand::random::<f32>() < config.full_search_prob
            }
        } else {
            true
        };
        let step_sims = if is_full_search {
            config.mcts_sims
        } else {
            config.fast_mcts_sims
        };

        let mut step_gumbel_cfg = gumbel_cfg.clone();
        step_gumbel_cfg.num_simulations = step_sims;

        let mut mcts = GumbelMCTS::new(&env, evaluator, step_gumbel_cfg);
        let search_result = match mcts.run() {
            Ok(Some(r)) => r,
            Ok(None) => {
                let (_, _, winner) = env.check_game_over_conditions();
                let ep = finalize_episode(episode_data, winner, env.terminal_health_diff_red(), nnue_meta_and_features(env.as_darkchess_ref(), nnue_features));
                return outcome_from_episode(ep, player_a_is_red);
            }
            Err(e) => {
                // 推理失败：本局作废（不产出 episode），由 bin 按「0 局产出」报错，
                // 避免把无效局面写进训练数据。
                eprintln!("❌ 自对弈评估失败，本局作废: {e}");
                return GameOutcome { result: 0, moves: step, episode: None, nnue_episode: None };
            }
        };

        // NNUE 特征收集：须在 step 前取当前局面的行棋方/对方双视角索引
        if let Some(feats) = &mut nnue_features {
            let d = env.as_darkchess_ref();
            let mover_player = d.get_current_player();
            let mut mover = Vec::with_capacity(d.config.total_positions + 16);
            let mut opponent = Vec::with_capacity(d.config.total_positions + 16);
            d.nnue_active_features_for_player_into(mover_player, &mut mover);
            d.nnue_active_features_for_player_into(mover_player.opposite(), &mut opponent);
            feats.push(super::NnueStepFeatures {
                mover: mover.into_iter().map(|i| i as u16).collect(),
                opponent: opponent.into_iter().map(|i| i as u16).collect(),
            });
        }

        // 落子：直接采用 Gumbel 搜索选出的动作。探索由每次搜索重新抽的 Gumbel
        // 噪声提供（sample_gumbel_top_k），无需根温度采样，详见 SelfPlayConfig。
        let action = search_result.action;
        let completed_q = search_result.completed_q;

        // 全记录：无论 Full/Fast 都收集样本，并标记 is_full_search，交由 Python 侧
        // 选择性使用（losses.py 仅让 Full Search 样本参与训练）。
        episode_data.push((
            search_result.state,
            search_result.improved_policy,
            search_result.mcts_value,
            completed_q,
            search_result.root_visit_count,
            search_result.player,
            search_result.action_mask,
            action,
            is_full_search,
        ));

        match env.step(action) {
            Ok((_, terminated, truncated, winner)) => {
                if terminated || truncated {
                    let ep = finalize_episode(episode_data, winner, env.terminal_health_diff_red(), nnue_meta_and_features(env.as_darkchess_ref(), nnue_features));
                    return outcome_from_episode(ep, player_a_is_red);
                }
            }
            Err(e) => {
                eprintln!("⚠️ 记录对局 step 错误: {e}");
                let ep = GameEpisode {
                    samples: Vec::new(),
                    game_length: step,
                    winner: None,
                    health_diff_red: env.terminal_health_diff_red(),
                    nnue: None,
                };
                return outcome_from_episode(ep, player_a_is_red);
            }
        }

        step += 1;
        if step >= env.max_steps() {
            let ep = finalize_episode(episode_data, Some(0), env.terminal_health_diff_red(), nnue_meta_and_features(env.as_darkchess_ref(), nnue_features));
            return outcome_from_episode(ep, player_a_is_red);
        }
    }
}

/// 从 Episode 推导选手 A 视角结果与步数。
fn outcome_from_episode(ep: GameEpisode, player_a_is_red: bool) -> GameOutcome {
    let moves = ep.game_length;
    let r = match ep.winner {
        Some(1) => {
            if player_a_is_red { 1 } else { -1 }
        }
        Some(-1) => {
            if player_a_is_red { -1 } else { 1 }
        }
        _ => 0,
    };
    GameOutcome {
        result: r,
        moves,
        episode: Some(ep),
        nnue_episode: None,
    }
}

// ============================================================================
// 统一主干 run_match_core
// ============================================================================

/// 多局比赛参数。
pub struct MatchParams<'a, G: GameEnv> {
    pub player_a: &'a PlayerSpec<G>,
    pub player_b: &'a PlayerSpec<G>,
    pub n_games: usize,
    pub config: &'a SelfPlayConfig,
    pub seed: Option<u64>,
    /// 是否记录完整 Episode（自对弈数据）；false 时仅收集胜负统计（评估）。
    pub record_episodes: bool,
    /// 模型选手的 MCTS 模拟数（评估路径）。
    pub model_sims: usize,
    /// 线程池：Some = 原生多线程（Rust 持模型 / 规则），None = 单线程。
    pub thread_pool: Option<&'a rayon::ThreadPool>,
    /// 环境工厂：每局调用一次（支持课程参数注入的闭包形式）。
    pub make_env: Arc<dyn Fn() -> G + Send + Sync>,
}

/// 多局比赛结果。
pub struct MatchResult {
    pub episodes: Vec<GameEpisode>,
    /// Expectimax 记录路径产出的 NNUE 专属 Episode（与 `episodes` 互斥出现）。
    pub nnue_episodes: Vec<super::NnueEpisode>,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    /// 逐局结果（A/player_a 视角：+1 胜 / 0 和 / -1 负），顺序与对局索引一致；
    /// rating 任务据此按成对（i 与 i+1 换色）推导五项计数。
    pub game_outcomes: Vec<i32>,
    pub block_wr: Vec<f32>,
    pub avg_moves: f32,
}

/// 统一对局主干：驱动 `n_games` 局 A vs B。
pub fn run_match_core<G>(params: MatchParams<'_, G>) -> MatchResult
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Send + Sync + Default + 'static,
{
    let n = params.n_games;
    let indices: Vec<usize> = (0..n).collect();

    let play = |i: usize| -> GameOutcome {
        let player_a_is_red = (i % 2) == 0;
        let game_seed = params.seed.map(|s| s.wrapping_add(i as u64));
        if params.record_episodes {
            play_one_game_recorded(
                params.player_a,
                params.player_b,
                player_a_is_red,
                params.config,
                game_seed,
                &params.make_env,
            )
        } else {
            play_one_game(
                params.player_a,
                params.player_b,
                player_a_is_red,
                params.model_sims,
                game_seed,
                &params.make_env,
            )
        }
    };

    let games: Vec<GameOutcome> = match params.thread_pool {
        Some(pool) => pool.install(|| indices.into_par_iter().map(play).collect()),
        None => indices.into_iter().map(play).collect(),
    };

    let mut wins = 0;
    let mut draws = 0;
    let mut losses = 0;
    let mut total_moves = 0;
    let mut episodes = Vec::new();
    let mut nnue_episodes = Vec::new();
    let block_size = 20;
    let mut block_wr = Vec::new();
    let mut blk_w = 0;
    let mut blk_tot = 0;

    for (i, g) in games.iter().enumerate() {
        total_moves += g.moves;
        if g.result == 1 {
            wins += 1;
            blk_w += 1;
        } else if g.result == -1 {
            losses += 1;
        } else {
            draws += 1;
        }
        blk_tot += 1;

        if (i + 1) % block_size == 0 {
            block_wr.push(100.0 * (blk_w as f32) / (blk_tot as f32));
            blk_w = 0;
            blk_tot = 0;
        }
        if let Some(ep) = &g.episode {
            if !ep.samples.is_empty() {
                episodes.push(ep.clone());
            }
        }
        if let Some(ep) = &g.nnue_episode {
            if !ep.features.is_empty() {
                nnue_episodes.push(ep.clone());
            }
        }
    }
    if blk_tot > 0 && n % block_size != 0 {
        block_wr.push(100.0 * (blk_w as f32) / (blk_tot as f32));
    }

    let avg_moves = (total_moves as f32) / (n.max(1) as f32);
    let game_outcomes = games.iter().map(|g| g.result).collect();
    MatchResult {
        episodes,
        nnue_episodes,
        wins,
        draws,
        losses,
        game_outcomes,
        block_wr,
        avg_moves,
    }
}
