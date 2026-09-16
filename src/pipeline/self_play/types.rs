// src/pipeline/self_play/types.rs - 自对弈数据结构与同步运行器
//
// 自 rust_4x8/src/pipeline/self_play/types.rs 迁入（banqi-collector 拆分），
// 领域核心依赖由 crate::core 改为 banqi_core::core。

use banqi_core::core::env::{GameEnv, ResNetObservation};
use banqi_core::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ================ 数据结构定义 ================

/// 单步 NNUE 稀疏特征（双视角）。
///
/// 索引编码与 `DarkChessEnv::nnue_active_features_for_player_into` 一致，
/// 布局参数由 `NnueEpisodeMeta` 描述（随变体推导，无硬编码维度）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NnueStepFeatures {
    /// 行棋方视角的活性特征索引
    pub mover: Vec<u16>,
    /// 对方视角的活性特征索引
    pub opponent: Vec<u16>,
}

/// NNUE 特征布局元信息（由 env.config 推导，供 Python 侧免硬编码还原维度）。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NnueEpisodeMeta {
    pub feature_dim: usize,
    pub states_per_square: usize,
    pub bag_stride: usize,
    pub num_active: usize,
    pub total_positions: usize,
}

/// Expectimax 自对弈产出的 NNUE 专属数据契约。
///
/// 仅含稀疏特征索引 + 搜索标量，无 `GameEpisode` 的 ResNetObservation 大张量；
/// `search_values` 为每步行棋方视角的 Expectimax 根值（最深完成迭代）。
/// 每步 `features` / `search_values` / `players` / `actions` 严格对齐。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NnueEpisode {
    pub meta: NnueEpisodeMeta,
    pub features: Vec<NnueStepFeatures>,
    pub search_values: Vec<f32>,
    /// 每步行棋方 (1=红, -1=黑)
    pub players: Vec<i32>,
    pub actions: Vec<usize>,
    pub game_length: usize,
    pub winner: Option<i32>,
}

impl NnueEpisode {
    /// 终局后按行棋方视角回填每步 game_result（红胜=+1 / 黑胜=-1 / 平=0）。
    pub fn game_results(&self) -> Vec<f32> {
        let r = match self.winner {
            Some(1) => 1.0,
            Some(-1) => -1.0,
            _ => 0.0,
        };
        self.players.iter().map(|&p| p as f32 * r).collect()
    }

    pub fn num_steps(&self) -> usize {
        self.features.len()
    }
}

/// 游戏简要统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStats {
    /// 游戏总步数
    pub steps: usize,
    /// 获胜方: Some(1)=红胜, Some(-1)=黑胜, None/Some(0)=平局
    pub winner: Option<i32>,
}

/// 单局游戏的完整数据记录
///
/// 包含该局游戏中每一步的观测状态、MCTS 搜索产生的策略概率、
/// MCTS 估算的根节点价值、实际选择的动作以及最终的游戏结果。
///
/// 样本中的观测统一使用 `ResNetObservation`（各游戏按自身通道/尺寸编码），
/// 因此 `GameEpisode` 本身不携带游戏泛型参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameEpisode {
    /// 训练样本列表: (观测状态, 策略概率分布, MCTS根节点价值, completed_Q, 根节点访问次数, 最终回报, 动作掩码, 实际动作, 终局归一化血量差, 是否 Full Search)
    /// health_diff 为终局血量差按该样本玩家视角取号（红方视角为正）。
    /// is_full_search=true 表示该样本来自 Full Search（算力随机化下用于训练的选择性标记，
    /// 见 playout_cap_random_enabled / full_search_prob 字段）。
    pub samples: Vec<(ResNetObservation, Vec<f32>, f32, f32, u32, f32, Vec<i32>, usize, f32, bool)>,
    /// 游戏总步数
    pub game_length: usize,
    /// 获胜方
    pub winner: Option<i32>,
    /// 终局归一化血量差（红方视角为正）：(红HP-黑HP)/(初始总HP+最大子力分值)，大致落在 [-1,1]。
    pub health_diff_red: Option<f32>,
    /// NNUE 稀疏特征：None = 未收集；Some 时元组为 (布局元信息, 每步特征)，
    /// 每步特征长度与 samples 严格对齐。
    #[serde(default)]
    pub nnue: Option<(NnueEpisodeMeta, Vec<NnueStepFeatures>)>,
}

// ================ 场景定义 ================

/// 训练场景类型枚举（暗棋专用）
///
/// 预留扩展点：未来可实现特定残局/开局场景 (如 TwoAdvisors, HiddenThreats)。
/// 目前所有场景均退化为标准开局。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioType {
    /// 场景1: 双士残局 (R_A vs B_A) — 未实现，回退为 Standard
    TwoAdvisors,
    /// 场景2: 隐藏威胁 (Hidden Threat) — 未实现，回退为 Standard
    HiddenThreats,
    /// 标准开局 - 正常的完整游戏
    Standard,
}

impl ScenarioType {
    /// 根据枚举值创建对应的游戏环境（当前所有场景均创建标准环境）
    pub fn create_env(&self) -> banqi_core::core::env::DarkChessEnv {
        banqi_core::core::env::DarkChessEnv::new()
    }

    /// 获取场景的描述名称
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioType::TwoAdvisors => "TwoAdvisors (R_A vs B_A) [unimplemented=Standard]",
            ScenarioType::HiddenThreats => "HiddenThreats [unimplemented=Standard]",
            ScenarioType::Standard => "Standard",
        }
    }

    /// 获取该场景下的期望最优动作索引 (预留验证接口，未实现场景默认返回 0)
    pub fn expected_action(&self) -> usize {
        match self {
            ScenarioType::TwoAdvisors | ScenarioType::HiddenThreats | ScenarioType::Standard => 0,
        }
    }
}

// ================ 自对弈配置 ================

/// 自对弈配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SelfPlayConfig {
    /// 每次决策执行的 MCTS 模拟次数
    pub mcts_sims: usize,
    /// Gumbel Top-K 候选动作数
    pub max_considered_actions: usize,
    // 注意：根节点 Dirichlet 噪声注入已移除。Gumbel AlphaZero 的探索由
    // Gumbel 噪声（Top-K 采样）与 Sequential Halving 提供，根节点子节点
    // prior 不参与任何搜索决策（Top-K 用 logit、根选择不经 PUCT），
    // 注入 Dirichlet 无效，请勿重新添加 dirichlet_alpha / dirichlet_epsilon 字段。
    //
    // 注意：落子不再使用根节点温度（temperature / temperature_steps）。Gumbel
    // 噪声已在每次搜索时为动作选择注入探索（见 search.rs 的 sample_gumbel_top_k），
    // 再对 completed_Q 做 softmax 温度采样是多余的随机源，且会让「实际落子」偏离
    // 搜索选出的最优动作（search_result.action），造成行为与训练目标
    // improved_policy（logit + σ·Q）脱节。请勿重新添加 temperature_steps 字段。
    /// 训练场景
    pub scenario: ScenarioType,
    /// PUCT 探索系数（c_puct）与训练目标 σ 的缩放因子。默认 1.0。
    pub c_scale: f32,
    /// Gumbel 噪声尺度（根节点 Top-K 采样探索强度）。默认 1.0（标准 Gumbel）。
    pub gumbel_scale: f32,
    /// 是否启用算力分配随机化 (Playout Cap Randomization)
    pub playout_cap_random_enabled: bool,
    /// Fast Search 模拟次数；0 = 由 mcts_sims 推导（见 `fast_sims()`）
    pub fast_mcts_sims: usize,
    /// Full Search 出现概率 (如 0.25)
    pub full_search_prob: f32,
    /// 是否启用血量差异头参与 MCTS 搜索（复合效用 U = Q_win + λ·Q_hp）
    pub health_enabled: bool,
    /// 复合效用中血量期望权重 λ（0 = 纯胜率）
    pub health_weight: f32,
    /// λ 随 |v_win| 的自适应幂指数（0 = 常量 λ）
    pub health_confidence_exp: f32,
    /// 是否在记录 Episode 时同步收集 NNUE 稀疏特征（供 NNUE 训练管道）
    pub collect_nnue_features: bool,
    /// 整局复用同一棵 MCTS 树：每步用 `step_next` 把根推进到实际走子的子节点。
    ///
    /// 收益：已积累的访问 / Q 直接参与下一步搜索（不必从零重搜），并省去每步的根评估推理。
    /// 代价：
    ///   1. 根节点访问计数随局内累积，而改进策略 sigma = c_scale·ln(1+N_root) 随之增大，
    ///      训练目标会逐步向 Q 主导的锐化分布偏移（用 train/policy_entropy 观测）；
    ///   2. arena 整局不释放，内存随对局步数线性增长（4x8 长局须实测 RSS）。
    /// 默认关闭。启用前提是双方为同一模型的自对弈（collector 的 selfplay 任务恒满足；
    /// 异构对手走非记录的评估路径，不经过此处）。
    pub tree_reuse: bool,
    /// 自对弈线程数（0 = CPU 核数）；只被采集进程用于建 rayon 线程池
    pub threads: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_sims: 64,
            max_considered_actions: 16,
            scenario: ScenarioType::Standard,
            c_scale: 1.0,
            gumbel_scale: 1.0,
            playout_cap_random_enabled: false,
            fast_mcts_sims: 0,
            full_search_prob: 0.25,
            health_enabled: false,
            health_weight: 0.0,
            health_confidence_exp: 0.0,
            collect_nnue_features: false,
            tree_reuse: false,
            threads: 0,
        }
    }
}

impl SelfPlayConfig {
    /// Fast Search 的模拟次数：`fast_mcts_sims = 0` 时按 `mcts_sims / 4` 推导（至少 1）。
    pub fn fast_sims(&self) -> usize {
        if self.fast_mcts_sims == 0 {
            (self.mcts_sims / 4).max(1)
        } else {
            self.fast_mcts_sims
        }
    }
}

// ================ 自对弈运行器 (同步) ================

/// 自对弈运行器
///
/// 直接持有评估器引用，同步执行。
/// `G` 为游戏环境类型，环境由 `make_env` 工厂创建。
pub struct SelfPlayRunner<'a, G: GameEnv, E: Evaluator<G>> {
    evaluator: &'a E,
    config: SelfPlayConfig,
    make_env: fn() -> G,
}

impl<'a, G: GameEnv, E: Evaluator<G>> SelfPlayRunner<'a, G, E> {
    /// 创建新的自对弈运行器
    pub fn new(evaluator: &'a E, config: SelfPlayConfig, make_env: fn() -> G) -> Self {
        Self {
            evaluator,
            config,
            make_env,
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults(evaluator: &'a E, mcts_sims: usize, make_env: fn() -> G) -> Self {
        let config = SelfPlayConfig {
            mcts_sims,
            ..Default::default()
        };
        Self {
            evaluator,
            config,
            make_env,
        }
    }

    /// 执行一局完整的自对弈 (同步)
    pub fn play_episode(&self, _episode_num: usize) -> GameEpisode {
        let _start_time = Instant::now();

        // 1. 初始化环境
        let mut env = (self.make_env)();

        // 2. 配置 MCTS
        let mcts_config = GumbelConfig {
            num_simulations: self.config.mcts_sims,
            max_considered_actions: self.config.max_considered_actions,
            c_scale: self.config.c_scale,
            gumbel_scale: self.config.gumbel_scale,
            health_enabled: self.config.health_enabled,
            health_weight: self.config.health_weight,
            health_confidence_exp: self.config.health_confidence_exp,
        };
        let mut mcts = GumbelMCTS::new(&env, self.evaluator, mcts_config.clone());

        let mut episode_data = Vec::new();
        let mut step = 0;

        // 3. 游戏主循环
        loop {
            // 算力随机化：判断本步是 Full Search 还是 Fast Search
            let is_full_search = if self.config.playout_cap_random_enabled {
                if step == 0 {
                    true
                } else {
                    rand::random::<f32>() < self.config.full_search_prob
                }
            } else {
                true
            };
            let step_sims = if is_full_search {
                self.config.mcts_sims
            } else {
                self.config.fast_sims()
            };

            let step_mcts_config = GumbelConfig {
                num_simulations: step_sims,
                max_considered_actions: self.config.max_considered_actions,
                c_scale: self.config.c_scale,
                gumbel_scale: self.config.gumbel_scale,
                health_enabled: self.config.health_enabled,
                health_weight: self.config.health_weight,
                health_confidence_exp: self.config.health_confidence_exp,
            };
            let mut step_mcts = GumbelMCTS::new(&env, self.evaluator, step_mcts_config);

            // --- MCTS 搜索 (同步) ---
            let search_result = match step_mcts.run() {
                Ok(Some(result)) => result,
                Ok(None) => {
                    let (_, _, winner) = env.check_game_over_conditions();
                    return crate::pipeline::self_play::finalize_episode(
                        episode_data,
                        winner,
                        env.terminal_health_diff_red(),
                        None,
                    );
                }
                Err(e) => {
                    // 推理失败：本局作废（winner 置 None），不写入训练数据。
                    eprintln!("❌ MCTS 评估失败，本局作废: {e}");
                    return crate::pipeline::self_play::finalize_episode(
                        episode_data,
                        None,
                        env.terminal_health_diff_red(),
                        None,
                    );
                }
            };

            // --- 落子：直接采用 Gumbel 搜索选出的动作。探索由每次搜索重新抽的
            // Gumbel 噪声提供（sample_gumbel_top_k），无需根温度采样，详见
            // SelfPlayConfig 字段注释。---
            let action = search_result.action;
            let completed_q = search_result.completed_q;

            // --- 全记录：无论 Full/Fast 都收集样本，并标记 is_full_search ---
            // 算力随机化下 Fast Search 步的样本同样入库，交给 Python 侧「选择性使用」
            // （losses.py 仅让 Full Search 样本参与训练，Fast 样本保留供未来逻辑使用）。
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

            // --- 执行动作 ---
            match env.step(action) {
                Ok((_, terminated, truncated, winner)) => {
                    // 推进 MCTS 树
                    mcts.step_next(&env, action);

                    if terminated || truncated {
                        // --- 游戏结束处理：统一回填 ---
                        return crate::pipeline::self_play::finalize_episode(
                            episode_data,
                            winner,
                            env.terminal_health_diff_red(),
                            None,
                        );
                    }
                }
                Err(e) => {
                    eprintln!("  ⚠️ 游戏错误 (step={}, action={}): {}", step, action, e);
                    return GameEpisode {
                        samples: Vec::new(),
                        game_length: step,
                        winner: None,
                        health_diff_red: env.terminal_health_diff_red(),
                        nnue: None,
                    };
                }
            }

            // --- 步数限制检查：使用环境给定的步数上限 ---
            step += 1;
            if step >= env.max_steps() {
                // 步数上限截断：环境视其为 truncated 平局 (winner=Some(0))，
                // 与终局分支语义对齐，game_result 回填 0.0。
                return crate::pipeline::self_play::finalize_episode(
                    episode_data,
                    Some(0),
                    env.terminal_health_diff_red(),
                    None,
                );
            }
        }
    }
}

// ================ 高级 API ================

/// 运行单局自对弈
///
/// `make_env` 为环境工厂（如 `DarkChessEnv::new`）。
pub fn run_self_play<G: GameEnv, E: Evaluator<G>>(
    evaluator: &E,
    config: &SelfPlayConfig,
    make_env: fn() -> G,
) -> GameEpisode {
    let runner = SelfPlayRunner::new(evaluator, config.clone(), make_env);
    runner.play_episode(0)
}

/// 批量运行多局自对弈
pub fn run_batch_self_play<G: GameEnv, E: Evaluator<G>>(
    evaluator: &E,
    config: &SelfPlayConfig,
    num_games: usize,
    make_env: fn() -> G,
) -> Vec<GameEpisode> {
    (0..num_games)
        .map(|i| {
            let runner = SelfPlayRunner::new(evaluator, config.clone(), make_env);
            runner.play_episode(i)
        })
        .collect()
}
