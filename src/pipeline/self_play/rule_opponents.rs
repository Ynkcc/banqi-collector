// src/pipeline/self_play/rule_opponents.rs
// 规则策略对手：标识解析 + `RulePolicy` 适配（仅用于评估路径，不产训练数据）。
//
// 具体策略实现复用 banqi-engine（优先吃子 / 优先翻棋），因此本模块随 `onnx`
// feature 编译——banqi-engine 在 banqi-collector 中是 optional 依赖。
// `match_core` 只认识 `RulePolicy` trait、不依赖 banqi-engine，依赖拓扑不变。
//
// 对手标识（与调度器 TaskResponse.opponent_spec 同源）：
//   "rule:capture_first" / "capture_first"  → 优先吃子
//   "rule:reveal_first"  / "reveal_first"   → 优先翻棋
//   "random"                                → 复用内置 PlayerSpec::Random

use std::sync::Arc;

use banqi_core::core::env::DarkChessEnv;
use banqi_core::core::env::GameEnv;
use banqi_engine::engine::{CaptureFirstPolicy, Policy, RevealFirstPolicy};

use super::match_core::{PlayerSpec, RulePolicy};

/// 规则对手种类（不含随机——随机走内置 `PlayerSpec::Random`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleOpponent {
    /// 优先吃子：有吃明子动作则随机吃一个，否则优先翻棋，再次随机静走。
    CaptureFirst,
    /// 优先翻棋：有翻棋动作则随机翻一个，否则在其余合法动作中随机。
    RevealFirst,
}

impl RuleOpponent {
    /// 解析对手标识；`rule:` 前缀可选。无法识别时返回 `None`（由调用方决定报错方式）。
    pub fn parse(spec: &str) -> Option<Self> {
        let s = spec.trim();
        let s = s.strip_prefix("rule:").unwrap_or(s);
        match s {
            "capture_first" => Some(Self::CaptureFirst),
            "reveal_first" => Some(Self::RevealFirst),
            _ => None,
        }
    }

    /// 规范化标识（回归到调度器上报用的稳定写法）。
    pub fn spec(self) -> &'static str {
        match self {
            Self::CaptureFirst => "rule:capture_first",
            Self::RevealFirst => "rule:reveal_first",
        }
    }

    /// 中文名（日志/展示用）。
    pub fn label(self) -> &'static str {
        match self {
            Self::CaptureFirst => "优先吃子",
            Self::RevealFirst => "优先翻棋",
        }
    }
}

/// `RulePolicy` 的 banqi-engine 适配器。
struct EngineRulePolicy {
    kind: RuleOpponent,
}

impl RulePolicy for EngineRulePolicy {
    fn choose_action(&self, env: &DarkChessEnv) -> Option<usize> {
        match self.kind {
            RuleOpponent::CaptureFirst => CaptureFirstPolicy::choose_action(env),
            RuleOpponent::RevealFirst => RevealFirstPolicy::choose_action(env),
        }
    }
}

/// 构造规则对手的选手规格。
///
/// `Random` 不在 `RuleOpponent` 里：随机走子由 `PlayerSpec::Random` 承担
/// （语义即「合法动作上均匀随机」），语义一致且不额外耦合 banqi-engine。
pub fn eval_opponent_spec<G: GameEnv>(kind: RuleOpponent) -> PlayerSpec<G> {
    PlayerSpec::Rule(Arc::new(EngineRulePolicy { kind }))
}

/// 内置阶梯的默认对手标识（由弱到强，供调度器 `SCHEDULER_EVAL_OPPONENTS` 默认值参考）。
pub const DEFAULT_EVAL_OPPONENT_SPECS: [&str; 3] = [
    "random",
    "rule:reveal_first",
    "rule:capture_first",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::self_play::match_core::{
        AsDarkChessRef, MatchParams, SeedableEnv, run_match_core,
    };
    use crate::pipeline::self_play::types::SelfPlayConfig;
    use banqi_core::core::env::variants::MiniDarkChessEnv;

    #[test]
    fn parse_accepts_prefixed_and_bare_specs() {
        assert_eq!(
            RuleOpponent::parse("rule:capture_first"),
            Some(RuleOpponent::CaptureFirst)
        );
        assert_eq!(
            RuleOpponent::parse(" capture_first "),
            Some(RuleOpponent::CaptureFirst)
        );
        assert_eq!(
            RuleOpponent::parse("reveal_first"),
            Some(RuleOpponent::RevealFirst)
        );
        // random 由内置选手承担，不属于规则对手
        assert_eq!(RuleOpponent::parse("random"), None);
        assert_eq!(RuleOpponent::parse("best"), None);
        // spec() 是 parse 的右逆
        for k in [RuleOpponent::CaptureFirst, RuleOpponent::RevealFirst] {
            assert_eq!(RuleOpponent::parse(k.spec()), Some(k));
        }
    }

    #[test]
    fn rule_policy_returns_legal_action() {
        for kind in [RuleOpponent::CaptureFirst, RuleOpponent::RevealFirst] {
            let mut env = MiniDarkChessEnv::default();
            SeedableEnv::set_seed(&mut env, 7);
            let spec = eval_opponent_spec::<MiniDarkChessEnv>(kind);
            let PlayerSpec::Rule(policy) = &spec else {
                panic!("规则对手应为 PlayerSpec::Rule");
            };
            let inner = env.as_darkchess_ref();
            let legal = inner.legal_action_indices();
            assert!(!legal.is_empty(), "初始局面应有合法动作");
            let action = policy.choose_action(inner).expect("有合法动作时应返回动作");
            assert!(legal.contains(&action), "{kind:?} 返回了非法动作 {action}");
        }
    }

    /// 规则选手在评估主干（record_episodes=false）中被正确驱动并产出终局结果。
    #[test]
    fn rule_opponent_runs_in_eval_path() {
        let spec_a = eval_opponent_spec::<MiniDarkChessEnv>(RuleOpponent::CaptureFirst);
        let spec_b = PlayerSpec::<MiniDarkChessEnv>::Random;
        let config = SelfPlayConfig::default();
        let make_env: Arc<dyn Fn() -> MiniDarkChessEnv + Send + Sync> =
            Arc::new(MiniDarkChessEnv::default);
        let r = run_match_core(MatchParams {
            player_a: &spec_a,
            player_b: &spec_b,
            n_games: 4,
            config: &config,
            seed: Some(1),
            record_episodes: false,
            batched: false,
            model_sims: 1,
            opponent_sims: None,
            thread_pool: None,
            make_env,
        });
        assert_eq!(r.wins + r.draws + r.losses, 4, "4 局都应产出终局结果");
        assert!(r.avg_moves > 0.0, "平均步数应大于 0");
        assert!(r.episodes.is_empty(), "评估路径不得产出 episode");
    }
}
