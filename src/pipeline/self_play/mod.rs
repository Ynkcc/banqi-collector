pub mod types;
pub mod batched;
pub mod codec;
pub mod finalize;
pub mod match_core;
pub mod reanalysis;
// 规则对手复用 banqi-engine 的策略实现，随 onnx feature 编译（该依赖为 optional）。
#[cfg(feature = "onnx")]
pub mod rule_opponents;

pub use types::{
    GameEpisode, GameStats, NnueEpisode, NnueEpisodeMeta, NnueStepFeatures, ScenarioType,
    SelfPlayConfig, SelfPlayRunner, run_batch_self_play, run_self_play,
};
pub use finalize::{finalize_episode, get_top_k_actions, select_completed_q_action};
pub use match_core::{
    AsDarkChessRef, MatchParams, MatchResult, PlayerSpec, RulePolicy, SeedableEnv, run_match_core,
};
#[cfg(feature = "onnx")]
pub use rule_opponents::{DEFAULT_EVAL_OPPONENT_SPECS, RuleOpponent, eval_opponent_spec};
pub use codec::{SCHEMA_VERSION, encode_episode_batch};
pub use reanalysis::{
    PAYLOAD_VERSION, ReanalysisItem, ReanalysisReport, decode_payload, run_reanalysis,
};
