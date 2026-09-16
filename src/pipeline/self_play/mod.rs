pub mod types;
pub mod batched;
pub mod codec;
pub mod finalize;
pub mod match_core;
pub mod reanalysis;

pub use types::{
    GameEpisode, GameStats, NnueEpisode, NnueEpisodeMeta, NnueStepFeatures, ScenarioType,
    SelfPlayConfig, SelfPlayRunner, run_batch_self_play, run_self_play,
};
pub use finalize::{finalize_episode, get_top_k_actions, select_completed_q_action};
pub use match_core::{
    AsDarkChessRef, MatchParams, MatchResult, PlayerSpec, SeedableEnv, run_match_core,
};
pub use codec::{SCHEMA_VERSION, encode_episode_batch};
pub use reanalysis::{
    PAYLOAD_VERSION, ReanalysisItem, ReanalysisReport, decode_payload, run_reanalysis,
};
