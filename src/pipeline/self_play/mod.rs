pub mod types;
pub mod batched;
pub mod finalize;
pub mod match_core;
pub mod serialize;

pub use types::{
    GameEpisode, GameStats, NnueEpisode, NnueEpisodeMeta, NnueStepFeatures, ScenarioType,
    SelfPlayConfig, SelfPlayRunner, run_batch_self_play, run_self_play,
};
pub use finalize::{finalize_episode, get_top_k_actions, select_completed_q_action};
pub use match_core::{
    AsDarkChessRef, MatchParams, MatchResult, PlayerSpec, SeedableEnv, run_match_core,
};
pub use serialize::{episode_to_dict_json, nnue_episode_to_dict_json, nnue_episode_to_jsonl};
