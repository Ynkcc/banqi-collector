# banqi-collector

分布式训练数据采集 crate（2026-09-11 自 rust_4x8 拆出）。

## 定位

- 仅服务分布式训练链路：作为 gRPC 客户端对接 Go 版 `banqi-scheduler`（GetTask / ReportEpisode / ReportMatchResult / Heartbeat 等 9 RPC），episode 数据经 R2 预签名 URL 直传。
- 依赖 `banqi-core`（领域核心）与 `banqi-engine`（ONNX 推理 `OnnxModel` / `OnnxEvaluator`，feature `onnx` / `onnx-cuda`）。
- 不含本地采集（`--backend local` 的 LocalRegistry / LocalEpisodeStore），该形态留在 rust_4x8 主仓库。

## 模块

- `src/registry/scheduler_registry.rs`：`SchedulerRegistry` — gRPC 客户端 + reqwest 预签名下载/直传 + sha256 SRI 校验 + 30s 后台心跳；模型按 sha 进程内缓存。解析 `SelfPlayParams.extra_config`（JSON 透传）中的 `initial_revealed_pieces`，注入 `SchedulerTask.initial_revealed`。
- `src/pipeline/self_play/match_core.rs`：`MatchParams.make_env` 为 `Arc<dyn Fn() -> G + Send + Sync>` 环境工厂（支持课程参数闭包注入）；`run_match_core` 其余语义不变。
- `src/pipeline/self_play/`：自对弈主干 `run_match_core` / `PlayerSpec` / `MatchResult`（`types` / `match_core` / `finalize` / `serialize`），已移除 PyO3（PyPredictor）分支；`serialize::episode_to_dict_json` 行格式与主仓库及 Python 侧契约一致。
- `proto/scheduler.proto`：与 banqi-scheduler 仓库副本同步维护（双侧同步，变更须同时更新）。

## 入口

`src/bin/collector.rs` → bin `banqi-collector`（required-features = `onnx`）：
`cargo build --release --features onnx`（CUDA 推理用 `--features onnx-cuda`）。

课程学习：调度器经 `extra_config` 下发 `initial_revealed_pieces` 时，bin 以 `banqi_core::core::env::CurriculumEnv::with_initial_revealed(n)` 构造每局环境（覆盖变体默认值，棋盘/动作空间/特征维度不变，网络跨阶段通用）；未下发时用变体默认配置。
