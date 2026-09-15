# banqi-collector

分布式训练数据采集 crate（2026-09-11 自 rust_4x8 拆出）。

## 定位

- 仅服务分布式训练链路：作为 gRPC 客户端对接 Go 版 `banqi-scheduler`（GetTask / ReportEpisode / ReportMatchResult / Heartbeat 等 9 RPC），episode 数据经 R2 预签名 URL 直传。
- 依赖 `banqi-core`（领域核心）与 `banqi-engine`（ONNX 推理 `OnnxModel` / `OnnxEvaluator`，feature `onnx` / `onnx-cuda`）。
- 不含本地采集（`--backend local` 的 LocalRegistry / LocalEpisodeStore），该形态留在 rust_4x8 主仓库。

## 模块

- `src/config.rs`：分层配置 `CollectorConfig`（`[scheduler]` / `[selfplay]` 两段，并发旋钮归入各自消费方）：加载顺序为「默认值 → `--config <TOML>` → CLI 覆盖」，`--config-dump <PATH>` 回写生效快照；字段拼错 / 取值越界在启动时即报错，不会带进运行期。
- `src/registry/scheduler_registry.rs`：`SchedulerRegistry` — gRPC 客户端（`Channel` 长连接，一次建连全程复用）+ reqwest 预签名下载/直传 + sha256 SRI 校验 + 30s 后台心跳；模型按 sha 进程内缓存（每个 sha 一个**会话池**，通道数由 `scheduler.sessions` 决定，A/B、rating 双方共用）。上报异步化：`submit_episode_report` / `submit_match_report` 立即返回，JSON 序列化 + gzip + sha256 + R2 直传都在后台 tokio 任务中完成，`report_slots` 信号量限制在途批数（背压 + 约束驻留内存）；心跳发现 best 变化时后台预取该网络。解析 `SelfPlayParams.extra_config`（JSON 透传）中的 `initial_revealed_pieces`，注入 `SchedulerTask.initial_revealed`。
- `src/pipeline/self_play/match_core.rs`：`MatchParams.make_env` 为 `Arc<dyn Fn() -> G + Send + Sync>` 环境工厂（支持课程参数闭包注入）；`run_match_core` 其余语义不变。
- `src/pipeline/self_play/`：自对弈主干 `run_match_core` / `PlayerSpec` / `MatchResult`（`types` / `match_core` / `finalize` / `serialize`），已移除 PyO3（PyPredictor）分支；`serialize::episode_to_dict_json` 行格式与主仓库及 Python 侧契约一致。
- `proto/scheduler.proto`：与 banqi-scheduler 仓库副本同步维护（双侧同步，变更须同时更新）。

## 变更记录

- 2026-09-15：会话池取代 A/B 双会话（同日早些时候的 `opponent_model` 方案作废，该方法已删除；4x2 实测 35.6 → 160 局/s，老侧 `run_native_match` 为 37.0 局/s）：
  - **`banqi-engine::OnnxModel` 会话池**（详见 banqi-engine 变更记录）：`Vec<Mutex<Session>>` + `AtomicUsize` 轮转分配，新增 `with_sessions(path, device, n)` / `session_count()`，`new()` 仍为单会话；每会话 ORT intra-op 线程取 `核数 / 会话数`。
  - **配置**：新增 `scheduler.sessions`（并发推理通道数，`0` = 自动 = 自对弈线程数，bin 启动时解析并写回生效配置；`--sessions` 可覆盖）。每会话持有独立的 ORT 会话与线程池，故内存随会话数线性增长。
  - **registry**：`model(sha)` 改为加载会话池并按 sha 缓存；`opponent_model` 删除。自对弈 A/B 与 rating 双方共用同一会话池 —— 池内通道等价，谁先到谁用，比「每方固定一条通道」更省通道（此前 A/B 各一条固定通道，任一侧空闲时另一侧也用不上）。
  - **配套**：同一批还修复了 `banqi-core` 热路径全局锁缓存（见 banqi-core 变更记录），环境/搜索层 12 线程由 155 → 677 局/s，消除会话池之外的另一处瓶颈。
- 2026-09-15：修复自对弈推理的单会话串行化（4x2 实测吞吐 17.9 → 35.6 局/s，老侧 `run_native_match` 为 37.0 局/s）：`OnnxModel` 内部以 `Mutex<Session>` 串行化推理，此前 bin 把同一个 `Arc<OnnxModel>` 同时交给 `player_a`/`player_b`，双色对局挤在同一条推理通道上排队；当时由 `SchedulerRegistry::opponent_model(sha)` 为同一 sha 再开一个独立会话（按 sha 缓存），与老侧「每方各建会话」对齐。**该方案已被上一条的会话池取代。**
- 2026-09-15：并发与配置重构（配置分层思路参考 xmrig 的 `Config`/`ConfigTransform`，后台 I/O 思路参考其 worker 与网络线程分离）：
  - **配置分层**：新增 `CollectorConfig`（`src/config.rs`），加载顺序「默认值 → `--config <TOML>` → CLI 覆盖 → 校验」，`--config-dump` 落盘生效快照；CLI 参数改为 `Option`，只覆盖显式给出的项。删除无用参数 `games_per_iter`（每批局数由服务端 `TaskResponse.games` 下发，该参数此前从未被读取）。`SchedulerConfig.client_version` 字段移除，版本声明改由 `registry::CLIENT_VERSION` 常量注入。
  - **并发旋钮**：自对弈线程数 `selfplay.threads`（`0` = CPU 核数，bin 内 `build_pool` 建 rayon 池），在途上报批数 `scheduler.max_inflight_reports`（`SchedulerRegistry` 的信号量额度）。
  - **计算与 I/O 重叠**：`Channel` 长连接复用（原先每个 RPC 都重新 `connect`，含心跳）；`submit_episode_report` / `submit_match_report` 异步化 —— 序列化 + gzip + sha256 + R2 PUT 全部移出主循环，主循环跑完一批立刻拉下一批；信号量背压；心跳感知 best 变化后后台预取网络（`prefetch_network`），换网不再阻塞采集；退出前 `flush_reports` 等在途上报落地并汇总失败批数。
  - **序列化上报副作用**：主循环日志中的「耗时」自此只含计算段，不再包含上传。
  - **`SelfPlayConfig`** 接入 serde 并收敛默认值：`playout_cap_random_enabled` 默认改为 `false`（与生产一致，此前 `Default` 为 `true` 会让 `onnx_sims_curve` 的胜率曲线被算力随机化污染）；`fast_mcts_sims = 0` 表示按 `mcts_sims / 4` 推导，新增 `SelfPlayConfig::fast_sims()` 统一读取点；`ScenarioType` 增加 serde（snake_case 字面量 `standard` 等）。
- 2026-09-15：同步 `banqi-core` 的 MCTS 评估契约变更（`Evaluator::evaluate` / `GumbelMCTS::run` 改为 `Result`）：`RandomEval` / `PlayerEval` 实现改为返回 `Result<EvaluatorOutput, EvaluatorError>`；`model_mcts_action` / `policy_argmax_action` / `play_one_game_recorded` / `SelfPlayRunner::play_episode` 在推理失败时**打印错误并令本局作废**（`episode: None` 或 `winner: None`），不再 panic、也不写入无效训练数据；bin 侧原有「record 模式下 0 局产出即 `bail!`」的判定会把失败暴露为批次错误。

## 入口

`src/bin/collector.rs` → bin `banqi-collector`（required-features = `onnx`）：
`cargo build --release --features onnx`（CUDA 推理用 `--features onnx-cuda`）。

配置：两段 TOML（`[scheduler]` / `[selfplay]`），CLI 同名参数逐项覆盖。
启动会打印生效配置；`--config-dump <PATH>` 可落盘快照，该文件可直接用 `--config` 回灌。

```bash
# 用默认值 + CLI 覆盖
banqi-collector --scheduler-endpoint http://host:50051 --threads 12 --mcts-sims 64

# 用配置文件，并落盘本次生效快照
banqi-collector --config collector.toml --config-dump /tmp/collector.effective.toml
```

```toml
[scheduler]
endpoint = "http://127.0.0.1:50051"
worker_id = ""            # 空 = worker-<pid>
cache_dir = "outputs/distributed_cache"
device = "auto"
sessions = 0              # ONNX 会话数（并发推理通道数），0 = 自动 = 自对弈线程数
max_inflight_reports = 2  # 在途上报批数上限（背压）

[selfplay]
threads = 12              # 自对弈线程数，0 = CPU 核数
mcts_sims = 64
max_considered_actions = 16
scenario = "standard"
playout_cap_random_enabled = false
fast_mcts_sims = 0        # 0 = mcts_sims / 4
full_search_prob = 0.25
```

课程学习：调度器经 `extra_config` 下发 `initial_revealed_pieces` 时，bin 以 `banqi_core::core::env::CurriculumEnv::with_initial_revealed(n)` 构造每局环境（覆盖变体默认值，棋盘/动作空间/特征维度不变，网络跨阶段通用）；未下发时用变体默认配置。
