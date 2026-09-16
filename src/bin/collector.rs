// src/bin/collector.rs — 分布式采集进程（banqi-collector，仅 scheduler backend）
//
// 自 rust_4x8/src/bin/collector.rs 迁入（banqi-collector 拆分），仅保留分布式形态：
// SchedulerRegistry（gRPC GetTask + R2 预签名直传；selfplay/rating 双任务）。
// 本地采集（LocalRegistry/LocalEpisodeStore）留在 rust_4x8 主仓库。
//
// 配置：分层加载（默认值 → TOML → CLI 覆盖）见 src/config.rs；--config-dump 可落盘快照。
// 并发：主循环串行拉任务，批内 rayon 局级并行；上报异步化，计算与上传重叠（见 registry 头注释）。

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::ThreadPoolBuilder;

use banqi_collector::config::{CliOverrides, CollectorConfig};
use banqi_collector::pipeline::self_play::{
    AsDarkChessRef, GameEpisode, MatchParams, MatchResult, PlayerSpec, SeedableEnv,
    SelfPlayConfig, run_match_core,
};
use banqi_collector::registry::{
    CLIENT_VERSION, EpisodeBatch, MatchReport, SchedulerRegistry,
};
use banqi_collector::pb::TaskKind;
use banqi_core::core::env::traits::GameEnv;
use banqi_core::core::env::{
    CurriculumEnv, DarkChessEnv,
    variants::{Game4x4Env, MiniDarkChessEnv},
};
use banqi_engine::inference::onnx::{OnnxModel, OnnxEvaluator};

/// 无任务时的轮询退避
const TASK_BACKOFF: Duration = Duration::from_secs(30);

#[derive(Parser, Debug)]
#[command(name = "banqi-collector", about = "分布式自对弈采集进程（scheduler backend）")]
struct Args {
    #[command(flatten)]
    overrides: CliOverrides,
    /// 总批数上限（0 = 无限循环）
    #[arg(long, default_value_t = 0)]
    iterations: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = CollectorConfig::load(&args.overrides)?;
    let pool = build_pool(cfg.selfplay.threads)?;
    // 会话数 0 = 自动：与并发对局数一致，避免推理被少数会话串行化
    if cfg.scheduler.sessions == 0 {
        cfg.scheduler.sessions = pool.current_num_threads();
    }

    println!("=== banqi-collector 启动（scheduler）v{CLIENT_VERSION} ===");
    println!("生效配置：\n{}", cfg.to_toml()?);
    println!(
        "自对弈线程池 = {} 线程，ONNX 会话数 = {} （variant 由服务端 GetTask 下发）",
        pool.current_num_threads(),
        cfg.scheduler.sessions
    );
    if !cfg.selfplay.batched_variants.is_empty() {
        println!(
            "批量锁步自对弈变体 = {:?}（并发 = 线程池大小 {}）；批量靠 batch 吃算力，\
             建议把 sessions 调小（1~{}）以给每个会话更多 intra-op 线程",
            cfg.selfplay.batched_variants,
            pool.current_num_threads(),
            batch_worker_hint(pool.current_num_threads())
        );
    }
    if let Some(path) = &args.overrides.config_dump {
        cfg.dump_to(path)?;
        println!("配置快照已写入: {}", path.display());
    }

    let mut registry = SchedulerRegistry::new(cfg.scheduler.clone())?;
    run_scheduler(&args, &cfg.selfplay, &mut registry, &pool)?;

    let failed = registry.failed_reports();
    if failed > 0 {
        eprintln!("⚠️ 本次运行有 {failed} 批上报失败，请检查网络与调度器日志");
    }
    Ok(())
}

fn run_scheduler(
    args: &Args,
    base_selfplay: &SelfPlayConfig,
    registry: &mut SchedulerRegistry,
    pool: &rayon::ThreadPool,
) -> Result<()> {
    // 服务端下发的 mcts_sims 会覆盖本地值，故运行时持有可变副本
    let mut selfplay = base_selfplay.clone();
    let mut iteration: usize = 0;

    while args.iterations == 0 || iteration < args.iterations {
        let task = match registry.get_task()? {
            Some(t) => {
                registry.set_running_task(&t.task_id);
                t
            }
            None => {
                std::thread::sleep(TASK_BACKOFF);
                continue;
            }
        };

        // 服务端下发覆盖本地值（0 = 不覆盖）；fast_mcts_sims 归零即按新值重新推导
        if task.mcts_sims > 0 {
            selfplay.mcts_sims = task.mcts_sims;
            selfplay.fast_mcts_sims = 0;
        }

        let started = Instant::now();
        match task.kind {
            TaskKind::TaskSelfplay => {
                // 同一模型（会话池）供 A/B 共用：会话池内各通道等价，A/B 谁都能用满，
                // 比「每方固定一条会话」更省通道（见 scheduler_registry::model）。
                let model = registry.model(&task.network_sha)?;
                let result = run_variant_dispatch(
                    &task.variant,
                    task.initial_revealed,
                    Arc::clone(&model),
                    model,
                    &selfplay,
                    task.games,
                    true,
                    pool,
                )?;

                let games = result.episodes.len();
                let avg = avg_steps(&result.episodes);
                // 批内聚合胜方（调度端仅作日志/统计用）
                let winner = if result.wins > result.losses {
                    1
                } else if result.wins < result.losses {
                    -1
                } else {
                    0
                };

                registry.add_completed_games(games);
                println!(
                    "[iter {iteration}] selfplay task={} 🎮 {games} 局（步均 {avg:.1}）计算耗时 {:.1}s",
                    task.task_id,
                    started.elapsed().as_secs_f64()
                );
                // 异步上报：编码/gzip/sha256/R2 直传都在后台，主循环立刻进入下一批
                registry.submit_episode_report(EpisodeBatch {
                    task_id: task.task_id.clone(),
                    network_sha: task.network_sha.clone(),
                    variant: task.variant.clone(),
                    data_kind: task.data_kind,
                    winner,
                    episodes: result.episodes,
                    nnue_episodes: result.nnue_episodes,
                });
            }
            TaskKind::TaskRating => {
                let candidate = registry.model(&task.network_sha)?;
                let opponent = registry.model(&task.opponent_sha)?;
                // 换色配对要求偶数局
                let n = (task.games - task.games % 2).max(2);
                let result = run_variant_dispatch(
                    &task.variant,
                    task.initial_revealed,
                    candidate,
                    opponent,
                    &selfplay,
                    n,
                    false,
                    pool,
                )?;
                let pairs = pairs_from_outcomes(&result.game_outcomes);
                println!(
                    "[iter {iteration}] rating task={} games={n} w/l/d={}/{}/{} pairs={pairs:?} 计算耗时 {:.1}s",
                    task.task_id,
                    result.wins,
                    result.losses,
                    result.draws,
                    started.elapsed().as_secs_f64()
                );
                registry.add_completed_games(n);
                registry.submit_match_report(MatchReport {
                    task_id: task.task_id.clone(),
                    network_sha: task.network_sha.clone(),
                    opponent_sha: task.opponent_sha.clone(),
                    games: n,
                    wins: result.wins,
                    losses: result.losses,
                    draws: result.draws,
                    pairs,
                });
            }
            TaskKind::TaskNone => unreachable!("get_task 已过滤 TASK_NONE"),
        }
        iteration += 1;
    }

    // 退出前等待在途上报落地（背压信号量全部可用即表示无在途任务）
    registry.flush_reports();
    Ok(())
}

/// 批量路径建议的会话数上限：与 `batched::run_batched_games` 的评估 worker 数同规则
/// （`concurrency.min(8)`），使「在途批数 ≈ 会话数」，避免多余会话各自摊薄 intra-op 线程。
fn batch_worker_hint(concurrency: usize) -> usize {
    concurrency.min(8).max(1)
}

/// 自对弈线程池：`threads = 0` 取 CPU 核数。
fn build_pool(threads: usize) -> Result<rayon::ThreadPool> {
    let threads = if threads > 0 { threads } else { num_cpus::get() };
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .with_context(|| format!("构建 rayon 线程池失败（threads={threads}）"))
}

fn avg_steps(episodes: &[GameEpisode]) -> f64 {
    if episodes.is_empty() {
        0.0
    } else {
        episodes.iter().map(|e| e.game_length as f64).sum::<f64>() / episodes.len() as f64
    }
}

/// 按变体分发到泛型主干（run_match_core 需静态类型 G）。
#[allow(clippy::too_many_arguments)]
fn run_variant_dispatch(
    variant: &str,
    initial_revealed: Option<usize>,
    model_a: Arc<OnnxModel>,
    model_b: Arc<OnnxModel>,
    config: &SelfPlayConfig,
    n_games: usize,
    record_episodes: bool,
    pool: &rayon::ThreadPool,
) -> Result<MatchResult> {
    // 批量锁步路径仅用于记录模式（自对弈，双方同一模型）：rating 是异构对手且不产数据，
    // 必须走单树路径。
    let batched = record_episodes && config.batched_for(variant);
    match variant {
        "4x8" => run_games::<DarkChessEnv>(initial_revealed, model_a, model_b, config, n_games, record_episodes, batched, pool),
        "4x4" => run_games::<Game4x4Env>(initial_revealed, model_a, model_b, config, n_games, record_episodes, batched, pool),
        "4x2" => run_games::<MiniDarkChessEnv>(initial_revealed, model_a, model_b, config, n_games, record_episodes, batched, pool),
        other => anyhow::bail!("未知变体: {other}（可选 4x8 / 4x4 / 4x2）"),
    }
}

fn run_games<G>(
    initial_revealed: Option<usize>,
    model_a: Arc<OnnxModel>,
    model_b: Arc<OnnxModel>,
    config: &SelfPlayConfig,
    n_games: usize,
    record_episodes: bool,
    batched: bool,
    pool: &rayon::ThreadPool,
) -> Result<MatchResult>
where
    G: GameEnv
        + AsDarkChessRef
        + SeedableEnv
        + CurriculumEnv
        + Send
        + Sync
        + Default
        + 'static,
{
    let make_env: Arc<dyn Fn() -> G + Send + Sync> = match initial_revealed {
        Some(n) => {
            println!("[curriculum] 初始翻子数 override = {n}（变体默认值已覆盖）");
            Arc::new(move || G::with_initial_revealed(n))
        }
        None => Arc::new(G::default),
    };
    let spec_a = PlayerSpec::ModelEval(Arc::new(OnnxEvaluator::<G>::new(model_a)));
    let spec_b = PlayerSpec::ModelEval(Arc::new(OnnxEvaluator::<G>::new(model_b)));
    let result = run_match_core(MatchParams {
        player_a: &spec_a,
        player_b: &spec_b,
        n_games,
        config,
        seed: None,
        record_episodes,
        batched,
        model_sims: config.mcts_sims,
        thread_pool: Some(pool),
        make_env,
    });
    if result.episodes.is_empty() && result.nnue_episodes.is_empty() && record_episodes {
        anyhow::bail!("自对弈 0 局产出（检查模型与配置）");
    }
    Ok(result)
}

/// 从逐局结果推导五项成对计数（换色配对：i 与 i+1 一组，均为 candidate 视角）。
/// 得分映射 负=0 / 和=1 / 胜=2，两局得分之和 0..4 对应 LL/LD/DD/DW/WW。
fn pairs_from_outcomes(outcomes: &[i32]) -> [usize; 5] {
    let score = |r: i32| match r {
        1 => 2usize,
        0 => 1,
        _ => 0,
    };
    let mut pairs = [0usize; 5];
    for chunk in outcomes.chunks(2) {
        if chunk.len() == 2 {
            pairs[score(chunk[0]) + score(chunk[1])] += 1;
        }
    }
    pairs
}
