// banqi-collector/examples/selfplay_throughput.rs
// 自对弈数据生成速率基准：复刻 bin/collector.rs 的生产配置（SelfPlayConfig 字面量、
// PlayerSpec::ModelEval 双方同网、record_episodes=true、局级 rayon 并行），
// 只计「自对弈产出 episode」与「jsonl.gz 序列化」两段耗时，不含 gRPC/R2。
//
// 用法（须带 onnx feature）：
//   cargo run --release --features onnx --example selfplay_throughput -- \
//     --model ../banqi-training/outputs/4x2/checkpoints/random_health.onnx \
//     --variant 4x2 --sims 64 --games 240 --threads 12 --batches 3 --warmup 1
//
// 输出每批一行 METRIC，便于两侧脚本统一聚合对比。

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;

use banqi_collector::pipeline::self_play::{
    AsDarkChessRef, GameEpisode, MatchParams, PlayerSpec, ScenarioType, SeedableEnv,
    SelfPlayConfig, serialize::episode_to_dict_json, run_match_core,
};
use banqi_core::core::env::traits::GameEnv;
use banqi_core::core::env::variants::{Game4x4Env, MiniDarkChessEnv};
use banqi_core::core::env::DarkChessEnv;
use banqi_engine::inference::onnx::{OnnxEvaluator, OnnxModel};

#[derive(Parser, Debug)]
#[command(
    name = "selfplay_throughput",
    about = "自对弈数据生成速率基准（4x8 / 4x4 / 4x2）"
)]
struct Args {
    /// ONNX 模型路径（双方选手共用同一模型）
    #[arg(long)]
    model: String,
    /// 变体：4x8 / 4x4 / 4x2
    #[arg(long, default_value = "4x2")]
    variant: String,
    /// MCTS 模拟次数
    #[arg(long, default_value_t = 64)]
    sims: usize,
    /// Gumbel Top-K 候选动作数
    #[arg(long, default_value_t = 16)]
    mca: usize,
    /// 每批局数
    #[arg(long, default_value_t = 240)]
    games: usize,
    /// 线程数（0 = CPU 核数）
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 计时批数（不含预热）
    #[arg(long, default_value_t = 3)]
    batches: usize,
    /// 预热批数（结果丢弃）
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// 推理设备：cpu / auto
    #[arg(long, default_value = "cpu")]
    device: String,
    /// 选手类型：onnx（双方同网，默认）/ random（绕开 ONNX，用于隔离环境与搜索层）
    #[arg(long, default_value = "onnx")]
    players: String,
    /// ONNX 会话数（并发推理通道数；0 = 自动 = 线程数）
    #[arg(long, default_value_t = 0)]
    sessions: usize,
    /// 固定种子（缺省不固定）
    #[arg(long)]
    seed: Option<u64>,
}

fn build_config(args: &Args) -> SelfPlayConfig {
    // 与 bin/collector.rs main() 中的生产配置逐字段对齐。
    SelfPlayConfig {
        mcts_sims: args.sims,
        max_considered_actions: args.mca,
        scenario: ScenarioType::Standard,
        c_scale: 1.0,
        gumbel_scale: 1.0,
        playout_cap_random_enabled: false,
        fast_mcts_sims: 0,
        full_search_prob: 0.25,
        ..Default::default()
    }
}

fn run_games<G>(
    model: &Option<Arc<OnnxModel>>,
    players: &str,
    config: &SelfPlayConfig,
    n_games: usize,
    seed: Option<u64>,
    pool: &rayon::ThreadPool,
) -> banqi_collector::pipeline::self_play::MatchResult
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Send + Sync + Default + 'static,
{
    let (spec_a, spec_b) = if players == "random" {
        (PlayerSpec::Random, PlayerSpec::Random)
    } else {
        let m = model.as_ref().expect("onnx 选手需要已加载的模型");
        (
            PlayerSpec::ModelEval(Arc::new(OnnxEvaluator::<G>::new(Arc::clone(m)))),
            PlayerSpec::ModelEval(Arc::new(OnnxEvaluator::<G>::new(Arc::clone(m)))),
        )
    };
    run_match_core(MatchParams {
        player_a: &spec_a,
        player_b: &spec_b,
        n_games,
        config,
        seed,
        record_episodes: true,
        model_sims: config.mcts_sims,
        thread_pool: Some(pool),
        make_env: Arc::new(G::default),
    })
}

/// 与 bin/collector.rs::episodes_gz 等价的序列化耗时测量。
fn episodes_gz(episodes: &[GameEpisode]) -> Result<Vec<u8>> {
    use std::io::Write;
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    for ep in episodes {
        writeln!(gz, "{}", episode_to_dict_json(ep)).context("序列化 episode 失败")?;
    }
    gz.finish().context("gzip 收尾失败")
}

fn dispatch(
    args: &Args,
    model: &Option<Arc<OnnxModel>>,
    config: &SelfPlayConfig,
    n_games: usize,
    seed: Option<u64>,
    pool: &rayon::ThreadPool,
) -> Result<banqi_collector::pipeline::self_play::MatchResult> {
    let r = match args.variant.as_str() {
        "4x8" => run_games::<DarkChessEnv>(model, &args.players, config, n_games, seed, pool),
        "4x4" => run_games::<Game4x4Env>(model, &args.players, config, n_games, seed, pool),
        "4x2" => run_games::<MiniDarkChessEnv>(model, &args.players, config, n_games, seed, pool),
        other => return Err(anyhow!("未知变体: {other}（可选 4x8 / 4x4 / 4x2）")),
    };
    if r.episodes.is_empty() {
        return Err(anyhow!("自对弈 0 局产出（检查模型与配置）"));
    }
    Ok(r)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = build_config(&args);
    let threads = if args.threads > 0 {
        args.threads
    } else {
        num_cpus::get()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("构建 rayon 线程池失败")?;
    let sessions = if args.sessions > 0 { args.sessions } else { threads };

    let t_load = Instant::now();
    let model = if args.players == "random" {
        None
    } else {
        Some(Arc::new(
            OnnxModel::with_sessions(&args.model, &args.device, sessions)
                .map_err(|e| anyhow!("ONNX 模型加载失败 ({}): {e}", args.model))?,
        ))
    };
    let load_s = t_load.elapsed().as_secs_f64();

    println!(
        "CONFIG variant={} sims={} mca={} games={} threads={} batches={} warmup={} device={} players={} sessions={sessions} record=true",
        args.variant,
        config.mcts_sims,
        config.max_considered_actions,
        args.games,
        pool.current_num_threads(),
        args.batches,
        args.warmup,
        args.device,
        args.players,
    );
    println!("METRIC stage=model_load elapsed_s={load_s:.4}");

    for i in 0..args.warmup {
        let _ = dispatch(&args, &model, &config, args.games, args.seed, &pool)?;
        println!("METRIC stage=warmup batch={i}");
    }

    for i in 0..args.batches {
        let t0 = Instant::now();
        let r = dispatch(&args, &model, &config, args.games, args.seed, &pool)?;
        let gen_s = t0.elapsed().as_secs_f64();

        let n_games = r.episodes.len();
        let steps: usize = r.episodes.iter().map(|e| e.game_length).sum();
        let samples: usize = r.episodes.iter().map(|e| e.samples.len()).sum();

        let t1 = Instant::now();
        let gz = episodes_gz(&r.episodes)?;
        let ser_s = t1.elapsed().as_secs_f64();

        let gen_games_s = n_games as f64 / gen_s;
        let gen_samples_s = samples as f64 / gen_s;
        let e2e_games_s = n_games as f64 / (gen_s + ser_s);
        println!(
            "METRIC stage=batch batch={i} games={n_games} steps={steps} samples={samples} \
             avg_moves={avg_moves:.2} gen_s={gen_s:.4} gen_games_s={gen_games_s:.3} \
             gen_samples_s={gen_samples_s:.3} ser_s={ser_s:.4} gz_bytes={gz_bytes} \
             end_to_end_games_s={e2e_games_s:.3} w={wins} d={draws} l={losses}",
            avg_moves = r.avg_moves,
            gen_s = gen_s,
            ser_s = ser_s,
            gz_bytes = gz.len(),
            wins = r.wins,
            draws = r.draws,
            losses = r.losses,
        );
    }

    Ok(())
}
