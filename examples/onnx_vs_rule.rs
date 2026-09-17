// banqi-collector/examples/onnx_vs_rule.rs
// 绝对强度阶梯评测 CLI：ONNX 模型 vs 规则/内建对手（优先吃子 / 优先翻棋 / 随机）。
//
// 为什么要它：相对门禁（gatekeeper：candidate vs 当前 best + GSPRT）只能回答「这代比
// 上代强吗」，结构上测不出「所有版本都打不过一个 3 行的优先吃子启发式」。本 CLI 提供
// 绝对强度口径 —— 对阵固定对手的胜率阶梯，可用于训练前基线、单发复现、CI 回归，
// 与调度器侧的 TASK_EVAL 自动评测（同一条 run_match_core 链路）互为对照。
//
// 规则对手直接复用库内实现（pipeline::self_play::rule_opponents），不在此另写适配，
// 避免 CLI 与调度器评测路径两套语义漂移。
//
// 用法（须带 onnx feature）：
//   cargo run --release --features onnx --example onnx_vs_rule -- \
//     --variant 4x8 --model <model.onnx> --games 1000 --seed 1
//   cargo run --release --features onnx --example onnx_vs_rule -- \
//     --model <model.onnx> --opponents rule:capture_first --sims 0,64,256 --games 300
//   cargo run --release --features onnx --example onnx_vs_rule -- ... --json
//
// ⚠️ --variant 必须与模型的动作空间一致，否则评估器维度不符。

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use clap::Parser;
use rayon::ThreadPoolBuilder;
use serde_json::json;

use banqi_collector::pipeline::self_play::{
    AsDarkChessRef, MatchParams, PlayerSpec, RuleOpponent, SeedableEnv, SelfPlayConfig,
    eval_opponent_spec, run_match_core,
};
use banqi_core::core::env::variants::{Game4x4Env, MiniDarkChessEnv};
use banqi_core::core::env::{DarkChessEnv, GameEnv};
use banqi_engine::inference::onnx::{OnnxEvaluator, OnnxModel};

#[derive(Parser, Debug)]
#[command(
    name = "onnx_vs_rule",
    about = "绝对强度阶梯评测：ONNX 模型（纯策略/MCTS）vs 规则对手（优先吃子/优先翻棋/随机）"
)]
struct Args {
    /// 被测 ONNX 模型（选手 A）
    #[arg(long)]
    model: String,
    /// 变体 id（4x8 / 4x4 / 4x2）；必须与模型的动作空间一致
    #[arg(long, default_value = "4x8")]
    variant: String,
    /// 对手列表（逗号分隔）：random / rule:capture_first / rule:reveal_first
    #[arg(
        long,
        default_value = "rule:capture_first,rule:reveal_first,random"
    )]
    opponents: String,
    /// 搜索档位（逗号分隔）；0 = 纯策略 argmax（无搜索，门禁主口径）
    #[arg(long, default_value = "0")]
    sims: String,
    /// 每档对局数（逐局换色）
    #[arg(long, default_value_t = 1000)]
    games: usize,
    /// 线程数（0 = CPU 核数）
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 固定环境种子（缺省不固定；注意对手策略自带随机性）
    #[arg(long)]
    seed: Option<u64>,
    /// 纯策略（等价 --sims 0）；显式给出时覆盖 --sims
    #[arg(long)]
    policy_only: bool,
    /// 以 JSON 输出（便于脚本 / 门禁自动化消费）
    #[arg(long)]
    json: bool,
}

/// 一个（对手 × 搜索档位）的评测结果。
struct Level {
    opponent: String,
    sims: usize,
    wins: usize,
    draws: usize,
    losses: usize,
    avg_moves: f32,
    elapsed_s: f64,
}

impl Level {
    fn total(&self) -> f32 {
        (self.wins + self.draws + self.losses).max(1) as f32
    }
    fn win_rate(&self) -> f64 {
        self.wins as f64 / self.total() as f64
    }
    fn score_rate(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.total() as f64
    }
    fn non_loss_rate(&self) -> f64 {
        (self.wins + self.draws) as f64 / self.total() as f64
    }
    fn mode(&self) -> &'static str {
        if self.sims == 0 {
            "policy_argmax"
        } else {
            "mcts"
        }
    }
}

/// 对手标识 → 选手规格（复用库内规则对手实现；random 走内置随机选手）。
fn opponent_spec<G: GameEnv>(spec: &str) -> Result<PlayerSpec<G>> {
    let s = spec.trim();
    if s == "random" {
        return Ok(PlayerSpec::Random);
    }
    match RuleOpponent::parse(s) {
        Some(kind) => Ok(eval_opponent_spec::<G>(kind)),
        None => Err(anyhow!(
            "未知对手 {s}（可选 random / rule:capture_first / rule:reveal_first）"
        )),
    }
}

fn parse_sims(raw: &str, policy_only: bool) -> Result<Vec<usize>> {
    if policy_only {
        return Ok(vec![0]);
    }
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(
            part.parse::<usize>()
                .map_err(|e| anyhow!("模拟数解析失败 {part:?}: {e}"))?,
        );
    }
    if out.is_empty() {
        return Err(anyhow!("--sims 不能为空（0 = 纯策略）"));
    }
    Ok(out)
}

fn run<G>(
    args: &Args,
    opponents: &[String],
    sims_list: &[usize],
    pool: &rayon::ThreadPool,
) -> Result<Vec<Level>>
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Send + Sync + Default + 'static,
{
    let model = Arc::new(
        OnnxModel::new(&args.model, "auto")
            .map_err(|e| anyhow!("ONNX 模型加载失败 ({}): {e}", args.model))?,
    );
    let make_env: Arc<dyn Fn() -> G + Send + Sync> = Arc::new(G::default);
    let mut levels = Vec::new();

    for opponent in opponents {
        for &sims in sims_list {
            let eval_a = Arc::new(OnnxEvaluator::<G>::new(Arc::clone(&model)));
            let spec_a = if sims == 0 {
                PlayerSpec::PolicyArgmax(eval_a)
            } else {
                PlayerSpec::ModelEval(eval_a)
            };
            let spec_b = opponent_spec::<G>(opponent)?;
            let config = SelfPlayConfig {
                mcts_sims: sims.max(1),
                max_considered_actions: 16,
                c_scale: 1.0,
                gumbel_scale: 1.0,
                ..Default::default()
            };

            let started = Instant::now();
            let r = run_match_core(MatchParams {
                player_a: &spec_a,
                player_b: &spec_b,
                n_games: args.games.max(1),
                config: &config,
                seed: args.seed,
                record_episodes: false,
                batched: false,
                model_sims: sims.max(1),
                opponent_sims: None,
                thread_pool: Some(pool),
                make_env: Arc::clone(&make_env),
            });
            let elapsed = started.elapsed().as_secs_f64();
            levels.push(Level {
                opponent: opponent.clone(),
                sims,
                wins: r.wins,
                draws: r.draws,
                losses: r.losses,
                avg_moves: r.avg_moves,
                elapsed_s: elapsed,
            });
        }
    }
    Ok(levels)
}

fn print_table(args: &Args, levels: &[Level], threads: usize) {
    println!("被测模型: {} [{}]", args.model, args.variant);
    println!(
        "每档 {} 局、逐局换色；线程池 {}；--sims 0 = 纯策略 argmax",
        args.games.max(1),
        threads
    );
    println!();
    println!("| 对手 | 模式 | 胜 | 和 | 负 | 胜率 | 得分率 | 不输率 | 平均步数 | 用时(s) |");
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for l in levels {
        let mode = if l.sims == 0 {
            "纯策略".to_string()
        } else {
            format!("MCTS@{}", l.sims)
        };
        println!(
            "| {} | {} | {} | {} | {} | {:.1}% | {:.1}% | {:.1}% | {:.1} | {:.1} |",
            l.opponent,
            mode,
            l.wins,
            l.draws,
            l.losses,
            100.0 * l.win_rate(),
            100.0 * l.score_rate(),
            100.0 * l.non_loss_rate(),
            l.avg_moves,
            l.elapsed_s
        );
    }
}

fn print_json(args: &Args, levels: &[Level], threads: usize) -> Result<()> {
    let results: Vec<_> = levels
        .iter()
        .map(|l| {
            json!({
                "opponent": l.opponent,
                "mode": l.mode(),
                "sims": l.sims,
                "wins": l.wins,
                "draws": l.draws,
                "losses": l.losses,
                "win_rate": l.win_rate(),
                "score_rate": l.score_rate(),
                "non_loss_rate": l.non_loss_rate(),
                "avg_moves": l.avg_moves,
                "elapsed_s": l.elapsed_s,
            })
        })
        .collect();
    let view = json!({
        "model": args.model,
        "variant": args.variant,
        "games_per_level": args.games.max(1),
        "seed": args.seed,
        "threads": threads,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&view)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sims_list = parse_sims(&args.sims, args.policy_only)?;
    let opponents: Vec<String> = args
        .opponents
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if opponents.is_empty() {
        return Err(anyhow!("--opponents 不能为空"));
    }
    // 对手标识先行校验：避免跑完一半才发现拼错（规则解析在泛型上下文里做）
    for spec in &opponents {
        if spec != "random" && RuleOpponent::parse(spec).is_none() {
            return Err(anyhow!(
                "未知对手 {spec}（可选 random / rule:capture_first / rule:reveal_first）"
            ));
        }
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(if args.threads > 0 {
            args.threads
        } else {
            num_cpus::get()
        })
        .build()
        .map_err(|e| anyhow!("构建 rayon 线程池失败: {e}"))?;

    let levels = match args.variant.as_str() {
        "4x8" => run::<DarkChessEnv>(&args, &opponents, &sims_list, &pool)?,
        "4x4" => run::<Game4x4Env>(&args, &opponents, &sims_list, &pool)?,
        "4x2" => run::<MiniDarkChessEnv>(&args, &opponents, &sims_list, &pool)?,
        other => return Err(anyhow!("未知变体: {other}（可选 4x8 / 4x4 / 4x2）")),
    };

    if args.json {
        print_json(&args, &levels, pool.current_num_threads())?;
    } else {
        print_table(&args, &levels, pool.current_num_threads());
    }
    Ok(())
}
