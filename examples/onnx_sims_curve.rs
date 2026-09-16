// banqi-collector/examples/onnx_sims_curve.rs
// 离线胜率曲线：ONNX 选手（选手 A）在 4x2 变体下的「MCTS 模拟数 → 胜/和/负」。
//
// 复用 pipeline::self_play::run_match_core 的评估路径（record_episodes=false），
// 逐局换色（A 在第 i 局执红当 i 为偶数），因此胜率不含先手偏差。
//
// 用法（须带 onnx feature）：
//   cargo run --release --features onnx --example onnx_sims_curve -- \
//     --model ../banqi-training/outputs/4x2/checkpoints/random_health.onnx \
//     --opponent random --sims 64,256,1024,4096 --games 40
//
// 纯策略验收（不搜索，policy head 掩码后 argmax）：
//   cargo run --release --features onnx --example onnx_sims_curve -- \
//     --model <model.onnx> --opponent random --policy-only --games 100
//
// 注意：评估路径的 MCTS 参数由 match_core::model_mcts_action 设定：
// c_scale 取自 SelfPlayConfig.c_scale（默认 1.0，与自对弈生成路径同口径，
// 可用 --config 或 --mcts-sims 之外的 selfplay.c_scale 覆盖）、
// max_considered_actions 固定为 16。本曲线用于判断「搜索规模 → 强度」的
// 趋势与饱和点。

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use rayon::ThreadPoolBuilder;

use banqi_collector::pipeline::self_play::{
    MatchParams, PlayerSpec, SelfPlayConfig, run_match_core,
};
use banqi_core::core::env::variants::MiniDarkChessEnv;
use banqi_core::core::env::{DarkChessEnv, Player};
use banqi_core::core::expectimax::ExpectimaxEngine;
use banqi_core::core::expectimax::nnue::{NnueAccumulator, NnueEvaluate};
use banqi_engine::inference::onnx::{OnnxEvaluator, OnnxModel};

/// Expectimax 搜索强制要求 NNUE 叶评估；本地无 4x2 `.nnue` 时用「血量差」代理
/// 实例化引擎（仅用于基准，强度不等价于训练过的 NNUE）。
#[derive(Debug)]
struct MaterialEval;

fn hp_diff(env: &DarkChessEnv, me: Player) -> f32 {
    let opp = me.opposite();
    let scale = env.config.initial_health as f32;
    (env.get_hp(me) - env.get_hp(opp)) as f32 / scale
}

impl NnueEvaluate for MaterialEval {
    fn evaluate(&self, env: &DarkChessEnv) -> f32 {
        hp_diff(env, env.get_current_player())
    }

    fn validate_feature_dim(&self, _expected: usize) -> Result<(), String> {
        Ok(())
    }

    fn init_accumulator(self: Arc<Self>, env: &DarkChessEnv) -> Box<dyn NnueAccumulator> {
        Box::new(MaterialAccumulator {
            hp: [env.get_hp(Player::Red), env.get_hp(Player::Black)],
            scale: env.config.initial_health as f32,
        })
    }
}

#[derive(Clone, Debug)]
struct MaterialAccumulator {
    hp: [i32; 2],
    scale: f32,
}

impl NnueAccumulator for MaterialAccumulator {
    fn clone_box(&self) -> Box<dyn NnueAccumulator> {
        Box::new(self.clone())
    }

    fn apply_step(&mut self, _before: &DarkChessEnv, after: &DarkChessEnv, _action: usize) {
        self.hp = [after.get_hp(Player::Red), after.get_hp(Player::Black)];
    }

    fn evaluate(&self, player: Player) -> f32 {
        let (me, opp) = (player.idx(), player.opposite().idx());
        (self.hp[me] - self.hp[opp]) as f32 / self.scale
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "onnx_sims_curve",
    about = "ONNX 模型「搜索次数 → 胜/和/负」基准曲线（4x2 变体）"
)]
struct Args {
    /// 被测 ONNX 模型（选手 A）
    #[arg(long)]
    model: String,
    /// 对手类型：random / engine / onnx
    #[arg(long, default_value = "random")]
    opponent: String,
    /// opponent=onnx 时的对手模型（注意：与选手 A 共用同一模拟数）
    #[arg(long)]
    opponent_model: Option<String>,
    /// opponent=engine 时的 .nnue 权重（缺省用血量差材质评估代理）
    #[arg(long)]
    nnue: Option<String>,
    /// opponent=engine 的节点预算
    #[arg(long, default_value_t = 300_000)]
    engine_budget: u64,
    /// 选手 A 的模拟数列表（逗号分隔）
    #[arg(long, default_value = "64,256,1024,4096")]
    sims: String,
    /// 选手 B（对手）的模拟数。缺省 = 与 --sims 相同；用于「同一模型、不同搜索深度」
    /// 的 Elo 阶梯测量（B4：搜索深度 → 老师强度）。
    #[arg(long)]
    opponent_sims: Option<usize>,
    /// 每档对局数（逐局换色）
    #[arg(long, default_value_t = 40)]
    games: usize,
    /// 线程数（0 = CPU 核数）
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 固定种子（缺省不固定）
    #[arg(long)]
    seed: Option<u64>,
    /// 被测选手 A 改用纯策略（policy head argmax，无搜索）；此时 --sims 仅取首档占位
    #[arg(long)]
    policy_only: bool,
}

fn build_opponent(args: &Args) -> Result<PlayerSpec<MiniDarkChessEnv>> {
    match args.opponent.as_str() {
        "random" => Ok(PlayerSpec::Random),
        "engine" => {
            let evaluator: Arc<dyn NnueEvaluate> = match &args.nnue {
                Some(path) => Arc::new(
                    banqi_engine::nnue::NnueEvaluator::load_from_file(path)
                        .map_err(|e| anyhow!("NNUE 加载失败 ({path}): {e}"))?,
                ),
                None => {
                    println!("[warn] 未提供 --nnue，Engine 对手使用血量差材质叶评估代理");
                    Arc::new(MaterialEval)
                }
            };
            let mut engine = ExpectimaxEngine::with_nnue(evaluator);
            engine.set_node_budget(args.engine_budget);
            Ok(PlayerSpec::Expectimax(Arc::new(engine)))
        }
        "onnx" => {
            let path = args
                .opponent_model
                .as_deref()
                .context("--opponent onnx 需要同时指定 --opponent-model")?;
            let model = Arc::new(
                OnnxModel::new(path, "auto")
                    .map_err(|e| anyhow!("对手 ONNX 模型加载失败 ({path}): {e}"))?,
            );
            Ok(PlayerSpec::ModelEval(Arc::new(
                OnnxEvaluator::<MiniDarkChessEnv>::new(model),
            )))
        }
        other => Err(anyhow!("未知对手: {other}（可选 random / engine / onnx）")),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let sims_list: Vec<usize> = args
        .sims
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .with_context(|| format!("模拟数解析失败: {s:?}"))
        })
        .collect::<Result<_>>()?;
    if sims_list.is_empty() || sims_list.contains(&0) {
        return Err(anyhow!("--sims 需为非零模拟数列表，如 64,256,1024"));
    }
    // 纯策略模式没有搜索次数维度，压成单档，避免重复打同样的对局。
    let sims_list: Vec<usize> = if args.policy_only {
        vec![sims_list[0]]
    } else {
        sims_list
    };

    let model_a = Arc::new(
        OnnxModel::new(&args.model, "auto")
            .map_err(|e| anyhow!("ONNX 模型加载失败 ({}): {e}", args.model))?,
    );
    let eval_a = Arc::new(OnnxEvaluator::<MiniDarkChessEnv>::new(model_a));
    let spec_a = if args.policy_only {
        PlayerSpec::PolicyArgmax(eval_a)
    } else {
        PlayerSpec::ModelEval(eval_a)
    };
    let spec_b = build_opponent(&args)?;

    let pool = ThreadPoolBuilder::new()
        .num_threads(if args.threads > 0 {
            args.threads
        } else {
            num_cpus::get()
        })
        .build()
        .context("构建 rayon 线程池失败")?;

    let config = SelfPlayConfig {
        mcts_sims: sims_list[0],
        max_considered_actions: 16,
        c_scale: 1.0,
        gumbel_scale: 1.0,
        ..Default::default()
    };
    let make_env: Arc<dyn Fn() -> MiniDarkChessEnv + Send + Sync> =
        Arc::new(MiniDarkChessEnv::default);

    println!(
        "选手 A（被测）: {}{}",
        args.model,
        if args.policy_only { " [纯策略 argmax]" } else { "" }
    );
    println!(
        "选手 B（对手）: {}{}",
        args.opponent,
        args.nnue
            .as_deref()
            .map(|p| format!(" (nnue={p}, budget={})", args.engine_budget))
            .unwrap_or_default()
    );
    println!(
        "每档 {} 局、逐局换色；线程池 {}",
        args.games,
        pool.current_num_threads()
    );
    println!();
    println!("| 策略 | 胜 | 和 | 负 | 胜率 | 得分率 | 不输率 | 平均步数 | 用时(s) |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");

    for &sims in &sims_list {
        let started = Instant::now();
        let r = run_match_core(MatchParams {
            player_a: &spec_a,
            player_b: &spec_b,
            n_games: args.games,
            config: &config,
            seed: args.seed,
            record_episodes: false,
            batched: false, // 评估路径不产生 episode，批量（记录模式专用）不适用
            model_sims: sims,
            opponent_sims: args.opponent_sims,
            thread_pool: Some(&pool),
            make_env: make_env.clone(),
        });
        let elapsed = started.elapsed().as_secs_f64();
        let n = args.games.max(1) as f32;
        let label = if args.policy_only {
            "纯策略".to_string()
        } else {
            sims.to_string()
        };
        println!(
            "| {} | {} | {} | {} | {:.1}% | {:.1}% | {:.1}% | {:.1} | {:.1} |",
            label,
            r.wins,
            r.draws,
            r.losses,
            100.0 * r.wins as f32 / n,
            100.0 * (r.wins as f32 + 0.5 * r.draws as f32) / n,
            100.0 * (r.wins as f32 + r.draws as f32) / n,
            r.avg_moves,
            elapsed
        );
    }

    Ok(())
}
