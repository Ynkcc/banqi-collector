// src/pipeline/self_play/batched.rs - 批量（流水线）自对弈
//
// 同时驱动 `concurrency` 局游戏，把各棵 MCTS 树的叶子评估合并成一个大 batch 送给
// evaluator，提升「单次评估的算力利用率」——因此受益的是 GPU 与大网络；CPU + 小网络
// （如 4x2）下固定同步开销大于收益，应保持单树路径（由 `SelfPlayConfig.batched_variants`
// 按变体选择，见该字段注释与 ARCHITECTURE.md）。
//
// 移植自 rust_4x8/src/pipeline/self_play/batched.rs（原仓库保留该实现，勿改），适配点：
//   - 领域核心路径 crate::core → banqi_core::core；
//   - finalize 走本 crate 的 `self_play::finalize`；评估器复用 `match_core::PlayerEval`；
//   - `make_env` 改为 `Arc` 工厂（支持课程学习 `initial_revealed` 的闭包注入）；
//   - 支持固定 seed（第 i 局 = seed + i，与单树路径一致）；
//   - 返回 `Vec<GameOutcome>`（与单树路径统一口径），由 `run_match_core` 汇总 MatchResult；
//   - 推理失败（`Evaluator::evaluate` 返回 `Err`）时作废本批涉及的局，与单树路径
//     「推理失败 → 本局作废、不产出 episode」语义一致。
//
// 与单树路径的语义差异（调用方须知情）：
//   1. A/B 共用同一评估器 → 仅可用于**双方为同一模型**的自对弈（collector 的 selfplay
//      任务恒满足：同一 network_sha）；异构对手请走单树路径。
//   2. 未接算力随机化（PCR）：所有样本标记 `is_full_search = true`。
//
// 吞吐优化——流水线：评估器跑在后台线程上，主线程负责 MCTS 选择/回填；主线程把一批
// 叶子交给后台后立即去推进其他未阻塞的树，把 CPU 遍历与推理重叠起来。
//
// 正确性：任何树的叶子在被评估/回填前该树不会继续前进；不同树互不依赖，可安全并发推进。

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use banqi_core::core::env::seed::SeedableEnv;
use banqi_core::core::env::{GameEnv, Player, ResNetObservation};
use banqi_core::core::mcts::{
    BatchedTree, Evaluator, GumbelConfig, PendingEval, health_logits_expectation,
};

use super::SelfPlayConfig;
use super::finalize_episode;
use super::match_core::{GameOutcome, outcome_from_episode};

/// 单局样本元组，与 `GameEpisode.samples` 的元素类型一致。
type SampleTuple = (
    ResNetObservation,
    Vec<f32>,
    f32,
    f32,
    u32,
    Player,
    Vec<i32>,
    usize,
    bool,
);

/// 评估请求：一批待评估环境。
struct EvalRequest<G: GameEnv> {
    id: u64,
    envs: Vec<G>,
}

/// 评估响应：按请求顺序返回 logits 与 values（含可选的血量分桶 logits）。
/// `failed = true` 表示该批推理失败（作废），`logits` 为空。
struct EvalResponse {
    id: u64,
    failed: bool,
    logits: Vec<Vec<f32>>,
    values: Vec<f32>,
    health: Option<Vec<Vec<f32>>>,
}

/// 共享请求队列（多消费者）：多个评估线程从这里取批。
struct EvalQueue<G: GameEnv> {
    reqs: Mutex<VecDeque<EvalRequest<G>>>,
    cvar: Condvar,
}

impl<G: GameEnv> Default for EvalQueue<G> {
    fn default() -> Self {
        Self {
            reqs: Mutex::new(VecDeque::new()),
            cvar: Condvar::new(),
        }
    }
}

impl<G: GameEnv> EvalQueue<G> {
    fn push(&self, req: EvalRequest<G>) {
        let mut q = self.reqs.lock().unwrap();
        q.push_back(req);
        self.cvar.notify_one();
    }

    /// 关闭：唤醒所有等待线程，使其退出。
    fn shutdown(&self) {
        self.cvar.notify_all();
    }
}

/// 后台评估线程：循环从共享队列取批、评估、回传结果。
///
/// 队列关闭（shutdown）后 `pop` 仍返回 Some 的话会空转，因此用 `stopped` 标志。
fn eval_worker<G: GameEnv, E: Evaluator<G> + Sync>(
    evaluator: &E,
    queue: &EvalQueue<G>,
    tx: Sender<EvalResponse>,
    stopped: &Arc<Mutex<bool>>,
) {
    loop {
        let req = {
            let mut q = queue.reqs.lock().unwrap();
            loop {
                if *stopped.lock().unwrap() {
                    return;
                }
                if let Some(req) = q.pop_front() {
                    break req;
                }
                match queue.cvar.wait(q) {
                    Ok(guard) => q = guard,
                    Err(_) => return, // 锁中毒：无人会再唤醒，退出以免死锁
                }
            }
        };
        let resp = match evaluator.evaluate(&req.envs) {
            Ok(out) => EvalResponse {
                id: req.id,
                failed: false,
                logits: out.logits,
                values: out.values,
                health: out.health,
            },
            Err(e) => {
                eprintln!("❌ 批量自对弈推理失败（批 {}），本批涉及的局作废: {e}", req.id);
                EvalResponse { id: req.id, failed: true, logits: Vec::new(), values: Vec::new(), health: None }
            }
        };
        if tx.send(resp).is_err() {
            break;
        }
    }
}

/// 运行批量自对弈，返回 `n_games` 局的对局结果（顺序为完成顺序）。
///
/// - `evaluator`：评估器（A/B 共用，故要求双方同一模型；要求 `Sync` 供后台线程共享）
/// - `config`：自对弈配置（mcts_sims / max_considered_actions / c_scale / gumbel_scale / health_*）
/// - `concurrency`：同时推进的对局数（越大单批 batch 越大，流水线也越深）
/// - `seed`：Some 时第 i 局固定为 `seed + i`；None 用随机布局
pub(crate) fn run_batched_games<G, E>(
    evaluator: &E,
    config: &SelfPlayConfig,
    n_games: usize,
    concurrency: usize,
    seed: Option<u64>,
    make_env: &Arc<dyn Fn() -> G + Send + Sync>,
) -> Vec<GameOutcome>
where
    G: GameEnv + SeedableEnv,
    E: Evaluator<G> + Sync,
{
    let concurrency = concurrency.max(1);
    let gumbel_cfg = GumbelConfig {
        num_simulations: config.mcts_sims,
        max_considered_actions: config.max_considered_actions,
        c_scale: config.c_scale,
        gumbel_scale: config.gumbel_scale,
        health_enabled: config.health_enabled,
        health_weight: config.health_weight,
        health_confidence_exp: config.health_confidence_exp,
    };

    // 共享请求队列 + 响应通道
    let queue = Arc::new(EvalQueue::<G>::default());
    let stopped = Arc::new(Mutex::new(false));
    let (resp_tx, resp_rx) = channel::<EvalResponse>();

    // 评估 worker 数量：与并发对局数挂钩，但限制上限，避免过多线程竞争同一会话池
    let num_workers = concurrency.min(8).max(1);

    // 由于需要借用 `&evaluator`（非 'static），使用 scoped 线程。
    // 主循环结束后必须调用 `queue.shutdown()` 唤醒所有 worker 退出，
    // 否则 `thread::scope` 在 join 时死锁。
    thread::scope(move |scope| {
        for _ in 0..num_workers {
            let q = Arc::clone(&queue);
            let st = Arc::clone(&stopped);
            let tx = resp_tx.clone();
            scope.spawn(move || eval_worker(evaluator, q.as_ref(), tx, &st));
        }

        let mut games: Vec<GameOutcome> = Vec::with_capacity(n_games);

        // 分批（wave）推进：每波启动 concurrency 局新游戏，全部结束后进入下一波。
        // 对局索引用全局序号（`games.len()`），保证 seed 与先后手分配与单树路径一致。
        while games.len() < n_games {
            let wave = concurrency.min(n_games - games.len());
            let wave_base = games.len();

            // 初始化本波的游戏树 + 每局的样本收集
            let mut trees: Vec<BatchedTree<'_, G, E>> = Vec::with_capacity(wave);
            let mut episode_data: Vec<Vec<SampleTuple>> = Vec::with_capacity(wave);
            for i in 0..wave {
                let mut env = (make_env)();
                if let Some(s) = seed {
                    env.set_seed(s.wrapping_add((wave_base + i) as u64));
                }
                trees.push(BatchedTree::new(&env, evaluator, &gumbel_cfg));
                episode_data.push(Vec::new());
            }
            let mut active: Vec<bool> = vec![true; wave];
            // 推理失败作废的局：不产出 episode，也不参与胜/和/负统计
            let mut abandoned: Vec<bool> = vec![false; wave];
            // 每棵树是否正等待一个在途批（Some(batch_id)），等待期间不得继续选择
            let mut blocked: Vec<Option<u64>> = vec![None; wave];
            // 在途批：batch_id -> (每个 eval 属于哪棵树, 待评估项)
            let mut in_flight: HashMap<u64, (Vec<usize>, Vec<PendingEval<G>>)> = HashMap::new();
            let mut next_batch_id: u64 = 0;

            // 交替推进各树，直到本波全部结束
            while active.iter().any(|&a| a) {
                // 1) 完成已就绪（Ready）且未被阻塞的决策
                for i in 0..wave {
                    if !active[i] || blocked[i].is_some() {
                        continue;
                    }
                    if trees[i].finalize_step() {
                        if let Some(r) = &trees[i].result {
                            episode_data[i].push((
                                r.state.clone(),
                                r.improved_policy.clone(),
                                r.mcts_value,
                                r.completed_q,
                                r.root_visit_count,
                                r.player,
                                r.action_mask.clone(),
                                r.action,
                                true, // 批量路径未做算力随机化，全部视为 Full Search
                            ));
                        }
                        if trees[i].game_over {
                            active[i] = false;
                        } else {
                            trees[i].start_next_step();
                        }
                    }
                }

                // 2) 从所有未被阻塞的活跃树上收集待评估项，合并成一批
                let mut pool: Vec<PendingEval<G>> = Vec::new();
                let mut pool_targets: Vec<usize> = Vec::new();
                let mut touched: Vec<usize> = Vec::new();
                for i in 0..wave {
                    if !active[i] || blocked[i].is_some() {
                        continue;
                    }
                    let mut local: Vec<PendingEval<G>> = Vec::new();
                    if trees[i].collect(&mut local) {
                        for p in local {
                            pool.push(p);
                            pool_targets.push(i);
                        }
                        touched.push(i);
                    }
                }

                // 3) 有收集到东西：提交给后台线程评估（非阻塞），并阻塞涉及到的树
                if !pool.is_empty() {
                    let batch_id = next_batch_id;
                    next_batch_id += 1;
                    let envs: Vec<G> = pool.iter().map(|p| p.env).collect();
                    queue.push(EvalRequest { id: batch_id, envs });
                    for &t in &touched {
                        blocked[t] = Some(batch_id);
                    }
                    in_flight.insert(batch_id, (pool_targets, pool));
                }

                // 4) 尽力 drain 后台线程已返回的批并回填（非阻塞轮询）
                while let Ok(resp) = resp_rx.try_recv() {
                    apply_response(
                        &mut trees, &mut active, &mut abandoned, &mut blocked, &mut in_flight, resp,
                    );
                }

                // 5) 没有任何树可推进（要么全结束，要么全部在等待在途批）→
                //    若还有在途批，阻塞等待一个结果后再继续，避免空转；
                //    若无在途批也无活跃树则结束本波。
                let has_active = active.iter().any(|&a| a);
                let any_unblocked = (0..wave).any(|i| active[i] && blocked[i].is_none());
                if has_active && !any_unblocked && !in_flight.is_empty() {
                    // 主线程在此阻塞，等待后台线程返回任意一个结果，然后继续
                    match resp_rx.recv() {
                        Ok(resp) => apply_response(
                            &mut trees, &mut active, &mut abandoned, &mut blocked, &mut in_flight,
                            resp,
                        ),
                        Err(_) => break,
                    }
                } else if !has_active {
                    break;
                }
                // 其余情况（仍有未阻塞树，或刚从阻塞中被唤醒）→ 回到循环顶继续
            }

            // 收尾：把本波完成的局 finalize 成 GameOutcome。
            // 空样本局与作废局同样计入对局数，只是不产出 episode —— 与单树路径
            // 「episodes 只收非空局」口径一致。
            for i in 0..wave {
                let player_a_is_red = ((wave_base + i) % 2) == 0;
                if abandoned[i] || episode_data[i].is_empty() {
                    games.push(GameOutcome { result: 0, moves: 0, episode: None, nnue_episode: None });
                    continue;
                }
                let winner = trees[i].step_outcome.2;
                let health_diff_red = trees[i]
                    .tree
                    .root_env()
                    .and_then(|e| e.terminal_health_diff_red());
                let ep = finalize_episode(
                    std::mem::take(&mut episode_data[i]),
                    winner,
                    health_diff_red,
                    None,
                );
                games.push(outcome_from_episode(ep, player_a_is_red));
            }
        }

        // 全部对局完成：置停止标志并唤醒所有 worker 退出（否则 scope 会死锁）
        *stopped.lock().unwrap() = true;
        queue.shutdown();

        games
    })
}

/// 把一批评估结果按树分组回填，并解除对应树的阻塞。
///
/// 推理失败（`resp.failed`）→ 本批涉及的局作废：标记 `abandoned` 并停止推进，
/// 与单树路径「推理失败 → 本局作废、不写入训练数据」一致；这同时避免「重新收集同
/// 一个叶子、再失败、再收集」的空转死循环。
/// 结果数量与待评估项不一致（后端返回残缺）→ 视为同一类失败处理。
fn apply_response<G, E>(
    trees: &mut [BatchedTree<'_, G, E>],
    active: &mut [bool],
    abandoned: &mut [bool],
    blocked: &mut [Option<u64>],
    in_flight: &mut HashMap<u64, (Vec<usize>, Vec<PendingEval<G>>)>,
    resp: EvalResponse,
) where
    G: GameEnv,
    E: Evaluator<G>,
{
    let Some((targets, evals)) = in_flight.remove(&resp.id) else {
        return;
    };
    let degrade = resp.failed || resp.logits.len() != evals.len();
    if degrade {
        if !resp.failed {
            eprintln!(
                "⚠️ batched_self_play: 批 {} 结果数量 {} != 待评估 {}，本批涉及的局作废",
                resp.id,
                resp.logits.len(),
                evals.len()
            );
        }
        for &t in &targets {
            abandoned[t] = true;
            active[t] = false;
            blocked[t] = None;
        }
        return;
    }

    let mut by_tree: HashMap<usize, Vec<usize>> = HashMap::new();
    for (k, &t) in targets.iter().enumerate() {
        by_tree.entry(t).or_default().push(k);
    }
    for (t, idxs) in by_tree {
        if !active[t] {
            continue;
        }
        let mut applied: Vec<(&PendingEval<G>, &[f32], f32, f32)> = Vec::with_capacity(idxs.len());
        for &k in &idxs {
            let health = health_logits_expectation(resp.health.as_deref(), k).unwrap_or(0.0);
            applied.push((&evals[k], &resp.logits[k], resp.values[k], health));
        }
        trees[t].apply(&applied);
        blocked[t] = None;
    }
}
