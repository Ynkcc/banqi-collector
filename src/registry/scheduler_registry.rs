// src/registry/scheduler_registry.rs — SchedulerRegistry（分布式 Registry 实现）
//
// 自 rust_4x8/src/registry/scheduler_registry.rs 迁入（banqi-collector 拆分），
// ONNX 模型类型由 crate::inference::onnx 切换为 banqi-engine 的实现。
//
// 语义：get_task（= GetTask，无任务返回 None 由调用方退避轮询）、
// model（= 按 sha 惰性下载 + 按对象键扩展名分派加载，进程内缓存会话池）、
// submit_episode_report（ReportEpisode 签发预签名 PUT → HTTP 直传 R2）、
// submit_match_report（rating 五项计数上报，服务端 GSPRT 判停晋级）。
// 模型换网感知：GetTask 返回的 best sha 变化即触发新模型下载加载。
// 本地缓存路径与权重格式均来自服务端下发的对象键（networks/<sha>.<ext>）。
//
// 并发形态（2026-09-15 重构）：
// - 长连接：`Channel` 只建一次并复用（原先每次 RPC 都重新 connect），
//   tonic 的 Channel 自带重连，调度器重启后无需额外处理。
// - 异步上报：gRPC + EpisodeBatch 编码 + gzip + sha256 + R2 PUT 全部丢到后台 tokio 任务，
//   主循环（拉任务 → rayon 跑批）不再被 I/O 阻塞，计算与上传重叠。
// - 背压：`report_slots` 信号量限制在途批数（同时约束驻留内存），名额用尽时
//   提交侧阻塞，主循环自然不会再拉下一批。
// - 模型预取：心跳发现 best 变化即在后台下载到缓存目录，下次 GetTask 直接命中。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::transport::{Channel, Endpoint};

use banqi_engine::inference::onnx::OnnxModel;

use crate::pb;
use crate::pipeline::self_play::{GameEpisode, NnueEpisode, encode_episode_batch};

use pb::scheduler_service_client::SchedulerServiceClient;
use pb::{EpisodeMeta, HeartbeatRequest, MatchResult as PbMatchResult, NetworkRequest, TaskKind, TaskRequest};

/// 心跳间隔
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// HTTP 超时（模型下载 / R2 直传）
const HTTP_TIMEOUT: Duration = Duration::from_secs(300);

/// 客户端版本声明（来自 crate 版本，不作为配置项）
pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    pub endpoint: String,
    /// worker 标识（空 = 由配置层填 `worker-<pid>`）
    pub worker_id: String,
    pub cache_dir: PathBuf,
    pub device: String,
    /// ONNX 会话数（每 sha 的并发推理通道数；0 = 自动 = 自对弈线程数）。
    /// `OnnxModel` 内部每个会话用互斥锁串行化推理，通道数少于并发对局数时推理成为瓶颈。
    pub sessions: usize,
    /// 允许同时在途的上报批数（背压上限；同时约束驻留内存）
    pub max_inflight_reports: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50051".to_string(),
            worker_id: String::new(),
            cache_dir: PathBuf::from("outputs/distributed_cache"),
            device: "auto".to_string(),
            sessions: 0,
            max_inflight_reports: 2,
        }
    }
}

/// 一次 GetTask 得到的任务（分布式形态，参数由服务端下发）。
#[derive(Debug, Clone)]
pub struct SchedulerTask {
    pub task_id: String,
    pub kind: TaskKind,
    pub network_sha: String,
    pub opponent_sha: String,
    pub games: usize,
    pub mcts_sims: usize,
    pub variant: String,
    /// 课程学习：服务端下发的初始预翻棋子数（extra_config 透传；None = 变体默认值）。
    pub initial_revealed: Option<usize>,
}

/// 一批自对弈 episode 的上报载荷（编码/gzip/sha256 在后台完成）。
pub struct EpisodeBatch {
    pub task_id: String,
    pub network_sha: String,
    /// 变体标识（服务端下发）：编码进记录供训练端校验数据未串变体
    pub variant: String,
    /// 批内聚合胜方（1 红 / -1 黑 / 0 平；调度端仅作日志统计）
    pub winner: i32,
    pub episodes: Vec<GameEpisode>,
    /// Expectimax 强自对弈记录（与 episodes 互斥出现；当前采集路径恒为空）
    pub nnue_episodes: Vec<NnueEpisode>,
}

/// rating 结果上报载荷（五项成对计数）。
pub struct MatchReport {
    pub task_id: String,
    pub network_sha: String,
    pub opponent_sha: String,
    pub games: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub pairs: [usize; 5],
}

pub struct SchedulerRegistry {
    rt: tokio::runtime::Runtime,
    /// 与调度器的长连接（clone 廉价，tonic 内部自动重连）
    channel: Channel,
    http: reqwest::Client,
    cfg: SchedulerConfig,
    /// 本地已加载的模型缓存（sha -> 会话池）
    models: HashMap<String, Arc<OnnxModel>>,
    /// 网络对象键表（sha -> networks/<sha>.<ext>）：本地缓存路径与权重格式来源
    network_keys: HashMap<String, String>,
    /// 当前持有的网络 sha（GetTask 时上报，服务端据此免发下载 URL）
    current_network: String,
    /// 心跳共享状态：累计完成局数
    completed_games: Arc<AtomicU64>,
    /// 心跳共享状态：当前执行中的 task_id
    running_task_id: Arc<Mutex<String>>,
    /// 在途上报名额（背压；同时约束驻留内存）
    report_slots: Arc<Semaphore>,
    report_slots_total: u32,
    /// 累计上报失败批数（进程退出时汇总）
    failed_reports: Arc<AtomicU64>,
    /// 网络下载临时文件名去重计数
    tmp_seq: Arc<AtomicU64>,
}

impl SchedulerRegistry {
    pub fn new(cfg: SchedulerConfig) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("构建 tokio runtime 失败")?;

        let channel = rt.block_on(async {
            Endpoint::from_shared(cfg.endpoint.clone())
                .with_context(|| format!("非法调度器地址: {}", cfg.endpoint))?
                .connect()
                .await
                .with_context(|| format!("连接调度器失败: {}", cfg.endpoint))
        })?;

        let http = reqwest::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .context("构建 HTTP 客户端失败")?;

        let report_slots_total = cfg.max_inflight_reports as u32;
        let report_slots = Arc::new(Semaphore::new(cfg.max_inflight_reports));
        let completed_games = Arc::new(AtomicU64::new(0));
        let running_task_id = Arc::new(Mutex::new(String::new()));
        let failed_reports = Arc::new(AtomicU64::new(0));
        let tmp_seq = Arc::new(AtomicU64::new(0));

        spawn_heartbeat(
            rt.handle().clone(),
            channel.clone(),
            cfg.clone(),
            http.clone(),
            Arc::clone(&tmp_seq),
            Arc::clone(&completed_games),
            Arc::clone(&running_task_id),
        );

        Ok(Self {
            rt,
            channel,
            http,
            cfg,
            models: HashMap::new(),
            network_keys: HashMap::new(),
            current_network: String::new(),
            completed_games,
            running_task_id,
            report_slots,
            report_slots_total,
            failed_reports,
            tmp_seq,
        })
    }

    /// 标记当前执行中的任务（心跳上报 running_task_id）
    pub fn set_running_task(&self, task_id: &str) {
        *self.running_task_id.lock().unwrap() = task_id.to_string();
    }

    /// 累加已完成局数（心跳上报）
    pub fn add_completed_games(&self, n: usize) {
        self.completed_games.fetch_add(n as u64, Ordering::Relaxed);
    }

    /// 累计上报失败批数
    pub fn failed_reports(&self) -> u64 {
        self.failed_reports.load(Ordering::Relaxed)
    }

    fn client(&self) -> SchedulerServiceClient<Channel> {
        SchedulerServiceClient::new(self.channel.clone())
    }

    /// 获取一个在途上报名额；名额用尽则阻塞主线程直到有上报完成（背压）。
    fn acquire_report_slot(&self) -> OwnedSemaphorePermit {
        let slots = Arc::clone(&self.report_slots);
        self.rt
            .block_on(slots.acquire_owned())
            .expect("上报信号量不会被关闭")
    }

    /// 等待全部在途上报结束（进程退出前调用，避免丢数据）。
    pub fn flush_reports(&self) {
        let slots = Arc::clone(&self.report_slots);
        let total = self.report_slots_total;
        // 持有全部名额即代表在途上报已清空；随后 drop 立即释放。
        let _all = self
            .rt
            .block_on(slots.acquire_many_owned(total))
            .expect("上报信号量不会被关闭");
    }

    /// 提交一批 episode 上报：立即返回，序列化与网络传输在后台 tokio 任务中完成。
    pub fn submit_episode_report(&self, batch: EpisodeBatch) {
        let permit = self.acquire_report_slot();
        let http = self.http.clone();
        let mut client = self.client();
        let worker_id = self.cfg.worker_id.clone();
        let task_id = batch.task_id.clone();
        let failed_reports = Arc::clone(&self.failed_reports);

        self.rt.spawn(async move {
            let _permit = permit;
            match report_episode_async(&http, &mut client, &worker_id, batch).await {
                Ok((games, steps, object_key)) => println!(
                    "[scheduler] ✅ episode 已直传: task={task_id} games={games} steps={steps} -> {object_key}"
                ),
                Err(e) => {
                    failed_reports.fetch_add(1, Ordering::Relaxed);
                    eprintln!("[scheduler] ⚠️ episode 上报失败（本批丢弃）: task={task_id} {e:#}");
                }
            }
        });
    }

    /// 提交 rating 结果上报（报文很小，同样异步，主线程不等待网络）。
    pub fn submit_match_report(&self, rep: MatchReport) {
        let permit = self.acquire_report_slot();
        let mut client = self.client();
        let worker_id = self.cfg.worker_id.clone();
        let task_id = rep.task_id.clone();
        let failed_reports = Arc::clone(&self.failed_reports);

        self.rt.spawn(async move {
            let _permit = permit;
            match report_match_async(&mut client, &worker_id, rep).await {
                Ok((concluded, promoted, best)) => println!(
                    "[scheduler] ✅ match result: task={task_id} concluded={concluded} promoted={promoted} best={best}"
                ),
                Err(e) => {
                    failed_reports.fetch_add(1, Ordering::Relaxed);
                    eprintln!("[scheduler] ⚠️ rating 结果上报失败（本批丢弃）: task={task_id} {e:#}");
                }
            }
        });
    }

    /// 拉取任务；TASK_NONE（无任务/版本过旧）返回 None，由调用方退避轮询。
    /// 返回 Some 时主网络（rating 任务时含对手网络）已确保就绪于本地缓存。
    pub fn get_task(&mut self) -> Result<Option<SchedulerTask>> {
        let mut client = self.client();
        let req = TaskRequest {
            worker_id: self.cfg.worker_id.clone(),
            client_version: CLIENT_VERSION.to_string(),
            threads: num_cpus::get() as i32,
            memory_mb: available_memory_mb(),
            current_network: self.current_network.clone(),
        };
        let resp = self
            .rt
            .block_on(async move { client.get_task(req).await })
            .with_context(|| "GetTask 调用失败".to_string())?
            .into_inner();

        if resp.kind() == TaskKind::TaskNone {
            if !resp.message.is_empty() {
                println!("[scheduler] 无任务: {}", resp.message);
            }
            return Ok(None);
        }

        // 主网络：对象键恒下发（本地缓存命名 + 权重格式判定），URL 仅在需拉取时存在
        if resp.network_key.is_empty() {
            anyhow::bail!("服务端未下发 network_key（调度器版本过旧），无法确定权重格式");
        }
        self.ensure_downloaded(&resp.network_sha, &resp.network_key, &resp.network_url)?;

        let (mcts_sims, variant, initial_revealed) =
            resp.params.as_ref().map_or((0, String::new(), None), |p| {
                let mcts = p.mcts_sims.max(0) as usize;
                let variant = p.variant.trim().to_lowercase();
                let revealed = if p.extra_config.is_empty() {
                    None
                } else {
                    parse_extra_config(&p.extra_config)
                };
                (mcts, variant, revealed)
            });
        if variant.is_empty() {
            anyhow::bail!("服务端未下发变体（SelfPlayParams.variant 为空），请升级调度器并配置 SCHEDULER_VARIANT");
        }

        let task = SchedulerTask {
            task_id: resp.task_id.clone(),
            kind: resp.kind(),
            network_sha: resp.network_sha.clone(),
            opponent_sha: resp.opponent_sha.clone(),
            games: resp.games.max(0) as usize,
            mcts_sims,
            variant,
            initial_revealed,
        };

        // rating 任务：对手网络同样需就绪（各自的对象键按格式区分）
        if task.kind == TaskKind::TaskRating {
            if resp.opponent_key.is_empty() {
                anyhow::bail!("服务端未下发 opponent_key（调度器版本过旧），无法确定对手权重格式");
            }
            self.ensure_downloaded(&task.opponent_sha, &resp.opponent_key, &resp.opponent_url)?;
        }

        println!(
            "[scheduler] 任务 task={} kind={:?} variant={} network={} opponent={} games={} initial_revealed={:?}",
            task.task_id, task.kind, task.variant, task.network_sha, task.opponent_sha, task.games, task.initial_revealed
        );
        Ok(Some(task))
    }

    /// 下载（若本地缓存缺失）指定 sha 的网络文件，并更新 current_network。
    /// 本地路径由对象键决定（cache_dir/<key>），扩展名即权重格式。
    /// SRI 完整性校验：缓存命中与下载后都做 sha256 比对，不符则删除缓存并拒绝使用。
    fn ensure_downloaded(&mut self, sha: &str, key: &str, url: &str) -> Result<()> {
        let path = network_path(&self.cfg.cache_dir, key);
        if path.is_file() {
            if let Err(e) = verify_file_sha256(&path, sha) {
                println!(
                    "[scheduler] ⚠️ 缓存网络校验失败，删除并重新下载: {} ({e:#})",
                    path.display()
                );
                let _ = std::fs::remove_file(&path);
            }
        }
        if !path.is_file() {
            if url.is_empty() {
                anyhow::bail!("网络 {key} 本地无缓存且服务端未下发下载 URL");
            }
            self.rt
                .block_on(download(&self.http, url, &path, sha, &self.tmp_seq))
                .with_context(|| format!("下载网络失败: {key}"))?;
            println!("[scheduler] ✅ 网络已下载并校验: {key} -> {}", path.display());
        }
        self.current_network = sha.to_string();
        self.network_keys.insert(sha.to_string(), key.to_string());
        Ok(())
    }

    /// 返回指定 sha 的推理模型（含 `sessions` 条并发推理通道）；首次使用时加载。
    /// 该 sha 必须已经过 `ensure_downloaded`（对象键是定位本地文件的唯一依据）。
    pub fn model(&mut self, sha: &str) -> Result<Arc<OnnxModel>> {
        if let Some(m) = self.models.get(sha) {
            return Ok(Arc::clone(m));
        }
        let key = self
            .network_keys
            .get(sha)
            .with_context(|| format!("网络 {sha} 未登记对象键（未下载），无法定位本地权重"))?
            .clone();
        let model = self.load_model(sha, &key)?;
        self.models.insert(sha.to_string(), Arc::clone(&model));
        Ok(model)
    }

    /// 从本地缓存加载会话池（每 sha 一份，由 `model` 缓存）。
    /// 加载器按对象键扩展名分派：权重格式自描述，不靠调用方约定。
    fn load_model(&self, sha: &str, key: &str) -> Result<Arc<OnnxModel>> {
        let path = network_path(&self.cfg.cache_dir, key);
        match path.extension().and_then(|e| e.to_str()) {
            Some("onnx") => {}
            Some(other) => anyhow::bail!(
                "无法加载网络 {key}：本构建仅含 onnx 推理后端（实际权重格式 .{other}）"
            ),
            None => anyhow::bail!("网络对象键 {key} 缺少扩展名，无法判定权重格式"),
        }
        let sessions = self.cfg.sessions.max(1);
        let model = OnnxModel::with_sessions(&path.display().to_string(), &self.cfg.device, sessions)
            .map_err(|e| anyhow::anyhow!("加载 onnx 失败 ({key}): {e}"))?;
        println!(
            "[scheduler] 模型 {sha} 就绪：{} 条并发推理通道（device={}，权重 {key}）",
            model.session_count(),
            self.cfg.device
        );
        Ok(Arc::new(model))
    }

    /// 查询当前 best 网络（供复用/调试）：返回 (sha, 对象键, 预签名 URL)。
    pub fn get_best_network(&self) -> Result<Option<(String, String, String)>> {
        let mut client = self.client();
        let info = self
            .rt
            .block_on(async move { client.get_network(NetworkRequest { sha: String::new() }).await })
            .with_context(|| "GetNetwork 调用失败".to_string())?
            .into_inner();
        Ok(Some((info.sha, info.key, info.download_url)))
    }
}

// ============================================================================
// 异步上报
// ============================================================================

async fn report_episode_async(
    http: &reqwest::Client,
    client: &mut SchedulerServiceClient<Channel>,
    worker_id: &str,
    batch: EpisodeBatch,
) -> Result<(usize, usize, String)> {
    let EpisodeBatch { task_id, network_sha, variant, winner, episodes, nnue_episodes } = batch;
    let game_count = episodes.len() + nnue_episodes.len();
    let total_steps: usize = episodes.iter().map(|e| e.game_length).sum::<usize>()
        + nnue_episodes.iter().map(|e| e.game_length).sum::<usize>();

    // 二进制编码 + gzip 是 CPU 密集段，放到 blocking 线程池，避免占住 tokio worker。
    let gz_body = tokio::task::spawn_blocking(move || {
        batch_gz(&variant, &episodes, &nnue_episodes)
    })
    .await
    .context("episode 编码任务异常退出")??;

    let meta = EpisodeMeta {
        worker_id: worker_id.to_string(),
        task_id,
        game_count: game_count as i32,
        total_steps: total_steps as i32,
        winner,
        network_sha,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        content_length: gz_body.len() as i64,
        content_sha256: hex_sha256(&gz_body),
    };

    let ack = client
        .report_episode(meta)
        .await
        .context("ReportEpisode 调用失败")?
        .into_inner();
    if !ack.accepted {
        anyhow::bail!("episode 被拒绝: {}", ack.message);
    }

    let object_key = ack.object_key.clone();
    upload(http, &ack.upload_url, gz_body)
        .await
        .with_context(|| format!("直传 R2 失败: {object_key}"))?;
    Ok((game_count, total_steps, ack.object_key))
}

async fn report_match_async(
    client: &mut SchedulerServiceClient<Channel>,
    worker_id: &str,
    rep: MatchReport,
) -> Result<(bool, bool, String)> {
    let ack = client
        .report_match_result(PbMatchResult {
            worker_id: worker_id.to_string(),
            task_id: rep.task_id,
            kind: TaskKind::TaskRating as i32,
            network_sha: rep.network_sha,
            opponent_sha: rep.opponent_sha,
            games: rep.games as i32,
            wins: rep.wins as i32,
            losses: rep.losses as i32,
            draws: rep.draws as i32,
            pair_ll: rep.pairs[0] as i32,
            pair_ld: rep.pairs[1] as i32,
            pair_dd: rep.pairs[2] as i32,
            pair_dw: rep.pairs[3] as i32,
            pair_ww: rep.pairs[4] as i32,
        })
        .await
        .context("ReportMatchResult 调用失败")?
        .into_inner();
    if !ack.accepted {
        anyhow::bail!("match result 被拒绝: {}", ack.message);
    }
    Ok((ack.match_concluded, ack.promoted, ack.best_sha))
}

async fn upload(http: &reqwest::Client, url: &str, body: Vec<u8>) -> Result<()> {
    if url.is_empty() {
        anyhow::bail!("预签名 PUT URL 为空");
    }
    http.put(url)
        .body(body)
        .send()
        .await
        .with_context(|| format!("HTTP PUT 请求失败: {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP PUT 返回错误状态: {url}"))?;
    Ok(())
}

// ============================================================================
// 网络缓存
// ============================================================================

/// 本地缓存路径 = cache_dir/<对象键>（与 Go 侧 r2.NetworkKey 一一对应，
/// 形如 networks/<sha>.onnx；扩展名即权重格式，供加载器分派）。
fn network_path(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(key)
}

/// 预签名 GET → 临时文件 → 原子替换 → sha256 校验。
/// 临时文件名带 pid/序号，避免与后台预取的同名写入互相踩踏。
async fn download(
    http: &reqwest::Client,
    url: &str,
    to: &Path,
    expect_sha: &str,
    tmp_seq: &AtomicU64,
) -> Result<()> {
    let bytes = http
        .get(url)
        .send()
        .await
        .with_context(|| format!("HTTP GET 请求失败: {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP GET 返回错误状态: {url}"))?
        .bytes()
        .await
        .with_context(|| format!("读取下载内容失败: {url}"))?;

    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("创建目录失败: {}", parent.display()))?;
    }
    let seq = tmp_seq.fetch_add(1, Ordering::Relaxed);
    let tmp = to.with_extension(format!("{}.{seq}.tmp", std::process::id()));
    std::fs::write(&tmp, &bytes).with_context(|| format!("写临时文件失败: {}", tmp.display()))?;
    std::fs::rename(&tmp, to).with_context(|| format!("原子替换失败: {}", to.display()))?;

    if let Err(e) = verify_file_sha256(to, expect_sha) {
        let _ = std::fs::remove_file(to);
        return Err(e);
    }
    Ok(())
}

/// SRI：文件内容 sha256 是否等于期望的 hex sha（大小写不敏感）。
fn verify_file_sha256(path: &Path, expect_sha: &str) -> Result<()> {
    let data = std::fs::read(path).with_context(|| format!("读取网络文件失败: {}", path.display()))?;
    let actual = hex_sha256(&data);
    if !actual.eq_ignore_ascii_case(expect_sha) {
        anyhow::bail!("sha256 不匹配: 期望 {expect_sha} 实际 {actual}");
    }
    Ok(())
}

// ============================================================================
// 后台任务
// ============================================================================

/// 后台心跳：周期上报版本声明/资源/进度，感知 best 网络变化与暂停指令。
/// best 变化时顺带在后台预取该网络，把下载从主循环里挪走。
fn spawn_heartbeat(
    handle: tokio::runtime::Handle,
    channel: Channel,
    cfg: SchedulerConfig,
    http: reqwest::Client,
    tmp_seq: Arc<AtomicU64>,
    completed_games: Arc<AtomicU64>,
    running_task_id: Arc<Mutex<String>>,
) {
    handle.spawn(async move {
        let mut last_best = String::new();
        loop {
            tokio::time::sleep(HEARTBEAT_INTERVAL).await;
            let req = HeartbeatRequest {
                worker_id: cfg.worker_id.clone(),
                current_threads: num_cpus::get() as i32,
                completed_games: completed_games.load(Ordering::Relaxed) as i32,
                running_task_id: running_task_id.lock().unwrap().clone(),
                client_version: CLIENT_VERSION.to_string(),
                memory_mb: available_memory_mb(),
            };
            let mut client = SchedulerServiceClient::new(channel.clone());
            match client.heartbeat(req).await.map(|r| r.into_inner()) {
                Ok(reply) => {
                    if reply.pause_self_play {
                        println!("[heartbeat] ⏸️ 服务端下发 pause_self_play");
                    }
                    if !reply.best_network.is_empty() && reply.best_network != last_best {
                        if !last_best.is_empty() {
                            println!(
                                "[heartbeat] 🔄 best 网络变化: {last_best} -> {}（后台预取）",
                                reply.best_network
                            );
                        }
                        last_best = reply.best_network.clone();
                        prefetch_network(
                            &http,
                            &cfg,
                            channel.clone(),
                            Arc::clone(&tmp_seq),
                            reply.best_network,
                        );
                    }
                }
                Err(status) => {
                    println!("[heartbeat] 上报失败（将重试）: {status}");
                }
            }
        }
    });
}

/// 后台预取 best 网络：命中本地缓存直接跳过，否则 GetNetwork 取对象键与 URL 后
/// 下载并校验。主路径 `ensure_downloaded` 命中后只做 sha256 校验，换网不再阻塞采集。
fn prefetch_network(
    http: &reqwest::Client,
    cfg: &SchedulerConfig,
    channel: Channel,
    tmp_seq: Arc<AtomicU64>,
    sha: String,
) {
    let http = http.clone();
    let cfg = cfg.clone();
    tokio::spawn(async move {
        let mut client = SchedulerServiceClient::new(channel);
        let info = match client
            .get_network(NetworkRequest { sha: String::new() })
            .await
            .map(|r| r.into_inner())
        {
            Ok(info) => info,
            Err(e) => {
                println!("[prefetch] ⚠️ 获取 best 网络信息失败: {e}");
                return;
            }
        };
        if info.sha != sha || info.download_url.is_empty() || info.key.is_empty() {
            return;
        }
        let path = network_path(&cfg.cache_dir, &info.key);
        if path.is_file() {
            return;
        }
        match download(&http, &info.download_url, &path, &sha, &tmp_seq).await {
            Ok(()) => println!("[prefetch] ✅ 已预取 best 网络: {}", info.key),
            Err(e) => println!(
                "[prefetch] ⚠️ 预取 best 网络失败（主路径将重试）: {} {e:#}",
                info.key
            ),
        }
    });
}

// ============================================================================
// 工具
// ============================================================================

/// 一批 episode 编码为 gzip 压缩的 EpisodeBatch 二进制载荷。
///
/// 编码失败（形状不一致 / 非 0/1 特征等契约破裂）时整批作废并向上报错：
/// 训练数据宁可丢一批也不能静默进入训练。
fn batch_gz(variant: &str, episodes: &[GameEpisode], nnue_episodes: &[NnueEpisode]) -> Result<Vec<u8>> {
    use std::io::Write;
    let raw = encode_episode_batch(variant, episodes, nnue_episodes)?;
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(&raw).context("gzip 写入失败")?;
    gz.finish().context("gzip 收尾失败")
}

fn hex_sha256(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// 解析 SelfPlayParams.extra_config（JSON 透传）中的课程参数。
/// 仅提取 `initial_revealed_pieces`；0 或解析失败返回 None（用变体默认值）。
fn parse_extra_config(json: &str) -> Option<usize> {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            println!("[scheduler] ⚠️ extra_config 解析失败（忽略课程参数）: {json} ({e})");
            return None;
        }
    };
    let n = v.get("initial_revealed_pieces")?.as_u64()? as usize;
    if n == 0 {
        return None;
    }
    Some(n)
}

/// 可用内存（MB）：Linux 读 /proc/meminfo MemAvailable，失败返回 0。
fn available_memory_mb() -> i64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: i64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
            return kb / 1024;
        }
    }
    0
}
