// src/registry/scheduler_registry.rs — SchedulerRegistry（分布式 Registry 实现）
//
// 自 rust_4x8/src/registry/scheduler_registry.rs 迁入（banqi-collector 拆分），
// ONNX 模型类型由 crate::inference::onnx 切换为 banqi-engine 的实现。
//
// 语义：get_task（= GetTask，无任务返回 None 由调用方退避轮询）、
// model（= ensure_model，按 sha 惰性下载 + 加载 onnx，进程内缓存）、
// report_episode（ReportEpisode 签发预签名 PUT → HTTP 直传 R2）、
// report_match_result（rating 五项计数上报，服务端 GSPRT 判停晋级）。
// 模型换网感知：GetTask 返回的 best sha 变化即触发新模型下载加载。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tonic::transport::Channel;

use banqi_engine::inference::onnx::OnnxModel;

pub mod pb {
    tonic::include_proto!("scheduler");
}

use pb::scheduler_service_client::SchedulerServiceClient;
use pb::{EpisodeMeta, HeartbeatRequest, MatchResult as PbMatchResult, NetworkRequest, TaskKind, TaskRequest};

/// 心跳间隔
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct SchedulerConfig {
    pub endpoint: String,
    pub worker_id: String,
    pub client_version: String,
    pub cache_dir: PathBuf,
    pub device: String,
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
}

pub struct SchedulerRegistry {
    rt: tokio::runtime::Runtime,
    http: reqwest::Client,
    cfg: SchedulerConfig,
    /// 本地已加载的模型缓存（sha -> 模型）
    models: HashMap<String, Arc<OnnxModel>>,
    /// 当前持有的网络 sha（GetTask 时上报，服务端据此免发下载 URL）
    current_network: String,
    /// 心跳共享状态：累计完成局数
    completed_games: Arc<AtomicU64>,
    /// 心跳共享状态：当前执行中的 task_id
    running_task_id: Arc<Mutex<String>>,
}

impl SchedulerRegistry {
    pub fn new(cfg: SchedulerConfig) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .context("构建 tokio runtime 失败")?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("构建 HTTP 客户端失败")?;
        let completed_games = Arc::new(AtomicU64::new(0));
        let running_task_id = Arc::new(Mutex::new(String::new()));
        spawn_heartbeat(
            rt.handle().clone(),
            cfg.clone(),
            Arc::clone(&completed_games),
            Arc::clone(&running_task_id),
        );
        Ok(Self {
            rt,
            http,
            cfg,
            models: HashMap::new(),
            current_network: String::new(),
            completed_games,
            running_task_id,
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

    fn connect(&self) -> Result<SchedulerServiceClient<Channel>> {
        let client = self
            .rt
            .block_on(SchedulerServiceClient::connect(self.cfg.endpoint.clone()))
            .with_context(|| format!("连接调度器失败: {}", self.cfg.endpoint))?;
        Ok(client)
    }

    /// 拉取任务；TASK_NONE（无任务/版本过旧）返回 None，由调用方退避轮询。
    /// 返回 Some 时主网络（rating 任务时含对手网络）已确保就绪于本地缓存。
    pub fn get_task(&mut self) -> Result<Option<SchedulerTask>> {
        let mut client = self.connect()?;
        let req = TaskRequest {
            worker_id: self.cfg.worker_id.clone(),
            client_version: self.cfg.client_version.clone(),
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

        // 主网络：sha 变化或本地无缓存时经预签名 URL 下载
        self.ensure_downloaded(&resp.network_sha, &resp.network_url)?;

        let (mcts_sims, variant) = resp.params.as_ref().map_or((0, String::new()), |p| {
            (
                p.mcts_sims.max(0) as usize,
                p.variant.trim().to_lowercase(),
            )
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
        };

        // rating 任务：对手网络同样需就绪
        if task.kind == TaskKind::TaskRating {
            self.ensure_downloaded(&task.opponent_sha, &resp.opponent_url)?;
        }

        println!(
            "[scheduler] 任务 task={} kind={:?} variant={} network={} opponent={} games={}",
            task.task_id, task.kind, task.variant, task.network_sha, task.opponent_sha, task.games
        );
        Ok(Some(task))
    }

    /// 下载（若本地缓存缺失）指定 sha 的网络文件，并更新 current_network。
    /// SRI 完整性校验：缓存命中与下载后都做 sha256 比对，不符则删除缓存并拒绝使用。
    fn ensure_downloaded(&mut self, sha: &str, url: &str) -> Result<()> {
        let path = self.network_path(sha);
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
                anyhow::bail!("网络 {sha} 本地无缓存且服务端未下发下载 URL");
            }
            self.download(url, &path)
                .with_context(|| format!("下载网络失败: {sha}"))?;
            if let Err(e) = verify_file_sha256(&path, sha) {
                let _ = std::fs::remove_file(&path);
                return Err(anyhow::anyhow!("下载的网络 SRI 校验失败 (sha={sha}): {e:#}"));
            }
            println!("[scheduler] ✅ 网络已下载并校验: {} -> {}", sha, path.display());
        }
        self.current_network = sha.to_string();
        Ok(())
    }

    fn network_path(&self, sha: &str) -> PathBuf {
        // 与 Go 侧 r2.NetworkKey 对应的本地缓存布局: cache_dir/networks/<sha>.bin
        self.cfg.cache_dir.join("networks").join(format!("{sha}.bin"))
    }

    fn download(&self, url: &str, to: &PathBuf) -> Result<()> {
        let bytes = self
            .rt
            .block_on(async {
                self.http.get(url).send().await?.error_for_status()?.bytes().await
            })
            .with_context(|| format!("HTTP 下载失败: {url}"))?;
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("创建目录失败: {}", parent.display()))?;
        }
        let tmp = to.with_extension("tmp");
        std::fs::write(&tmp, &bytes).with_context(|| format!("写临时文件失败: {}", tmp.display()))?;
        std::fs::rename(&tmp, to).with_context(|| format!("原子替换失败: {}", to.display()))?;
        Ok(())
    }

    /// 返回指定 sha 的推理模型；首次使用时从缓存文件加载 onnx。
    pub fn model(&mut self, sha: &str) -> Result<Arc<OnnxModel>> {
        if let Some(m) = self.models.get(sha) {
            return Ok(Arc::clone(m));
        }
        let path = self.network_path(sha);
        let model = Arc::new(OnnxModel::new(
            &path.display().to_string(),
            &self.cfg.device,
        )
        .map_err(|e| anyhow::anyhow!("加载 onnx 失败 ({sha}): {e}"))?);
        self.models.insert(sha.to_string(), Arc::clone(&model));
        Ok(model)
    }

    /// 上报一批 episode：元数据 → 预签名 PUT，数据直传 R2。
    pub fn report_episode(
        &self,
        task_id: &str,
        network_sha: &str,
        game_count: usize,
        total_steps: usize,
        winner: i32,
        gz_body: Vec<u8>,
    ) -> Result<()> {
        let content_sha256 = hex_sha256(&gz_body);
        let content_length = gz_body.len() as i64;
        let worker_id = self.cfg.worker_id.clone();
        let mut client = self.connect()?;
        let ack = self
            .rt
            .block_on(async move {
                let meta = EpisodeMeta {
                    worker_id,
                    task_id: task_id.to_string(),
                    game_count: game_count as i32,
                    total_steps: total_steps as i32,
                    winner,
                    network_sha: network_sha.to_string(),
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64,
                    content_length,
                    content_sha256,
                };
                client.report_episode(meta).await
            })
            .with_context(|| "ReportEpisode 调用失败".to_string())?
            .into_inner();
        if !ack.accepted {
            anyhow::bail!("episode 被拒绝: {}", ack.message);
        }
        self.upload(&ack.upload_url, gz_body)
            .with_context(|| format!("直传 R2 失败: {}", ack.object_key))?;
        println!(
            "[scheduler] ✅ episode 已直传: games={game_count} steps={total_steps} -> {}",
            ack.object_key
        );
        Ok(())
    }

    /// 上报 rating 结果（五项成对计数），返回服务端判停结论。
    pub fn report_match_result(
        &self,
        task_id: &str,
        network_sha: &str,
        opponent_sha: &str,
        games: usize,
        wins: usize,
        losses: usize,
        draws: usize,
        pairs: [usize; 5],
    ) -> Result<(bool, bool, String)> {
        let mut client = self.connect()?;
        let ack = self
            .rt
            .block_on(async move {
                let req = PbMatchResult {
                    worker_id: self.cfg.worker_id.clone(),
                    task_id: task_id.to_string(),
                    kind: TaskKind::TaskRating as i32,
                    network_sha: network_sha.to_string(),
                    opponent_sha: opponent_sha.to_string(),
                    games: games as i32,
                    wins: wins as i32,
                    losses: losses as i32,
                    draws: draws as i32,
                    pair_ll: pairs[0] as i32,
                    pair_ld: pairs[1] as i32,
                    pair_dd: pairs[2] as i32,
                    pair_dw: pairs[3] as i32,
                    pair_ww: pairs[4] as i32,
                };
                client.report_match_result(req).await
            })
            .with_context(|| "ReportMatchResult 调用失败".to_string())?
            .into_inner();
        if !ack.accepted {
            anyhow::bail!("match result 被拒绝: {}", ack.message);
        }
        println!(
            "[scheduler] match result: concluded={} promoted={} best={}",
            ack.match_concluded, ack.promoted, ack.best_sha
        );
        Ok((ack.match_concluded, ack.promoted, ack.best_sha))
    }

    /// trainer 侧注册新网络（本进程通常不调用；供复用/调试）。
    pub fn get_best_network(&self) -> Result<Option<(String, String)>> {
        let mut client = self.connect()?;
        let info = self
            .rt
            .block_on(async move { client.get_network(NetworkRequest { sha: String::new() }).await })
            .with_context(|| "GetNetwork 调用失败".to_string())?
            .into_inner();
        Ok(Some((info.sha, info.download_url)))
    }

    fn upload(&self, url: &str, body: Vec<u8>) -> Result<()> {
        if url.is_empty() {
            anyhow::bail!("预签名 PUT URL 为空");
        }
        self.rt
            .block_on(async {
                self.http.put(url).body(body).send().await?.error_for_status()?;
                Ok::<_, reqwest::Error>(())
            })
            .with_context(|| format!("HTTP PUT 失败: {url}"))?;
        Ok(())
    }
}

fn hex_sha256(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
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

/// 后台心跳：周期上报版本声明/资源/进度，日志感知 best 网络变化与暂停指令。
fn spawn_heartbeat(
    handle: tokio::runtime::Handle,
    cfg: SchedulerConfig,
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
                client_version: cfg.client_version.clone(),
                memory_mb: available_memory_mb(),
            };
            let result = match SchedulerServiceClient::connect(cfg.endpoint.clone()).await {
                Ok(mut client) => client.heartbeat(req).await.map(|r| r.into_inner()),
                Err(e) => Err(tonic::Status::unknown(format!("connect: {e}"))),
            };
            match result {
                Ok(reply) => {
                    if !last_best.is_empty() && reply.best_network != last_best {
                        println!(
                            "[heartbeat] 🔄 best 网络变化: {last_best} -> {}（下次 GetTask 生效）",
                            reply.best_network
                        );
                    }
                    if reply.pause_self_play {
                        println!("[heartbeat] ⏸️ 服务端下发 pause_self_play");
                    }
                    last_best = reply.best_network;
                }
                Err(status) => {
                    println!("[heartbeat] 上报失败（将重试）: {status}");
                }
            }
        }
    });
}
