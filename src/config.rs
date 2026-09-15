// src/config.rs — 分层配置：默认值 → TOML 文件 → CLI 覆盖。
//
// 参考 xmrig 的配置分层（内嵌默认 → config.json → CLI 合并进同一份文档 → 单一 read 路径，
// 见 xmrig/src/core/config/Config.cpp 与 base/kernel/config/ConfigTransform.cpp），
// 这里用 serde + toml 复刻同样语义：`CollectorConfig::load` 是唯一入口，
// `--config-dump` 可回写生效配置以便复现。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::pipeline::self_play::SelfPlayConfig;
use crate::registry::SchedulerConfig;

/// 采集进程完整配置。并发相关旋钮各自归属消费方：自对弈线程数在 `[selfplay]`，
/// 在途上报批数在 `[scheduler]`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CollectorConfig {
    pub scheduler: SchedulerConfig,
    /// 自对弈参数；服务端下发的 mcts_sims / initial_revealed 会在运行时覆盖对应字段。
    pub selfplay: SelfPlayConfig,
}

/// CLI 覆盖项：`None` 表示该参数未在命令行给出，保留配置文件或默认值。
#[derive(Debug, Default, Args)]
pub struct CliOverrides {
    /// 配置文件路径（TOML）；不传则仅用默认值 + CLI 覆盖
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// 把生效配置写入该 TOML 路径（启动时）
    #[arg(long, value_name = "PATH")]
    pub config_dump: Option<PathBuf>,
    /// 调度器 gRPC 地址（http://host:port）
    #[arg(long)]
    pub scheduler_endpoint: Option<String>,
    /// worker 标识（默认 worker-<pid>）
    #[arg(long)]
    pub worker_id: Option<String>,
    /// 网络/模型本地缓存目录
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// 推理设备：cpu / auto
    #[arg(long)]
    pub device: Option<String>,
    /// ONNX 会话数（并发推理通道数；0 = 自动 = 自对弈线程数）
    #[arg(long)]
    pub sessions: Option<usize>,
    /// 每次决策的 MCTS 模拟次数
    #[arg(long)]
    pub mcts_sims: Option<usize>,
    /// Gumbel Top-K 候选动作数
    #[arg(long)]
    pub max_considered_actions: Option<usize>,
    /// 自对弈线程数（0 = CPU 核数）
    #[arg(long)]
    pub threads: Option<usize>,
}

impl CollectorConfig {
    /// 唯一配置入口：默认值 → 可选 TOML → CLI 覆盖 → 校验。
    pub fn load(cli: &CliOverrides) -> Result<Self> {
        let mut cfg = match &cli.config {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("读取配置文件失败: {}", path.display()))?;
                toml::from_str::<Self>(&text)
                    .with_context(|| format!("解析配置文件失败: {}", path.display()))?
            }
            None => Self::default(),
        };

        if let Some(v) = &cli.scheduler_endpoint {
            cfg.scheduler.endpoint = v.clone();
        }
        if let Some(v) = &cli.worker_id {
            cfg.scheduler.worker_id = v.clone();
        }
        if let Some(v) = &cli.cache_dir {
            cfg.scheduler.cache_dir = v.clone();
        }
        if let Some(v) = &cli.device {
            cfg.scheduler.device = v.clone();
        }
        if let Some(v) = &cli.sessions {
            cfg.scheduler.sessions = *v;
        }
        if let Some(v) = &cli.mcts_sims {
            cfg.selfplay.mcts_sims = *v;
            // 归零 = 由新的 mcts_sims 重新推导（见 SelfPlayConfig::fast_sims）
            cfg.selfplay.fast_mcts_sims = 0;
        }
        if let Some(v) = &cli.max_considered_actions {
            cfg.selfplay.max_considered_actions = *v;
        }
        if let Some(v) = &cli.threads {
            cfg.selfplay.threads = *v;
        }

        if cfg.scheduler.worker_id.is_empty() {
            cfg.scheduler.worker_id = format!("worker-{}", std::process::id());
        }

        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.scheduler.max_inflight_reports == 0 {
            anyhow::bail!("scheduler.max_inflight_reports 必须 >= 1");
        }
        let sp = &self.selfplay;
        if sp.mcts_sims == 0 {
            anyhow::bail!("selfplay.mcts_sims 必须 >= 1");
        }
        if sp.max_considered_actions == 0 {
            anyhow::bail!("selfplay.max_considered_actions 必须 >= 1");
        }
        if sp.playout_cap_random_enabled && sp.fast_mcts_sims >= sp.mcts_sims {
            anyhow::bail!(
                "selfplay.fast_mcts_sims ({}) 必须小于 mcts_sims ({})，当前开启算力随机化",
                sp.fast_mcts_sims,
                sp.mcts_sims
            );
        }
        Ok(())
    }

    /// 生效配置的 TOML 文本（启动打印 + `--config-dump` 共用）。
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("序列化生效配置失败")
    }

    /// 启动时把生效配置落盘，便于事后复现。
    pub fn dump_to(&self, path: &Path) -> Result<()> {
        let text = self.to_toml()?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("创建目录失败: {}", parent.display()))?;
            }
        }
        std::fs::write(path, text).with_context(|| format!("写入配置快照失败: {}", path.display()))
    }
}
