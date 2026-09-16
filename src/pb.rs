// src/pb.rs — scheduler.proto 生成代码（gRPC 客户端 + 训练数据记录消息）
//
// 唯一契约源为 banqi-scheduler/proto/scheduler.proto，本仓库的 proto/ 是同步副本
// （见 README：proto 变更先改 scheduler 仓库再复制过来）。
// 生成方式见 build.rs（tonic-build，仅客户端）。

#![allow(clippy::all)]

tonic::include_proto!("scheduler");
