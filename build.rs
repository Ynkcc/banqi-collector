fn main() {
    // gRPC 代码生成（scheduler.proto：分布式调度器客户端，与 banqi-scheduler 仓库 proto 副本同步维护）
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&["proto/scheduler.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));
    println!("cargo:rerun-if-changed=proto/scheduler.proto");
}
