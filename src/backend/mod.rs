//! Switchable worker backends.
//!
//! After the HTTP router picks a worker URL:
//! - `http(s)://` reverse-proxies OpenAI JSON from `routers/http/router.rs`
//!   (same HTTP client as completions/embeddings; no GenerateRequest to build).
//! - `grpc(s)://` uses `preprocess` + `convert` + `grpc::dispatch_chat`.
//!
//! gRPC wire types come from crates.io `vllm-proto`. Chat+tokenize is
//! `vllm-chat` + `vllm-tokenizer` only (Cargo git, not `pip install`).
//! HTTP does not tokenize on the critical path.
//!
//! `VLLM_ROUTER_STAGES=1` emits stage clocks. HTTP shadow tokenize runs only
//! then; gRPC tokenize always runs.

pub mod convert;
pub mod detect;
pub mod grpc;
pub mod health;
pub mod openai;
pub mod preprocess;
mod vllm_frontend;

/// Opt-in router stage clocks / HTTP shadow tokenize.
pub fn stages_enabled() -> bool {
    matches!(
        std::env::var("VLLM_ROUTER_STAGES").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

pub use detect::{
    connection_mode_from_url, grpc_connect_uri, is_grpc_url, parse_dp_rank, strip_dp_suffix,
};
pub use grpc::GrpcEngineBackend;
pub use health::check_grpc_health;
pub use preprocess::TokenizerCache;

/// Prost/Tonic bindings for vLLM Inference + Control (`vllm-proto` on crates.io).
pub use vllm_proto as pb;
