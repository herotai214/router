//! Chat+tokenize for gRPC workers: **only** `vllm-chat` + `vllm-tokenizer`.
//!
//! There is no HuggingFace/minijinja fallback. If those crates are not
//! linked, `grpc://` is unavailable.
//!
//! `TokenizerCache` caches loaded frontend objects (`load_model_backends`),
//! not prior-request token ids or engine KV.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;

use super::vllm_frontend::{
    assistant_text, chat_request_from_openai, system_text, tool_text, user_text, VllmFrontend,
};
use crate::protocols::spec::{
    ChatCompletionRequest, ChatMessage as SpecChatMessage, UserMessageContent,
};

#[derive(Clone)]
enum Frontend {
    Vllm(Arc<VllmFrontend>),
    /// Integration-test pin: fake ids, still not this repo’s tokenizer.
    TestIds(Vec<u32>),
}

/// In-process cache of loaded `vllm-chat` / `vllm-tokenizer` objects.
///
/// First request for a model key calls `load_model_backends`; later
/// requests clone the `Arc`. This is **not** reuse of prior-request
/// `token_ids` and **not** engine KV / prefix-cache routing.
#[derive(Clone)]
pub struct TokenizerCache {
    pinned: Arc<RwLock<Option<Frontend>>>,
    by_model: Arc<DashMap<String, Arc<VllmFrontend>>>,
}

impl std::fmt::Debug for TokenizerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerCache").finish()
    }
}

impl Default for TokenizerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerCache {
    pub fn new() -> Self {
        Self {
            pinned: Arc::new(RwLock::new(None)),
            by_model: Arc::new(DashMap::new()),
        }
    }

    pub fn from_env() -> Self {
        Self::new()
    }

    /// Tests only. Does not load this crate’s HuggingFace stack.
    pub fn pin_test_token_ids(&self, token_ids: Vec<u32>) {
        *self.pinned.write() = Some(Frontend::TestIds(token_ids));
    }

    pub async fn resolve(&self, model: Option<&str>) -> Result<FrontendHandle> {
        if let Some(pinned) = self.pinned.read().clone() {
            return Ok(FrontendHandle(pinned));
        }
        let key = model_key(model)?;
        if let Some(hit) = self.by_model.get(&key) {
            return Ok(FrontendHandle(Frontend::Vllm(hit.clone())));
        }
        let frontend = Arc::new(
            VllmFrontend::load(&key, Default::default())
                .await
                .map_err(|e| anyhow!("vllm-chat load {key}: {e}"))?,
        );
        if self.pinned.read().is_none() {
            *self.pinned.write() = Some(Frontend::Vllm(frontend.clone()));
        }
        self.by_model.insert(key, frontend.clone());
        Ok(FrontendHandle(Frontend::Vllm(frontend)))
    }
}

fn model_key(request_model: Option<&str>) -> Result<String> {
    if let Ok(path) = std::env::var("VLLM_ROUTER_MODEL") {
        if !path.is_empty() {
            return Ok(path);
        }
    }
    if let Some(model) = request_model.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(model.to_string());
    }
    if let Ok(tok) = std::env::var("VLLM_ROUTER_TOKENIZER") {
        let path = Path::new(&tok);
        if path.is_file() || path.is_dir() {
            return Ok(tok);
        }
    }
    Err(anyhow!(
        "grpc worker requires a model dir: set request.model or VLLM_ROUTER_MODEL"
    ))
}

#[derive(Clone)]
pub struct FrontendHandle(Frontend);

impl FrontendHandle {
    pub fn tokenizer(&self) -> Option<super::vllm_frontend::DynTokenizer> {
        match &self.0 {
            Frontend::Vllm(frontend) => Some(frontend.tokenizer()),
            Frontend::TestIds(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizeOut {
    pub token_ids: Vec<u32>,
    pub adapt_ms: f64,
    pub template_ms: f64,
    pub encode_ms: f64,
    pub encode_backend: &'static str,
}

impl TokenizeOut {
    pub fn frontend_ms(&self) -> f64 {
        self.adapt_ms + self.template_ms + self.encode_ms
    }
}

pub fn tokenize_chat_request_timed(
    request: &ChatCompletionRequest,
    frontend: &FrontendHandle,
) -> Result<TokenizeOut> {
    match &frontend.0 {
        Frontend::TestIds(ids) => {
            if ids.is_empty() {
                return Err(anyhow!("pinned test token_ids are empty"));
            }
            Ok(TokenizeOut {
                token_ids: ids.clone(),
                adapt_ms: 0.0,
                template_ms: 0.0,
                encode_ms: 0.0,
                encode_backend: "test-pin",
            })
        }
        Frontend::Vllm(vllm) => tokenize_vllm(request, vllm),
    }
}

fn tokenize_vllm(request: &ChatCompletionRequest, frontend: &VllmFrontend) -> Result<TokenizeOut> {
    let t0 = Instant::now();
    let kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
    let upstream = chat_request_from_openai(
        format!("router-{}", request.model.as_deref().unwrap_or("model")),
        spec_messages_to_upstream(&request.messages),
        request.add_generation_prompt,
        request.continue_final_message,
        kwargs,
        false,
    );
    let adapt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (token_ids, template_ms, encode_ms) = frontend
        .chat_tokenize_timed(&upstream)
        .map_err(|e| anyhow!("vllm-chat tokenize: {e}"))?;
    if token_ids.is_empty() {
        return Err(anyhow!("vllm-chat produced empty token_ids"));
    }
    Ok(TokenizeOut {
        token_ids,
        adapt_ms,
        template_ms,
        encode_ms,
        encode_backend: "vllm-tokenizer",
    })
}

fn spec_messages_to_upstream(
    messages: &[SpecChatMessage],
) -> Vec<super::vllm_frontend::UpstreamChatMessage> {
    messages
        .iter()
        .map(|message| match message {
            SpecChatMessage::System { content, .. } => system_text(content),
            SpecChatMessage::User { content, .. } => user_text(user_content_text(content)),
            SpecChatMessage::Assistant { content, .. } => {
                assistant_text(content.clone().unwrap_or_default())
            }
            SpecChatMessage::Tool {
                content,
                tool_call_id,
                ..
            } => {
                let text = match content {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                tool_text(text, tool_call_id)
            }
            SpecChatMessage::Function { content, .. } => user_text(content),
        })
        .collect()
}

fn user_content_text(content: &UserMessageContent) -> String {
    match content {
        UserMessageContent::Text(text) => text.clone(),
        UserMessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                crate::protocols::spec::ContentPart::Text { text } => text.as_str(),
                crate::protocols::spec::ContentPart::ImageUrl { .. } => "<image>",
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn model_dir() -> Option<String> {
        let dir = std::env::var("VLLM_ROUTER_MODEL")
            .ok()
            .filter(|s| !s.is_empty())?;
        Path::new(&dir)
            .join("tokenizer.json")
            .is_file()
            .then_some(dir)
    }

    fn python_bin() -> PathBuf {
        if let Ok(py) = std::env::var("PYTHON") {
            if !py.is_empty() {
                return PathBuf::from(py);
            }
        }
        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            let cand = PathBuf::from(venv).join("bin/python");
            if cand.is_file() {
                return cand;
            }
        }
        PathBuf::from("python3")
    }

    fn python_vllm_chat_ids(
        model: &str,
        messages: &serde_json::Value,
    ) -> Result<Option<Vec<u32>>, String> {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python_vllm_chat_ids.py");
        let out = Command::new(python_bin())
            .arg(&script)
            .arg(model)
            .arg(messages.to_string())
            .output()
            .map_err(|e| format!("spawn python: {e}"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("ModuleNotFoundError") || err.contains("No module named") {
                return Ok(None);
            }
            return Err(format!(
                "python vllm tokenize failed ({:?}): {err}",
                out.status
            ));
        }
        serde_json::from_slice(&out.stdout).map(Some).map_err(|e| {
            format!(
                "python stdout not a json id list: {e}; stdout={}",
                String::from_utf8_lossy(&out.stdout)
            )
        })
    }

    async fn rust_vllm_chat_ids(model: &str, messages: serde_json::Value) -> Vec<u32> {
        let frontend = VllmFrontend::load(model, Default::default())
            .await
            .expect("vllm-chat load_model_backends");
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 8,
            "add_generation_prompt": true
        }))
        .unwrap();
        tokenize_vllm(&req, &frontend)
            .expect("vllm-chat tokenize")
            .token_ids
    }

    #[tokio::test]
    async fn vllm_chat_tokenize_matches_python_vllm_user() {
        let Some(model) = model_dir() else {
            eprintln!("skip: set VLLM_ROUTER_MODEL to a model dir with tokenizer.json");
            return;
        };
        let messages = serde_json::json!([{"role": "user", "content": "hello"}]);
        let Some(py_ids) = python_vllm_chat_ids(&model, &messages).expect("python vllm") else {
            eprintln!("skip: python cannot import vllm (set PYTHON or VIRTUAL_ENV)");
            return;
        };
        let rust_ids = rust_vllm_chat_ids(&model, messages).await;
        assert_eq!(
            rust_ids, py_ids,
            "vllm-chat+vllm-tokenizer must match Python vllm.tokenizers.get_tokenizer"
        );
    }

    #[tokio::test]
    async fn vllm_chat_tokenize_matches_python_vllm_system_user() {
        let Some(model) = model_dir() else {
            eprintln!("skip: set VLLM_ROUTER_MODEL to a model dir with tokenizer.json");
            return;
        };
        let messages = serde_json::json!([
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "ping"}
        ]);
        let Some(py_ids) = python_vllm_chat_ids(&model, &messages).expect("python vllm") else {
            eprintln!("skip: python cannot import vllm (set PYTHON or VIRTUAL_ENV)");
            return;
        };
        let rust_ids = rust_vllm_chat_ids(&model, messages).await;
        assert_eq!(
            rust_ids, py_ids,
            "vllm-chat+vllm-tokenizer must match Python vllm on system+user"
        );
    }
}
