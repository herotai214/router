//! OpenAI chat fields → `vllm-proto` `GenerateRequest`.
//!
//! This version always sets `prompt = TokenIds`. Proto also has text /
//! `media` / KV-transfer fields; they are unused here. Sampling-default
//! merge still lives in worker `vllm-text`; this only fills proto.

use crate::backend::pb;
use crate::protocols::spec::{ChatCompletionRequest, StringOrArray};

/// Wire adapter: OpenAI chat fields → crates.io `vllm-proto` 0.2.
pub fn chat_to_generate_request(
    request: &ChatCompletionRequest,
    token_ids: Vec<u32>,
    request_id: String,
) -> pb::GenerateRequest {
    let max_new_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(16);

    let (stop_strings, stop_token_ids) = split_stops(request);

    let mut decoding = pb::DecodingParameters {
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: request.repetition_penalty.unwrap_or(0.0),
        ..Default::default()
    };
    if let Some(regex) = &request.regex {
        decoding.structured_output = Some(pb::decoding_parameters::StructuredOutput::Regex(
            regex.clone(),
        ));
    } else if let Some(ebnf) = &request.ebnf {
        decoding.structured_output = Some(pb::decoding_parameters::StructuredOutput::Grammar(
            ebnf.clone(),
        ));
    }

    pb::GenerateRequest {
        request_id,
        model: std::env::var("VLLM_ROUTER_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| request.model.clone())
            .unwrap_or_default(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: token_ids,
        })),
        temperature: request.temperature,
        sampling: Some(pb::RandomSampling {
            num_sequences: request.n.unwrap_or(0),
            top_k: request.top_k.unwrap_or(0).max(0) as u32,
            top_p: request.top_p.unwrap_or(0.0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed,
        }),
        decoding: Some(decoding),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens,
            min_new_tokens: request.min_tokens.unwrap_or(0),
            stop_token_ids,
            stop_strings,
            include_stop_strings: request.no_stop_trim,
            ignore_eos: request.ignore_eos,
        }),
        response: Some(pb::ResponseOptions {
            // Token-out: router owns incremental detok (no worker prompt-id prime).
            output_text: Some(false),
            output_token_ids: true,
            skip_special_tokens: Some(request.skip_special_tokens),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn split_stops(request: &ChatCompletionRequest) -> (Vec<String>, Vec<u32>) {
    let stop_strings = match &request.stop {
        Some(StringOrArray::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(StringOrArray::Array(items)) => items.clone(),
        _ => Vec::new(),
    };
    let stop_token_ids = request
        .stop_token_ids
        .as_ref()
        .map(|ids| {
            ids.iter()
                .filter(|id| **id >= 0)
                .map(|id| *id as u32)
                .collect()
        })
        .unwrap_or_default();
    (stop_strings, stop_token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn token_ids_only_no_text_prompt() {
        // This-version default: convert always emits TokenIds. Not a forever
        // ban on a text prompt field if that is added later.
        let req = chat(serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0.2
        }));
        let proto = chat_to_generate_request(&req, vec![11, 22, 33], "req-1".into());
        match proto.prompt.unwrap() {
            pb::generate_request::Prompt::TokenIds(ids) => {
                assert_eq!(ids.ids, vec![11, 22, 33]);
            }
            other => panic!("expected token_ids, got {other:?}"),
        }
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 32);
        assert_eq!(proto.temperature, Some(0.2));
        assert_eq!(proto.model, "qwen");
        assert_eq!(proto.response.unwrap().output_text, Some(false));
    }

    #[test]
    fn prefers_max_completion_tokens() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "max_completion_tokens": 64
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into());
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 64);
    }
}
