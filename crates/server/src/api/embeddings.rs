// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: POST /v1/embeddings (OpenAI-compatible). On a model whose capabilities say it
//! does not generate, each input string is tokenized as the checkpoint's processor does it
//! (BOS, the role prompt, the text, truncated to the role's token limit) and the request goes
//! to the scheduler as one `EmbedJob`; the scheduler packs every job of a tick into one
//! forward. On a generative model the route refuses with `EmbedNotSupported` (HTTP 501), and
//! on an embedding model the generation routes refuse with `GenerateNotSupported` (HTTP 501,
//! [`refuse_generation_on_embedder`]). Both refusals carry the code's name in `error.code`.
//!
//! `input_type` is required and must name one of the checkpoint's role prompts (`query`,
//! `passage` for nemotron-embed): an asymmetric embedder has no neutral role, so a missing or
//! other one is refused, never guessed. The response carries, besides the OpenAI fields, the
//! `prefix` it applied, which is always `"<input_type>: "`. `encoding_format: "base64"` returns each vector's
//! little-endian FP32 bytes, bit-exact.
//!
//! Owner: server API.
//! Invariants:
//! - `data[i].index == i`, in input order.

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::Engine as _;
use metrale_config::embedding::{EmbedRole, EmbeddingSpec};
use metrale_error::{ErrorCode, error};
use serde::Deserialize;

use super::compact::{openai_error_response, openai_error_response_with_param};
use crate::main_modules::model_host::{CurrentModel, ModelHost};

/// 2026-10-24: The request body. `input` is one string or an array of strings.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub model: Option<String>,
    pub input_type: Option<String>,
    pub encoding_format: Option<String>,
    pub dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

/// 2026-10-24: How the vectors are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Float,
    Base64,
}

/// 2026-10-24: A refusal: the status and the message.
type Refusal = (StatusCode, String);

fn bad(msg: String) -> Refusal {
    (StatusCode::BAD_REQUEST, msg)
}

/// 2026-10-24: The checked request: the role, the encoding and the texts. The `input_type`
/// names are the checkpoint's own (`spec`), so the prefix echoed is `"<input_type>: "`.
pub fn check_request<'r>(
    req: &'r EmbeddingRequest,
    spec: &EmbeddingSpec,
    dim: usize,
) -> Result<(EmbedRole, Encoding, Vec<&'r str>), Refusal> {
    let internal = |e: anyhow::Error| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"));
    let names = spec.input_types().map_err(internal)?.join("`, `");
    let ty = req.input_type.as_deref().ok_or_else(|| {
        bad(format!(
            "`input_type` is required: one of `{names}` (this model embeds its roles differently)"
        ))
    })?;
    let role = spec
        .role_for_input_type(ty)
        .map_err(internal)?
        .ok_or_else(|| bad(format!("`input_type` {ty:?} is not one of `{names}`")))?;
    let encoding = match req.encoding_format.as_deref() {
        None | Some("float") => Encoding::Float,
        Some("base64") => Encoding::Base64,
        Some(other) => {
            return Err(bad(format!(
                "`encoding_format` {other:?} is not float or base64"
            )));
        }
    };
    if let Some(d) = req.dimensions
        && d != dim
    {
        return Err(bad(format!(
            "`dimensions` {d}: this model returns {dim} only"
        )));
    }
    let texts: Vec<&str> = match &req.input {
        EmbeddingInput::One(s) => vec![s.as_str()],
        EmbeddingInput::Many(v) => v.iter().map(String::as_str).collect(),
    };
    if texts.is_empty() {
        return Err(bad("`input` is empty".into()));
    }
    Ok((role, encoding, texts))
}

/// 2026-10-24: `[BOS] + tokens(prompt + text)`, truncated to `max_tokens` with BOS kept, as
/// the checkpoint's processor truncates. `encode` is the serve's tokenizer (no specials).
pub fn tokenize_one(
    encode: &dyn Fn(&str) -> anyhow::Result<Vec<u32>>,
    bos: u32,
    prompt: &str,
    text: &str,
    max_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = Vec::with_capacity(max_tokens.min(text.len() + 8));
    ids.push(bos);
    ids.extend(encode(&format!("{prompt}{text}"))?);
    ids.truncate(max_tokens);
    Ok(ids)
}

/// 2026-10-24: The OpenAI response body for `vectors` (`[n * dim]`).
pub fn response_body(
    model: &str,
    prefix: &str,
    vectors: &[f32],
    dim: usize,
    encoding: Encoding,
    prompt_tokens: usize,
) -> serde_json::Value {
    let data: Vec<serde_json::Value> = vectors
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| {
            let embedding = match encoding {
                Encoding::Float => serde_json::json!(v),
                Encoding::Base64 => {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    serde_json::Value::String(
                        base64::engine::general_purpose::STANDARD.encode(bytes),
                    )
                }
            };
            serde_json::json!({"object": "embedding", "index": i, "embedding": embedding})
        })
        .collect();
    serde_json::json!({
        "object": "list",
        "data": data,
        "model": model,
        "prefix": prefix,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    })
}

/// 2026-10-24: The role prompt and limit the endpoint applies.
fn role_of(spec: &EmbeddingSpec, role: EmbedRole) -> Result<(String, usize), Refusal> {
    spec.role(role)
        .map(|(p, n)| (p.to_string(), n))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))
}

/// 2026-10-24: A route the loaded model's capabilities do not serve: HTTP 501, the coded text
/// and the code's name in `error.code`, so a client branches on the code, not the text.
pub fn capability_refusal(e: &metrale_error::Error) -> Response {
    openai_error_response_with_param(
        StatusCode::NOT_IMPLEMENTED,
        crate::scheduler::coded(e),
        None,
        Some(e.code().name()),
    )
}

/// 2026-10-24: Route layer on the generation routes: a request reaching an embedding model is
/// refused before the handler parses it, so no sequence is ever allocated for it.
pub async fn refuse_generation_on_embedder(
    axum::extract::State(host): axum::extract::State<std::sync::Arc<ModelHost>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    match host.current() {
        Some(state) if state.embedding.is_some() => capability_refusal(&error!(
            ErrorCode::GenerateNotSupported,
            state.model_name.clone()
        )),
        _ => next.run(req).await,
    }
}

/// 2026-10-24: POST /v1/embeddings
pub async fn embeddings(
    CurrentModel(state): CurrentModel,
    req: Result<Json<EmbeddingRequest>, JsonRejection>,
) -> Response {
    let Some(ep) = state.embedding.as_ref() else {
        return capability_refusal(&error!(
            ErrorCode::EmbedNotSupported,
            state.model_name.clone()
        ));
    };
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    let (role, encoding, texts) = match check_request(&req, &ep.spec, ep.dim) {
        Ok(v) => v,
        Err((s, m)) => return openai_error_response(s, m),
    };
    let (prompt, max_tokens) = match role_of(&ep.spec, role) {
        Ok(v) => v,
        Err((s, m)) => return openai_error_response(s, m),
    };
    let encode = |t: &str| state.tokenizer.encode(t);
    let mut inputs = Vec::with_capacity(texts.len());
    for t in &texts {
        match tokenize_one(&encode, ep.bos_token_id, &prompt, t, max_tokens) {
            Ok(ids) => inputs.push(ids),
            Err(e) => {
                return openai_error_response(StatusCode::BAD_REQUEST, format!("tokenize: {e:#}"));
            }
        }
    }
    let prompt_tokens: usize = inputs.iter().map(Vec::len).sum();
    let (tx, rx) = tokio::sync::oneshot::channel();
    if ep
        .tx
        .send(crate::scheduler::EmbedJob { inputs, reply: tx })
        .await
        .is_err()
    {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "the scheduler is not running".into(),
        );
    }
    match rx.await {
        Ok(Ok(v)) => {
            let model = req.model.as_deref().unwrap_or(&state.model_name);
            Json(response_body(
                model,
                &prompt,
                &v,
                ep.dim,
                encoding,
                prompt_tokens,
            ))
            .into_response()
        }
        Ok(Err(f)) => {
            let status = if f.code == Some(ErrorCode::EmbedInputInvalid) {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            openai_error_response(status, f.text)
        }
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the scheduler dropped the job".into(),
        ),
    }
}

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
