// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Text and images through one admission of the shared recipe.
//!
//! Video never reaches a decoder here. An image is a data URI whose payload is
//! checked and counted, not decoded into pixels.

use serde_json::{Value, json};

use crate::recipe::Recipe;

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    code: &'static str,
    message: String,
}

impl ApiError {
    pub fn json(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": Value::Null,
                "code": self.code,
            }
        })
    }
}

fn err(code: &'static str, message: impl Into<String>) -> ApiError {
    ApiError {
        status: 400,
        code,
        message: message.into(),
    }
}

pub fn admit(recipe: &Recipe, model: &str, body: &[u8]) -> Result<Value, ApiError> {
    let doc: Value = serde_json::from_slice(body)
        .map_err(|e| err("malformed_json", format!("Invalid request JSON: {e}")))?;
    let messages = doc
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| err("malformed_input", "messages must be an array"))?;
    if messages.is_empty() {
        return Err(err("malformed_input", "messages is empty"));
    }

    let mut images: Vec<String> = Vec::new();
    let mut text_chars: u64 = 0;
    let mut image_bytes: u64 = 0;

    for message in messages {
        match message.get("content") {
            Some(Value::String(s)) => text_chars += s.chars().count() as u64,
            Some(Value::Array(parts)) => {
                for part in parts {
                    read_part(part, &mut images, &mut text_chars, &mut image_bytes)?;
                }
            }
            Some(Value::Null) | None => {}
            Some(_) => {
                return Err(err(
                    "malformed_input",
                    "message content must be a string or an array of parts",
                ));
            }
        }
    }

    if images.is_empty() && text_chars == 0 {
        return Err(err(
            "malformed_input",
            "the request has no text and no image",
        ));
    }
    if images.len() as u64 > recipe.max_images || text_chars > recipe.max_text_chars {
        return Err(err(
            "memory_pressure",
            format!(
                "context or memory pressure: {} images (max {}), {} text chars (max {})",
                images.len(),
                recipe.max_images,
                text_chars,
                recipe.max_text_chars
            ),
        ));
    }
    if image_bytes > recipe.max_image_bytes {
        return Err(err(
            "memory_pressure",
            format!(
                "context or memory pressure: image payload {image_bytes} bytes exceeds {}",
                recipe.max_image_bytes
            ),
        ));
    }

    let image_count = images.len() as u64;
    Ok(json!({
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "generation not run; coherence unmeasured",
            },
            "finish_reason": Value::Null,
        }],
        "qci_admission": {
            "recipe": recipe.id,
            "images_in_order": images,
            "image_count": image_count,
            "text_chars": text_chars,
            "coherence": "BLOCKER COHERENCE_UNMEASURED",
            "generation": "not-run",
            "video_decoding": "outside-runtime",
        }
    }))
}

fn read_part(
    part: &Value,
    images: &mut Vec<String>,
    text_chars: &mut u64,
    image_bytes: &mut u64,
) -> Result<(), ApiError> {
    let obj = part
        .as_object()
        .ok_or_else(|| err("malformed_input", "a content part must be an object"))?;
    let kind = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| err("malformed_input", "a content part has no type"))?;
    match kind {
        "text" => {
            let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
            *text_chars += text.chars().count() as u64;
            Ok(())
        }
        "image_url" | "image" | "input_image" => {
            let url = image_url(obj)?;
            if is_video_uri(&url) {
                return Err(video_error());
            }
            let bytes = inline_image(&url)?;
            *image_bytes += bytes;
            images.push(url);
            Ok(())
        }
        "video_url" | "video" | "input_video" => Err(video_error()),
        other => Err(err(
            "malformed_input",
            format!("unknown content part type {other}"),
        )),
    }
}

fn video_error() -> ApiError {
    err(
        "video_outside_runtime",
        "video decoding stays outside the runtime",
    )
}

fn image_url(obj: &serde_json::Map<String, Value>) -> Result<String, ApiError> {
    let raw = obj
        .get("image_url")
        .or_else(|| obj.get("image"))
        .ok_or_else(|| err("malformed_image", "image part has no image_url"))?;
    let url = match raw {
        Value::String(s) => s.clone(),
        Value::Object(o) => o
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => {
            return Err(err(
                "malformed_image",
                "image_url must be a string or an object with url",
            ));
        }
    };
    if url.is_empty() {
        return Err(err("malformed_image", "image_url.url is empty"));
    }
    Ok(url)
}

fn is_video_uri(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.starts_with("data:video/")
        || lower.starts_with("data:application/mp4")
        || lower.contains("video/mp4")
}

fn inline_image(url: &str) -> Result<u64, ApiError> {
    let Some(rest) = url.strip_prefix("data:") else {
        if url.starts_with("http://") || url.starts_with("https://") {
            return Err(err(
                "image_not_inline",
                "a remote image URL is outside this admission",
            ));
        }
        return Err(err(
            "malformed_image",
            "image_url must be an inline data:image URI",
        ));
    };
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err(err("malformed_image", "data URI has no comma"));
    };
    let meta = meta.to_ascii_lowercase();
    if !meta.starts_with("image/") || !meta.contains(";base64") {
        return Err(err("malformed_image", "data URI must be image/*;base64"));
    }
    base64_decoded_len(payload)
}

fn base64_decoded_len(payload: &str) -> Result<u64, ApiError> {
    if payload.is_empty() {
        return Err(err("malformed_image", "image base64 is empty"));
    }
    if payload.bytes().any(|b| b.is_ascii_whitespace()) {
        return Err(err("malformed_image", "image base64 contains whitespace"));
    }
    let padding = payload.bytes().rev().take_while(|b| *b == b'=').count();
    if padding > 2 || (padding > 0 && !payload.len().is_multiple_of(4)) {
        return Err(err("malformed_image", "image base64 padding is wrong"));
    }
    let body = &payload[..payload.len() - padding];
    if body.is_empty() || body.len() % 4 == 1 {
        return Err(err("malformed_image", "image base64 length is wrong"));
    }
    if !body
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/')
    {
        return Err(err("malformed_image", "image base64 is not valid"));
    }
    let len = (body.len() as u64) * 3 / 4;
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate;
    use crate::recipe::for_model;

    const MODEL: &str = "unsloth/gemma-4-26B-A4B-it-GGUF";

    fn png(payload: &str) -> String {
        format!("data:image/png;base64,{payload}")
    }

    fn user(content: Value) -> Vec<u8> {
        serde_json::to_vec(
            &json!({"model": MODEL, "messages": [{"role": "user", "content": content}]}),
        )
        .unwrap()
    }

    #[test]
    fn image_ordering() {
        let recipe = for_model(MODEL).unwrap();
        let first = png("AAAA");
        let second = png("BBBB");
        let body = user(json!([
            {"type": "text", "text": "which came first"},
            {"type": "image_url", "image_url": {"url": first}},
            {"type": "image_url", "image_url": {"url": second}},
        ]));
        let ok = admit(&recipe, MODEL, &body).unwrap();
        assert_eq!(
            ok["qci_admission"]["images_in_order"],
            json!([first, second])
        );
        let swapped = user(json!([
            {"type": "image_url", "image_url": {"url": second}},
            {"type": "image_url", "image_url": {"url": first}},
        ]));
        let other = admit(&recipe, MODEL, &swapped).unwrap();
        assert_eq!(
            other["qci_admission"]["images_in_order"],
            json!([second, first])
        );
    }

    #[test]
    fn malformed_input() {
        let recipe = for_model(MODEL).unwrap();
        let body = user(json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@@"}},
        ]));
        let err = admit(&recipe, MODEL, &body).unwrap_err();
        assert_eq!(err.status, 400);
        assert_eq!(err.code, "malformed_image");
        let missing = user(json!([{"type": "image_url"}]));
        assert_eq!(
            admit(&recipe, MODEL, &missing).unwrap_err().code,
            "malformed_image"
        );
    }

    #[test]
    fn context_memory_pressure() {
        let recipe = for_model(MODEL).unwrap();
        let chars = (recipe.max_image_bytes as usize).saturating_mul(4) / 3 + 8;
        let chars = chars.div_ceil(4) * 4;
        let huge = "A".repeat(chars);
        let body = user(json!([
            {"type": "image_url", "image_url": {"url": png(&huge)}},
        ]));
        let err = admit(&recipe, MODEL, &body).unwrap_err();
        assert_eq!(err.code, "memory_pressure");
        let mut parts = Vec::new();
        for _ in 0..=recipe.max_images {
            parts.push(json!({"type": "image_url", "image_url": {"url": png("AAAA")}}));
        }
        let many = user(Value::Array(parts));
        assert_eq!(
            admit(&recipe, MODEL, &many).unwrap_err().code,
            "memory_pressure"
        );
    }

    #[test]
    fn coherent_outputs() {
        let recipe = for_model(MODEL).unwrap();
        let url = png("AAAA");
        let body = user(json!([
            {"type": "text", "text": "describe the frame"},
            {"type": "image_url", "image_url": {"url": url}},
        ]));
        let ok = admit(&recipe, MODEL, &body).unwrap();
        let text = serde_json::to_string(&ok).unwrap();
        assert!(!text.to_ascii_lowercase().contains("supported"));
        assert_eq!(ok["object"], "chat.completion");
        assert_eq!(ok["choices"][0]["message"]["role"], "assistant");
        assert!(
            !ok["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .is_empty()
        );
        assert!(ok["choices"][0]["finish_reason"].is_null());
        assert_eq!(ok["qci_admission"]["images_in_order"], json!([url]));
        assert_eq!(
            ok["qci_admission"]["coherence"],
            "BLOCKER COHERENCE_UNMEASURED"
        );
        assert_eq!(ok["qci_admission"]["generation"], "not-run");
        assert!(ok.get("error").is_none());
        let text_only = user(json!("describe the frame"));
        let also = admit(&recipe, MODEL, &text_only).unwrap();
        assert_eq!(
            also["qci_admission"]["recipe"],
            ok["qci_admission"]["recipe"]
        );
    }

    #[test]
    fn api_errors() {
        let broken = gate(MODEL, b"{");
        match broken {
            crate::Gate::Respond { status, body } => {
                assert_eq!(status, 400);
                assert_eq!(body["error"]["type"], "invalid_request_error");
                assert_eq!(body["error"]["code"], "malformed_json");
            }
            crate::Gate::Pass => panic!("a campaign model must not pass a broken body"),
        }
        let remote = user(json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]));
        match gate(MODEL, &remote) {
            crate::Gate::Respond { status, body } => {
                assert_eq!(status, 400);
                assert_eq!(body["error"]["type"], "invalid_request_error");
                assert_eq!(body["error"]["code"], "image_not_inline");
            }
            crate::Gate::Pass => panic!("remote image must be an API error"),
        }
    }

    #[test]
    fn video_decoding_stays_outside_the_runtime() {
        let payload = "AAAAVIDEO";
        let body = user(json!([
            {"type": "text", "text": "what happens"},
            {"type": "video_url", "video_url": {"url": format!("data:video/mp4;base64,{payload}")}},
            {"type": "image_url", "image_url": {"url": png("AAAA")}},
        ]));
        match gate(MODEL, &body) {
            crate::Gate::Respond { status, body } => {
                assert_eq!(status, 400);
                let text = serde_json::to_string(&body).unwrap();
                assert_eq!(body["error"]["code"], "video_outside_runtime");
                assert!(text.contains("video decoding stays outside the runtime"));
                assert!(!text.contains(payload));
                assert!(body.get("qci_admission").is_none());
            }
            crate::Gate::Pass => panic!("video must not pass"),
        }
        assert!(gate("openai/gpt-oss-20b", &body) == crate::Gate::Pass);
    }
}
