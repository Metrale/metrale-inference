// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests for the `/v1/embeddings` request checks, tokenization and response body.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use super::*;

fn req(json: &str) -> EmbeddingRequest {
    serde_json::from_str(json).unwrap()
}

fn spec() -> EmbeddingSpec {
    EmbeddingSpec {
        pooling: metrale_config::embedding::Pooling::Mean,
        attention: metrale_config::embedding::AttentionMask::Bidirectional,
        query_max_tokens: 8,
        document_max_tokens: 16,
        prompts: Some(metrale_config::embedding::EmbeddingPrompts {
            query: "query: ".into(),
            document: "passage: ".into(),
        }),
    }
}

/// 2026-10-24: `input_type` is required and names a role by the checkpoint's prompts only.
#[test]
fn input_type_is_required_and_mapped() {
    let r = req(r#"{"input":"a"}"#);
    let (s, m) = check_request(&r, &spec(), 4).unwrap_err();
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(m.contains("`query`, `passage`"), "{m}");
    for ty in ["search", "document"] {
        let r = req(&format!(r#"{{"input":"a","input_type":"{ty}"}}"#));
        assert_eq!(
            check_request(&r, &spec(), 4).unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
    }
    let r = req(r#"{"input":["a","b"],"input_type":"passage"}"#);
    let (role, enc, texts) = check_request(&r, &spec(), 4).unwrap();
    assert_eq!(
        (role, enc, texts),
        (EmbedRole::Document, Encoding::Float, vec!["a", "b"])
    );
    let r = req(r#"{"input":"q","input_type":"query","encoding_format":"base64"}"#);
    let (role, enc, _) = check_request(&r, &spec(), 4).unwrap();
    assert_eq!((role, enc), (EmbedRole::Query, Encoding::Base64));
}

#[test]
fn refuses_other_widths_encodings_and_empty_input() {
    for body in [
        r#"{"input":"a","input_type":"query","dimensions":3}"#,
        r#"{"input":"a","input_type":"query","encoding_format":"int8"}"#,
        r#"{"input":[],"input_type":"query"}"#,
    ] {
        assert!(check_request(&req(body), &spec(), 4).is_err(), "{body}");
    }
    assert!(
        check_request(
            &req(r#"{"input":"a","input_type":"query","dimensions":4}"#),
            &spec(),
            4
        )
        .is_ok()
    );
}

/// 2026-10-24: BOS first, then the prompt and text as one string, and truncation keeps BOS
/// and the first `max - 1` text tokens (the checkpoint processor's truncation).
#[test]
fn tokenize_prepends_bos_and_prompt_and_truncates() {
    let seen = std::cell::RefCell::new(Vec::new());
    let enc = |t: &str| -> anyhow::Result<Vec<u32>> {
        seen.borrow_mut().push(t.to_string());
        Ok(t.bytes().map(u32::from).collect())
    };
    let ids = tokenize_one(&enc, 7, "p: ", "ab", 16).unwrap();
    assert_eq!(seen.borrow()[0], "p: ab");
    assert_eq!(
        ids,
        vec![
            7,
            b'p'.into(),
            b':'.into(),
            b' '.into(),
            b'a'.into(),
            b'b'.into()
        ]
    );
    let ids = tokenize_one(&enc, 7, "p: ", "abcdef", 4).unwrap();
    assert_eq!(ids, vec![7, b'p'.into(), b':'.into(), b' '.into()]);
}

/// 2026-10-24: Index order, the echoed prompt and usage, and base64 that decodes to the exact
/// FP32 bits.
#[test]
fn response_body_float_and_base64() {
    let v = [1.5f32, -0.0, f32::MIN_POSITIVE, 3.25, 7.0, -2.5];
    let b = response_body("m", "passage: ", &v, 3, Encoding::Float, 9);
    assert_eq!(b["data"].as_array().unwrap().len(), 2);
    assert_eq!(b["data"][1]["index"], 1);
    assert_eq!(b["data"][1]["embedding"][0].as_f64().unwrap(), 3.25);
    assert_eq!(b["prefix"], "passage: ");
    assert_eq!(b["usage"]["prompt_tokens"], 9);
    let b = response_body("m", "q", &v, 3, Encoding::Base64, 9);
    for (i, row) in v.chunks_exact(3).enumerate() {
        let s = b["data"][i]["embedding"].as_str().unwrap();
        let bytes = base64::engine::general_purpose::STANDARD.decode(s).unwrap();
        let got: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let want: Vec<u32> = row.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "row {i} must be bit-exact");
    }
}

/// 2026-10-24: A capability refusal is a 501 whose `error.code` names the registry code and
/// whose message leads with the coded form, for both directions.
#[test]
fn capability_refusals_are_coded_501s() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for (e, name) in [
        (
            error!(ErrorCode::GenerateNotSupported, "m"),
            "GenerateNotSupported",
        ),
        (
            error!(ErrorCode::EmbedNotSupported, "m"),
            "EmbedNotSupported",
        ),
    ] {
        let code = e.code();
        let r = capability_refusal(&e);
        assert_eq!(r.status(), StatusCode::NOT_IMPLEMENTED);
        let body = rt
            .block_on(axum::body::to_bytes(r.into_body(), 1 << 16))
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], name, "{v}");
        let msg = v["error"]["message"].as_str().unwrap();
        let lead = format!("[{:#06x} {name}] ", code.as_u16());
        assert!(msg.starts_with(&lead), "{msg}");
    }
}
