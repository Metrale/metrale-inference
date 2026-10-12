// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The `InferenceUsage` record reaches the `/v1/responses` usage object: from a
//! stream's finish usage through the chat wire shape `finalize_responses_stream` reads, and
//! omitted, not null, when the record is withheld.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use metrale_qci_dump::contract::usage::InferenceUsage;

use crate::api::responses_translate::{build_responses_usage, chat_wire_usage};

/// 2026-10-11: The vendored contract's valid `completed` record.
fn record() -> InferenceUsage {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/platform/v1/fixtures/inference-usage/valid/completed.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn usage() -> crate::ir::Usage {
    crate::ir::Usage {
        prompt_tokens: 10,
        completion_tokens: 4,
        cached_prompt_tokens: 2,
        reasoning_tokens: 1,
        accepted_prediction_tokens: 0,
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        response_tokens_per_second: 0.0,
    }
}

#[test]
fn a_streamed_response_carries_the_record_in_its_usage() {
    let r = record();
    let wire =
        serde_json::to_value(build_responses_usage(&chat_wire_usage(&usage(), Some(&r)))).unwrap();
    assert_eq!(wire["input_tokens"], 10);
    assert_eq!(wire["output_tokens"], 4);
    assert_eq!(wire["inference_usage"], serde_json::to_value(&r).unwrap());
}

#[test]
fn a_withheld_record_is_omitted_not_null() {
    let wire =
        serde_json::to_value(build_responses_usage(&chat_wire_usage(&usage(), None))).unwrap();
    assert!(wire.get("inference_usage").is_none(), "{wire}");
    assert_eq!(wire["total_tokens"], 14);
}
