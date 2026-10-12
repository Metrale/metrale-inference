// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Header parsing, minting and error-body annotation of request identity.

use serde_json::json;

use super::*;

const TP: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
const RID: &str = "0af7651916cd43dd8448eb211c80319c";

fn fresh() -> Entropy {
    Entropy {
        request_id: [0x11; 16],
        trace_id: [0x22; 16],
        span_id: [0x33; 8],
    }
}

#[test]
fn a_valid_traceparent_parses() {
    let tp = parse_traceparent(TP).unwrap();
    assert_eq!(tp.trace_id.to_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(tp.parent_id.to_hex(), "00f067aa0ba902b7");
    assert_eq!(tp.flags, 1);
    let later = "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00-extra";
    assert_eq!(parse_traceparent(later).unwrap().flags, 0);
}

#[test]
fn invalid_traceparents_are_refused() {
    for bad in [
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "",
    ] {
        assert_eq!(parse_traceparent(bad), None, "{bad:?}");
    }
}

#[test]
fn request_ids_are_sixteen_bytes_in_hex_or_uuid_form() {
    let a = parse_request_id(RID).unwrap();
    assert_eq!(a.to_hex(), RID);
    assert_eq!(parse_request_id(&RID.to_ascii_uppercase()), Some(a));
    assert_eq!(
        parse_request_id("0af76519-16cd-43dd-8448-eb211c80319c"),
        Some(a)
    );
    for bad in [
        "req_0af7651916cd43dd8448eb211c80319c",
        "0af7651916cd43dd8448eb211c80319",
        "0af76519-16cd43dd-8448-eb211c80319c-",
        "0af765191-6cd-43dd-8448-eb211c80319c",
        "00000000000000000000000000000000",
        "",
    ] {
        assert_eq!(parse_request_id(bad), None, "{bad:?}");
    }
}

#[test]
fn header_ids_are_kept_unchanged_and_labelled_header() {
    let t = resolve(HeaderRead::One(RID), HeaderRead::One(TP), fresh()).unwrap();
    assert_eq!(t.request_id.to_hex(), RID);
    assert_eq!(t.request_id_source, IdSource::Header);
    assert_eq!(t.trace_id.to_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(t.trace_id_source, IdSource::Header);
    assert_eq!(t.parent_id.unwrap().to_hex(), "00f067aa0ba902b7");
    assert_eq!(t.span_id, SpanId([0x33; 8]));
    assert!(t.sampled);
}

#[test]
fn absent_and_invalid_headers_mint_and_say_which() {
    let t = resolve(HeaderRead::Absent, HeaderRead::Absent, fresh()).unwrap();
    assert_eq!(t.request_id, RequestId([0x11; 16]));
    assert_eq!(t.request_id_source, IdSource::GeneratedAbsent);
    assert_eq!(t.trace_id, TraceId([0x22; 16]));
    assert_eq!(t.trace_id_source, IdSource::GeneratedAbsent);
    assert_eq!(t.parent_id, None);
    assert!(!t.sampled);

    let t = resolve(HeaderRead::One("abc"), HeaderRead::One("garbage"), fresh()).unwrap();
    assert_eq!(t.request_id, RequestId([0x11; 16]));
    assert_eq!(t.request_id_source, IdSource::GeneratedInvalid);
    assert_eq!(t.trace_id_source, IdSource::GeneratedInvalid);

    let t = resolve(HeaderRead::Unreadable, HeaderRead::Unreadable, fresh()).unwrap();
    assert_eq!(t.request_id_source, IdSource::GeneratedInvalid);
    assert_eq!(t.trace_id_source, IdSource::GeneratedInvalid);
}

#[test]
fn all_zero_entropy_is_refused_only_where_an_id_is_minted() {
    let mut zero_trace = fresh();
    zero_trace.trace_id = [0; 16];
    assert!(resolve(HeaderRead::One(RID), HeaderRead::One(TP), zero_trace).is_ok());
    assert_eq!(
        resolve(HeaderRead::One(RID), HeaderRead::Absent, zero_trace),
        Err(ZeroEntropy)
    );
    let mut zero_request = fresh();
    zero_request.request_id = [0; 16];
    assert!(resolve(HeaderRead::One(RID), HeaderRead::Absent, zero_request).is_ok());
    assert_eq!(
        resolve(HeaderRead::Absent, HeaderRead::Absent, zero_request),
        Err(ZeroEntropy)
    );
    let mut zero_span = fresh();
    zero_span.span_id = [0; 8];
    assert_eq!(
        resolve(HeaderRead::One(RID), HeaderRead::One(TP), zero_span),
        Err(ZeroEntropy)
    );
}

#[test]
fn response_headers_carry_both_ids_and_their_sources() {
    let t = resolve(HeaderRead::One(RID), HeaderRead::Absent, fresh()).unwrap();
    let h = t.response_headers();
    assert_eq!(h[0], ("x-request-id", RID.to_string()));
    assert_eq!(
        h[1],
        (
            "traceresponse",
            format!("00-{}-{}-00", "22".repeat(16), "33".repeat(8))
        )
    );
    assert_eq!(
        h[2],
        (
            "x-metrale-id-source",
            "request_id=header, trace_id=generated-absent".to_string()
        )
    );
}

#[test]
fn error_envelopes_gain_both_ids_and_keep_their_fields() {
    let t = resolve(HeaderRead::One(RID), HeaderRead::One(TP), fresh()).unwrap();
    let openai = json!({"error": {"message": "m", "type": "invalid_request_error", "code": null}});
    let out = t
        .annotate_error_body(openai.to_string().as_bytes())
        .unwrap();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["error"]["message"], "m");
    assert_eq!(v["error"]["request_id"], RID);
    assert_eq!(v["error"]["trace_id"], "4bf92f3577b34da6a3ce929d0e0e4736");
    let keys: Vec<&String> = v["error"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["message", "type", "code", "request_id", "trace_id"]);

    let anthropic = json!({"type": "error", "error": {"type": "overloaded_error", "message": "x"}});
    let v: Value = serde_json::from_slice(
        &t.annotate_error_body(anthropic.to_string().as_bytes())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["request_id"], RID);
}

#[test]
fn bodies_without_an_error_object_are_left_alone() {
    let t = resolve(HeaderRead::Absent, HeaderRead::Absent, fresh()).unwrap();
    assert_eq!(t.annotate_error_body(b"not json"), None);
    assert_eq!(t.annotate_error_body(br#"{"error": "a string"}"#), None);
    assert_eq!(t.annotate_error_body(br#"{"status": "faulted"}"#), None);
    let mut v = json!({"choices": []});
    assert!(!t.annotate_error(&mut v));
    assert_eq!(v, json!({"choices": []}));
}
