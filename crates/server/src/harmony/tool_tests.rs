// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Recipient, schema, JSON ambiguity and tool-ending refusal controls.
use super::{stream::ByteTokenizer, tool_response, tool_schema::ToolSchema};
use serde_json::json;
fn ids(text: &str) -> Vec<u32> {
    let alphabet: Vec<u8> = (33..=126)
        .chain(161..=172)
        .chain(174..=255)
        .chain((0..=255).filter(|b| !matches!(b,33..=126|161..=172|174..=255)))
        .collect();
    text.bytes()
        .map(|b| alphabet.iter().position(|v| *v == b).unwrap() as u32)
        .collect()
}
fn tool() -> ToolSchema {
    ToolSchema::new("lookup_part",json!({"type":"object","properties":{"part_id":{"type":"string"},"count":{"type":"integer"}},"required":["part_id","count"],"additionalProperties":false})).unwrap()
}
#[test]
fn exact_recipient_json_and_schema_are_required_for_handoff() {
    let tokenizer =
        ByteTokenizer::from_tokenizer_json(include_str!("fixtures/gpt-oss-byte-vocab.json"))
            .unwrap();
    let prompt = [vec![200006], ids("assistant")].concat();
    for (name, body, end, valid) in [
        (
            "lookup_part",
            r#"{"part_id":"A-42","count":2}"#,
            200012,
            true,
        ),
        (
            "LOOKUP_PART",
            r#"{"part_id":"A-42","count":2}"#,
            200012,
            false,
        ),
        (
            "lookup_part",
            r#"{"part_id":"A-42","count":true}"#,
            200012,
            false,
        ),
        ("lookup_part", r#"{"part_id":"A-42"}"#, 200012, false),
        (
            "lookup_part",
            r#"{"part_id":"A-42","count":2,"extra":1}"#,
            200012,
            false,
        ),
        (
            "lookup_part",
            r#"{"part_id":"A-42","count":2,"count":3}"#,
            200012,
            false,
        ),
        (
            "lookup_part",
            r#"{"part_id":"A-42","count":2} {}"#,
            200012,
            false,
        ),
        (
            "lookup_part",
            r#"{"part_id":"A-42","count":2}"#,
            200002,
            false,
        ),
    ] {
        let output = [
            ids(&format!(" to=functions.{name}")),
            vec![200005],
            ids("commentary json"),
            vec![200008],
            ids(body),
            vec![end],
        ]
        .concat();
        let result = tool_response::response(&tokenizer, &prompt, &output, &[tool()]);
        assert_eq!(result.is_ok(), valid, "{name} {body}");
        if let Ok(parsed) = result {
            let call = parsed.tool_call.unwrap();
            assert_eq!(call.name, "lookup_part");
            assert_eq!(call.arguments["count"], 2);
            assert!(parsed.content.is_none());
        }
    }
}
#[test]
fn schema_subset_fails_closed_and_nested_duplicate_keys_are_refused() {
    for schema in [
        json!({"type":"object","$ref":"https://example.com/schema"}),
        json!({"type":"object","properties":{"x":{"type":"string","pattern":"a"}}}),
        json!({"type":"object","required":["missing"]}),
        json!({"type":"object","additionalProperties":{}}),
    ] {
        assert!(ToolSchema::new("lookup_part", schema).is_err());
    }
    assert!(ToolSchema::new("lookup-part", json!({"type":"object"})).is_err());
    assert!(super::strict_json::parse(r#"{"outer":{"same":1,"same":2}}"#).is_err());
    assert!(super::strict_json::parse(r#"{"outer":[{"same":1,"same":2}]}"#).is_err());
    let t=ToolSchema::new("nested",json!({"type":"object","properties":{"values":{"type":"array","items":{"type":"string","enum":["ok"]}}},"required":["values"],"additionalProperties":false})).unwrap();
    assert!(t.validate(&json!({"values":["ok"]})).is_ok());
    assert!(t.validate(&json!({"values":["bad"]})).is_err());
    assert!(t.validate(&json!({"values":[1]})).is_err());
}

#[test]
fn numeric_enum_is_exact_without_boolean_or_large_integer_coercion() {
    let t=ToolSchema::new("numeric",json!({"type":"object","properties":{"n":{"type":"integer","enum":[1,9007199254740993u64]}},"required":["n"]})).unwrap();
    assert!(t.validate(&json!({"n":1.0})).is_ok());
    assert!(t.validate(&json!({"n":true})).is_err());
    assert!(t.validate(&json!({"n":9007199254740993u64})).is_ok());
    assert!(t.validate(&json!({"n":9007199254740992u64})).is_err());
    assert!(t.validate(&json!({"n":9007199254740992.0})).is_err());
}
