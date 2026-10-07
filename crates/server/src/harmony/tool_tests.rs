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

#[test]
fn constrained_json_header_is_token_aware_and_fail_closed() {
    let tokenizer =
        ByteTokenizer::from_tokenizer_json(include_str!("fixtures/gpt-oss-byte-vocab.json"))
            .unwrap();
    let prompt = [vec![200006], ids("assistant")].concat();
    for (format, duplicate, valid) in [
        ("json", false, true),
        ("xml", false, false),
        ("", false, false),
        ("json", true, false),
        ("json to=functions.other", false, false),
    ] {
        let mut output = [
            vec![200005],
            ids("commentary to=functions.lookup_part "),
            vec![200003],
            ids(format),
        ]
        .concat();
        if duplicate {
            output.extend([200003]);
            output.extend(ids("json"));
        }
        output.extend([200008]);
        output.extend(ids(r#"{"part_id":"A-42","count":2}"#));
        output.push(200012);
        assert_eq!(
            tool_response::response(&tokenizer, &prompt, &output, &[tool()]).is_ok(),
            valid
        );
    }
    // 2026-10-07: Format delimiters in body and tool-free constrained output are refused.
    for output in [
        [vec![200008, 200003], ids("json"), vec![200002]].concat(),
        [
            vec![200003],
            ids("json"),
            vec![200008],
            ids("{}"),
            vec![200002],
        ]
        .concat(),
    ] {
        assert!(tool_response::response(&tokenizer, &prompt, &output, &[tool()]).is_err());
    }
}

#[test]
fn stream_withholds_tool_until_valid_terminal_and_preserves_unicode() {
    use super::text_stream::TextStream;
    use std::sync::Arc;
    let tokenizer = Arc::new(
        ByteTokenizer::from_tokenizer_json(include_str!("fixtures/gpt-oss-byte-vocab.json"))
            .unwrap(),
    );
    let prompt = [vec![200006], ids("assistant")].concat();
    for (body, valid) in [
        (r#"{"part_id":"café 日本 😀","count":2}"#, true),
        (r#"{"part_id":"A","count":true}"#, false),
        (r#"{"part_id":"A","count":2,"count":3}"#, false),
        (r#"{"part_id":"A","count":2"#, false),
        (r#"{"part_id":"A","count":2,"extra":0}"#, false),
    ] {
        let mut parser = TextStream::with_tools(tokenizer.clone(), &prompt, vec![tool()]).unwrap();
        let tokens = [
            vec![200005],
            ids("commentary to=functions.lookup_part "),
            vec![200003],
            ids("json"),
            vec![200008],
            ids(body),
        ]
        .concat();
        for token in tokens {
            assert_eq!(parser.push(token).unwrap(), "");
            assert!(
                parser.take_tool_call().is_none(),
                "call leaked before validated terminal"
            );
        }
        assert!(parser.finish().is_err(), "truncated handoff accepted");
        assert_eq!(parser.push(200012).is_ok(), valid);
        if valid {
            parser.finish().unwrap();
            let call = parser.take_tool_call().unwrap();
            assert_eq!(call.arguments["part_id"], "café 日本 😀");
            assert!(parser.take_tool_call().is_none());
            assert!(parser.push(200012).is_err());
        } else {
            assert!(parser.take_tool_call().is_none());
            assert!(parser.finish().is_err());
        }
    }
}
