// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Tests for the DeepSeek-V4.1 encoder: the publisher's five reference
//! encodings byte for byte, then the behaviours the publisher's own unit tests pin
//! (effort budgets and rejections, the system token, spaced DSML tags, namespaces, drop
//! thinking, task tokens), and the Python float spelling of the shared JSON writer.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use serde_json::{Value, json};

use crate::tokenizer::deepseek_v4::python_json;
use crate::tokenizer::deepseek_v41::encode_messages;

const BOS: &str = "<｜begin▁of▁sentence｜>";

macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!("../../../test_data/deepseek_v41_encoding/", $name))
    };
}

const FIXTURES: [(&str, &str); 5] = [
    (fixture!("test_input_1.json"), fixture!("test_output_1.txt")),
    (fixture!("test_input_2.json"), fixture!("test_output_2.txt")),
    (fixture!("test_input_3.json"), fixture!("test_output_3.txt")),
    (fixture!("test_input_4.json"), fixture!("test_output_4.txt")),
    (fixture!("test_input_5.json"), fixture!("test_output_5.txt")),
];

/// 2026-10-18: The publisher's `load_cases` + `encode_case`: a case is an object or a bare
/// message list; `thinking_mode` defaults to chat. Case-level tools go to the encoder's
/// request-level `tools`, which lands them on `messages[0]` exactly as `load_cases` does
/// whenever that message is a system message (asserted, so the paths cannot diverge).
fn encode_fixture(input: &str) -> String {
    let case: Value = serde_json::from_str(input).expect("publisher fixture JSON");
    let (messages, case) = match &case {
        Value::Array(messages) => (messages, None),
        Value::Object(object) => (
            object["messages"].as_array().expect("messages"),
            Some(object),
        ),
        _ => panic!("fixture must be an object or a message list"),
    };
    let field = |key: &str| case.and_then(|c| c.get(key));
    let tools = field("tools").and_then(Value::as_array).map(Vec::as_slice);
    if tools.is_some() {
        assert_eq!(
            messages[0]["role"], "system",
            "fixture tools need a leading system"
        );
    }
    let thinking = match field("thinking_mode").and_then(Value::as_str) {
        None | Some("chat") => false,
        Some("thinking") => true,
        Some(other) => panic!("unknown fixture thinking_mode {other}"),
    };
    let effort = field("reasoning_effort").and_then(Value::as_str);
    encode_messages(messages, tools, thinking, effort).expect("publisher encoding")
}

fn assert_exact(case: usize, actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let mismatch = actual
        .bytes()
        .zip(expected.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| actual.len().min(expected.len()));
    let window = |text: &str| {
        let start = text.floor_char_boundary(mismatch.saturating_sub(80));
        let end = text.floor_char_boundary(text.len().min(mismatch + 120));
        text[start..end].to_string()
    };
    panic!(
        "DeepSeek-V4.1 fixture {case} mismatch at byte {mismatch}; lengths actual={} expected={}\nactual:   {:?}\nexpected: {:?}",
        actual.len(),
        expected.len(),
        window(actual),
        window(expected),
    );
}

fn encode(messages: Value, thinking: bool, effort: Option<&str>) -> anyhow::Result<String> {
    encode_messages(
        messages.as_array().expect("message list"),
        None,
        thinking,
        effort,
    )
}

fn effort_prefix(budget: u8) -> String {
    format!(
        "<｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
}

fn lookup_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    })
}

fn tool_call_messages(arguments: Value) -> Value {
    json!([
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": arguments}}],
        },
    ])
}

const TOOL_CALL_OUTPUT: &str = "  reason  </think>summary\n\n\
    <｜DSML｜ calls>\n\
    <｜DSML｜ invoke name=\"lookup\">\n\
    <｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n\
    <｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n\
    </｜DSML｜ invoke>\n\
    </｜DSML｜ calls><｜end▁of▁sentence｜>";

const QUESTION_PREFIX: &str = "<｜User｜>question<｜Assistant｜>";

#[test]
fn publisher_fixtures_are_byte_exact() {
    for (case, (input, output)) in FIXTURES.iter().enumerate() {
        assert_exact(case + 1, &encode_fixture(input), output);
    }
}

#[test]
fn effort_tiers_map_to_the_numeric_budget() {
    let messages = json!([{"role": "user", "content": "question"}]);
    for (effort, budget) in [
        (None, 75),
        (Some("low"), 50),
        (Some("high"), 75),
        (Some("max"), 100),
        (Some("xhigh"), 100),
    ] {
        assert_eq!(
            encode(messages.clone(), true, effort).unwrap(),
            format!(
                "{BOS}{}<｜User｜>question<｜Assistant｜><think>",
                effort_prefix(budget)
            ),
            "effort {effort:?}"
        );
    }
}

#[test]
fn unknown_efforts_are_rejected_in_both_modes() {
    let messages = json!([{"role": "user", "content": "question"}]);
    for thinking in [true, false] {
        for effort in ["medium", "minimal", "75", ""] {
            let error = encode(messages.clone(), thinking, Some(effort)).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("invalid DeepSeek-V4.1 reasoning effort"),
                "{effort:?}: {error}"
            );
        }
    }
}

#[test]
fn chat_mode_has_no_effort_and_no_system_token_without_a_system_message() {
    let prompt = encode(
        json!([{"role": "user", "content": "hello"}]),
        false,
        Some("max"),
    )
    .unwrap();
    assert_eq!(
        prompt,
        format!("{BOS}<｜User｜>hello<｜Assistant｜></think>")
    );
}

#[test]
fn leading_system_message_uses_the_system_token() {
    let prompt = encode(
        json!([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}]),
        false,
        None,
    )
    .unwrap();
    assert_eq!(
        prompt,
        format!(
            "{BOS}<｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>"
        )
    );
}

#[test]
fn mid_conversation_system_message_gets_its_token_and_the_assistant_header() {
    let prompt = encode(
        json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "system", "content": "mid sys"},
        ]),
        true,
        Some("low"),
    )
    .unwrap();
    assert_eq!(
        prompt,
        format!(
            "{BOS}{}sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>",
            effort_prefix(50)
        )
    );
}

#[test]
fn tool_calls_use_spaced_dsml_and_tolerate_double_encoded_arguments() {
    let arguments = r#"{"query":"value","limit":2}"#;
    for arguments in [
        json!(arguments),
        json!(serde_json::to_string(arguments).unwrap()),
        json!({"query": "value", "limit": 2}),
    ] {
        let prompt = encode(tool_call_messages(arguments.clone()), true, None).unwrap();
        assert_eq!(
            prompt,
            format!(
                "{BOS}{}{QUESTION_PREFIX}<think>{TOOL_CALL_OUTPUT}",
                effort_prefix(75)
            ),
            "arguments {arguments}"
        );
    }
}

#[test]
fn non_object_arguments_become_one_arguments_parameter() {
    let prompt = encode(tool_call_messages(json!("not json")), false, None).unwrap();
    assert!(prompt.contains(
        "<｜DSML｜ parameter name=\"arguments\" string=\"true\">not json</｜DSML｜ parameter>"
    ));
    let prompt = encode(tool_call_messages(json!("[1, 2]")), false, None).unwrap();
    assert!(prompt.contains(
        "<｜DSML｜ parameter name=\"arguments\" string=\"true\">[1, 2]</｜DSML｜ parameter>"
    ));
}

#[test]
fn tool_instructions_use_spaced_dsml_tags() {
    let prompt = encode(
        json!([{"role": "system", "content": "system", "tools": [lookup_tool()]}, {"role": "user", "content": "question"}]),
        false,
        None,
    )
    .unwrap();
    assert!(prompt.contains(
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>"
    ));
    for v4_tag in [
        "<｜DSML｜tool_calls>",
        "<｜DSML｜invoke",
        "<｜DSML｜parameter",
    ] {
        assert!(!prompt.contains(v4_tag), "{v4_tag}");
    }
}

#[test]
fn namespaced_tool_schemas_qualify_the_name_and_prefix_the_description() {
    let described = json!({"name": "search", "description": "Search tools."});
    for (on_tool, namespace) in [
        (true, json!("search")),
        (false, json!("search")),
        (true, described.clone()),
        (false, described.clone()),
    ] {
        let mut tool = lookup_tool();
        let target = if on_tool {
            &mut tool
        } else {
            &mut tool["function"]
        };
        target["namespace"] = namespace.clone();
        let prompt = encode(
            json!([{"role": "system", "content": "system", "tools": [tool]}]),
            false,
            None,
        )
        .unwrap();
        let description = if namespace.is_object() {
            "Search tools.\\nLook up a value"
        } else {
            "Look up a value"
        };
        assert!(
            prompt.contains(&format!(
                "{{\"name\": \"search::lookup\", \"description\": \"{description}\", \"parameters\": "
            )),
            "{prompt}"
        );
        assert!(!prompt.contains("\"namespace\":"));
    }
}

#[test]
fn namespaced_tool_calls_render_the_qualified_name() {
    let expected = TOOL_CALL_OUTPUT.replace("name=\"lookup\"", "name=\"search::lookup\"");
    let arguments = json!(r#"{"query":"value","limit":2}"#);
    for location in ["call", "function", "qualified"] {
        let mut messages = tool_call_messages(arguments.clone());
        let call = &mut messages[1]["tool_calls"][0];
        match location {
            "call" => call["namespace"] = json!("search"),
            "function" => call["function"]["namespace"] = json!("search"),
            _ => call["function"]["name"] = json!("search::lookup"),
        }
        let prompt = encode(messages, true, None).unwrap();
        assert!(prompt.ends_with(&expected), "{location}: {prompt}");
    }
}

#[test]
fn ambiguous_tool_namespaces_are_rejected() {
    for (name, namespace, error) in [
        (
            "search::lookup",
            json!("files"),
            "Conflicting tool namespaces",
        ),
        (
            "search::nested::lookup",
            Value::Null,
            "Tool name must not contain",
        ),
        (
            "lookup",
            json!("search::nested"),
            "Tool namespace must not contain",
        ),
    ] {
        let mut tool = lookup_tool();
        tool["function"]["name"] = json!(name);
        tool["namespace"] = namespace.clone();
        let schema_error = encode(
            json!([{"role": "system", "content": "s", "tools": [tool]}]),
            false,
            None,
        )
        .unwrap_err();
        assert!(schema_error.to_string().contains(error), "{schema_error}");

        let mut messages = tool_call_messages(json!("{}"));
        messages[1]["tool_calls"][0]["function"]["name"] = json!(name);
        messages[1]["tool_calls"][0]["namespace"] = namespace;
        let call_error = encode(messages, true, None).unwrap_err();
        assert!(call_error.to_string().contains(error), "{call_error}");
    }
}

#[test]
fn completed_turn_reasoning_is_dropped_without_tools() {
    let messages = json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]);
    let prompt = encode(messages.clone(), true, None).unwrap();
    assert!(prompt.contains("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"));
    assert!(!prompt.contains("r1"));
    assert!(prompt.contains("<think>r2</think>"));
    assert!(prompt.ends_with("<tool_result>value</tool_result><｜Assistant｜><think>"));

    let mut next_turn = messages.as_array().unwrap().clone();
    next_turn.push(json!({"role": "user", "content": "q3"}));
    let prompt = encode(Value::Array(next_turn), true, None).unwrap();
    assert!(!prompt.contains("r2"));
    assert!(prompt.ends_with("q3<｜Assistant｜><think>"));
}

#[test]
fn tool_results_follow_the_call_order() {
    let prompt = encode(
        json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "b", "content": "second"},
            {"role": "tool", "tool_call_id": "a", "content": "first"},
        ]),
        false,
        None,
    )
    .unwrap();
    assert!(prompt.ends_with(
        "<｜User｜><tool_result>first</tool_result>\n\n<tool_result>second</tool_result><｜Assistant｜></think>"
    ));
}

#[test]
fn non_action_task_appends_only_its_token() {
    let prompt = encode(
        json!([{"role": "user", "content": "classify me", "task": "query"}]),
        false,
        None,
    )
    .unwrap();
    assert!(prompt.ends_with("classify me<｜query｜>"));
    assert!(!prompt.contains("<｜Assistant｜>"));
    let error = encode(
        json!([{"role": "user", "content": "x", "task": "summarize"}]),
        false,
        None,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("invalid DeepSeek-V4.1 task"),
        "{error}"
    );
}

#[test]
fn request_tools_get_a_leading_system_message_when_the_conversation_has_none() {
    let messages = json!([{"role": "user", "content": "question"}]);
    let tools = [lookup_tool()];
    let prompt = encode_messages(messages.as_array().unwrap(), Some(&tools), false, None).unwrap();
    assert!(
        prompt.starts_with(&format!("{BOS}<｜System｜>\n\n## Tools\n")),
        "{prompt}"
    );
    assert!(prompt.ends_with("{\"name\": \"lookup\", \"description\": \"Look up a value\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>question<｜Assistant｜></think>"));
}

#[test]
fn developer_and_unknown_roles_are_rejected() {
    for role in ["developer", "critic"] {
        let error = encode(json!([{"role": role, "content": "x"}]), false, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported DeepSeek-V4.1 message role"),
            "{error}"
        );
    }
}

#[test]
fn image_placeholders_and_sourceless_images_are_rejected() {
    let placeholder = "<｜deepseek_image｜>";
    for message in [
        json!({"role": "user", "content": format!("look {placeholder}")}),
        json!({"role": "assistant", "content": "a", "reasoning_content": placeholder}),
        json!({"role": "user", "content": [{"type": "text", "text": placeholder}]}),
    ] {
        let error = encode(json!([message]), false, None).unwrap_err();
        assert!(error.to_string().contains("image placeholder"), "{error}");
    }
    let error = encode(
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}]),
        false,
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("valid source"), "{error}");
}

#[test]
fn json_floats_use_the_python_spelling() {
    for (value, python) in [
        (json!(1.0), "1.0"),
        (json!(0.1), "0.1"),
        (json!(-0.0), "-0.0"),
        (json!(123.456), "123.456"),
        (json!(0.0001), "0.0001"),
        (json!(0.00001), "1e-05"),
        (json!(1.5e-7), "1.5e-07"),
        (json!(100000.0), "100000.0"),
        (json!(1e15), "1000000000000000.0"),
        (json!(1e16), "1e+16"),
        (json!(1.2345678901234568e20), "1.2345678901234568e+20"),
        (json!(-2.5e300), "-2.5e+300"),
        (json!(7), "7"),
        (json!(-7), "-7"),
    ] {
        assert_eq!(python_json(&value).unwrap(), python, "{value}");
    }
}
