// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The fail-closed GLM-4.7 policy on completed output
//! (`glm47::judge`, `parse_glm47_answer`): normal calls, the five failure
//! shapes it exists for (a name that is not offered, a key outside the schema,
//! a name with markup after it, a call opened inside the reasoning, a call
//! abandoned for another), and the argument typing.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::glm47::{REFUSAL_KEY, Verdict, judge, parse_glm47_answer, resolve_name};
use super::super::*;

fn tool(name: &str, parameters: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
        },
    }
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        tool(
            "get_weather",
            serde_json::json!({"type": "object", "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer"},
                "verbose": {"type": "boolean"}
            }}),
        ),
        tool(
            "write_file",
            serde_json::json!({"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            }}),
        ),
        tool(
            "bash",
            serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        ),
        tool(
            "web_search",
            serde_json::json!({"type": "object", "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            }}),
        ),
        tool(
            "get_status",
            serde_json::json!({"type": "object", "properties": {}}),
        ),
    ]
}

fn args(v: &Verdict) -> serde_json::Value {
    serde_json::from_str(v.arguments()).expect("verdict arguments are JSON")
}

fn accepted(v: &Verdict) -> (String, serde_json::Value) {
    assert!(
        matches!(v, Verdict::Accept { .. }),
        "expected Accept, got {v:#?}"
    );
    (v.name().to_string(), args(v))
}

fn refused(v: &Verdict) -> (String, String) {
    let Verdict::Refuse { reason, .. } = v else {
        panic!("expected Refuse, got {v:#?}");
    };
    let a = args(v);
    assert_eq!(
        a.as_object().map(|o| o.len()),
        Some(1),
        "a refusal carries exactly one argument: {a}"
    );
    assert_eq!(a[REFUSAL_KEY], serde_json::json!(reason));
    (v.name().to_string(), reason.clone())
}

#[test]
fn a_well_formed_call_is_accepted_with_schema_types() {
    let v = judge(
        "get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value>\
         <arg_key>days</arg_key><arg_value>3</arg_value>\
         <arg_key>verbose</arg_key><arg_value>True</arg_value>",
        &tools(),
    );
    let (name, a) = accepted(&v);
    assert_eq!(name, "get_weather");
    assert_eq!(
        a,
        serde_json::json!({"location": "Paris", "days": 3, "verbose": true})
    );
}

#[test]
fn a_string_parameter_keeps_json_looking_text_as_a_string() {
    // 2026-10-08: The shared chain reads every value as JSON when it parses
    // (`parse_poolside_v1_call`), so file content `{"a": 1}` or `42` became an
    // object or a number. Here the schema decides.
    let body = "write_file<arg_key>path</arg_key><arg_value>/t/x.json</arg_value>\
                <arg_key>content</arg_key><arg_value>{\"a\": 1}</arg_value>";
    let (_, a) = accepted(&judge(body, &tools()));
    assert_eq!(a["content"], serde_json::json!("{\"a\": 1}"));

    // 2026-10-08: Negative control: the shared chain turns the same value into
    // an object, which is the difference this policy exists for.
    let (_, mut calls) =
        parse_tool_calls_promoting_bare_names(&format!("<tool_call>{body}</tool_call>"));
    coerce_all(&mut calls, &tools());
    let shared: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(shared["content"], serde_json::json!({"a": 1}));
}

#[test]
fn a_bare_name_is_a_zero_argument_call() {
    let (name, a) = accepted(&judge("get_status", &tools()));
    assert_eq!(name, "get_status");
    assert_eq!(a, serde_json::json!({}));
}

#[test]
fn shape_1_a_name_that_was_not_offered_is_refused_not_dropped() {
    let body = "get_wether<arg_key>location</arg_key><arg_value>Paris</arg_value>";
    let (name, reason) = refused(&judge(body, &tools()));
    assert_eq!(
        name, "get_wether",
        "a name-shaped name is kept so the client can report it"
    );
    assert!(reason.contains("not one of the tools offered"), "{reason}");
}

#[test]
fn a_garbled_name_is_refused_as_unknown_tool() {
    let (name, reason) = refused(&judge("{\"name\": 1}", &tools()));
    assert_eq!(name, "unknown_tool");
    assert!(reason.contains("not a valid tool name"), "{reason}");
}

#[test]
fn shape_2_a_key_outside_the_schema_is_refused_with_the_valid_keys() {
    let body = "get_weather<arg_key>locaton</arg_key><arg_value>Paris</arg_value>";
    let (name, reason) = refused(&judge(body, &tools()));
    assert_eq!(name, "get_weather");
    assert!(reason.contains("\"locaton\""), "{reason}");
    assert!(
        reason.contains("days, location, verbose"),
        "valid keys, sorted: {reason}"
    );
}

#[test]
fn a_key_that_is_not_name_shaped_is_refused_even_without_a_schema() {
    let loose = vec![tool("free", serde_json::json!({"type": "object"}))];
    let ok = judge(
        "free<arg_key>anything</arg_key><arg_value>1</arg_value>",
        &loose,
    );
    let (_, a) = accepted(&ok);
    assert_eq!(
        a,
        serde_json::json!({"anything": "1"}),
        "no schema: values stay strings"
    );
    let bad = judge(
        "free<arg_key>two words</arg_key><arg_value>1</arg_value>",
        &loose,
    );
    let (_, reason) = refused(&bad);
    assert!(reason.contains("not a valid argument name"), "{reason}");
}

#[test]
fn shape_3_markup_after_a_name_resolves_to_the_offered_tool() {
    assert_eq!(
        resolve_name("bash</arg_key>", &tools()).as_deref(),
        Some("bash")
    );
    assert_eq!(
        resolve_name("bash1635", &tools()),
        None,
        "a longer name is another tool"
    );
    assert_eq!(resolve_name("bash", &tools()).as_deref(), Some("bash"));
    let (name, a) = accepted(&judge(
        "bash</arg_key>\n<arg_key>command</arg_key><arg_value>ls</arg_value>",
        &tools(),
    ));
    assert_eq!(name, "bash");
    assert_eq!(a, serde_json::json!({"command": "ls"}));
    let (name, _) = refused(&judge(
        "bash1635<arg_key>command</arg_key><arg_value>ls</arg_value>",
        &tools(),
    ));
    assert_eq!(name, "bash1635");
}

#[test]
fn shape_4_a_call_opened_inside_reasoning_is_refused_with_the_real_call() {
    // 2026-10-08: The first `<tool_call>` ended the reasoning, so this body runs
    // from it to the first `</tool_call>`, the real call's.
    let body = "bash<arg_key>command</arg_key><arg_value>ls\nwait, read it first\
                </think><tool_call>write_file<arg_key>path</arg_key><arg_value>/a.txt</arg_value>\
                <arg_key>content</arg_key><arg_value>hi</arg_value>";
    let (name, reason) = refused(&judge(body, &tools()));
    assert_eq!(name, "bash", "the outer call is the one refused");
    assert!(reason.contains("inside your reasoning"), "{reason}");
    assert!(
        reason.contains(r#"write_file with arguments {"path":"/a.txt","content":"hi"}"#),
        "{reason}"
    );
}

#[test]
fn shape_5_an_abandoned_call_is_refused_with_the_call_it_was_abandoned_for() {
    let body = "bash<arg_key>command</arg_key><arg_value># look it up\n\
                web_search<arg_key>query</arg_key><arg_value>rust</arg_value>\
                <arg_key>limit</arg_key><arg_value>5</arg_value>";
    let (name, reason) = refused(&judge(body, &tools()));
    assert_eq!(name, "bash");
    assert!(reason.contains("left unfinished"), "{reason}");
    assert!(
        reason.contains(r#"web_search with arguments {"query":"rust","limit":5}"#),
        "{reason}"
    );
}

#[test]
fn recovery_needs_a_valid_inner_call_else_the_outer_call_stands() {
    // 2026-10-08: File content that merely looks like markup: the inner name is
    // not offered, so nothing is recovered and the write goes through intact.
    // A value ends at its first `</arg_value>`, so the content holds none.
    let content = "parse(\"nope<arg_key>k</arg_key><arg_value>v\")";
    let body = format!(
        "write_file<arg_key>path</arg_key><arg_value>t.py</arg_value>\
         <arg_key>content</arg_key><arg_value>{content}</arg_value>"
    );
    let (name, a) = accepted(&judge(&body, &tools()));
    assert_eq!(name, "write_file");
    assert_eq!(a["content"], serde_json::json!(content));

    // 2026-10-08: An offered name that is the tail of a longer word is not an
    // abandoned call.
    let body = "bash<arg_key>command</arg_key><arg_value>echo mybash<arg_key>x</arg_value>";
    assert!(matches!(judge(body, &tools()), Verdict::Accept { .. }));
    // 2026-10-08: Negative control: the same text after a space is one.
    let body = "bash<arg_key>command</arg_key><arg_value>echo bash<arg_key>x</arg_value>";
    assert!(matches!(judge(body, &tools()), Verdict::Refuse { .. }));
}

#[test]
fn a_recovered_call_is_never_delivered_as_a_call() {
    let body = "write_file<arg_key>path</arg_key><arg_value>t.txt</arg_value>\
                <arg_key>content</arg_key><arg_value>x\nweb_search<arg_key>query</arg_key>\
                <arg_value>q</arg_value>";
    let v = judge(body, &tools());
    assert_ne!(v.name(), "web_search");
    refused(&v);
}

#[test]
fn a_repeated_key_keeps_its_first_position_and_last_value() {
    let body = "get_weather<arg_key>location</arg_key><arg_value>A</arg_value>\
                <arg_key>days</arg_key><arg_value>1</arg_value>\
                <arg_key>location</arg_key><arg_value>B</arg_value>";
    let v = judge(body, &tools());
    assert_eq!(v.arguments(), r#"{"location":"B","days":1}"#);
}

#[test]
fn text_between_pairs_is_ignored_and_an_unclosed_pair_is_not_an_argument() {
    let body = "get_weather\n<arg_key> location </arg_key>\n<arg_value>Paris</arg_value>\n\
                <arg_key>days</arg_key><arg_value>3";
    let (_, a) = accepted(&judge(body, &tools()));
    assert_eq!(a, serde_json::json!({"location": "Paris"}));
}

#[test]
fn answer_parse_keeps_content_and_judges_every_envelope_in_order() {
    let text = "Checking both.\n<tool_call>get_weather<arg_key>location</arg_key>\
                <arg_value>Paris</arg_value></tool_call>\n<tool_call>nope</tool_call>\n";
    let (content, verdicts) = parse_glm47_answer(text, &tools());
    assert_eq!(content.as_deref(), Some("Checking both."));
    assert_eq!(verdicts.len(), 2);
    accepted(&verdicts[0]);
    assert_eq!(refused(&verdicts[1]).0, "nope");
}

#[test]
fn answer_parse_judges_a_call_the_output_cut_short() {
    let text = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value>\
                <arg_key>days</arg_key><arg_value>3";
    let (content, verdicts) = parse_glm47_answer(text, &tools());
    assert_eq!(content, None);
    let (_, a) = accepted(&verdicts[0]);
    assert_eq!(a, serde_json::json!({"location": "Paris"}));
}

#[test]
fn answer_parse_leaves_other_formats_and_plain_text_alone() {
    let text = "Use <function=bash>ls</function> in Qwen.\n";
    let (content, verdicts) = parse_glm47_answer(text, &tools());
    assert!(verdicts.is_empty());
    assert_eq!(
        content.as_deref(),
        Some(text),
        "no call: the text is returned unchanged"
    );
    // 2026-10-08: Negative control: the shared parser makes a call of it.
    let (_, calls) = parse_tool_calls(text);
    assert_eq!(calls.len(), 1);
}

#[test]
fn types_follow_the_schema_including_lists_unions_enums_and_aliases() {
    let t = vec![tool(
        "f",
        serde_json::json!({"type": "object", "properties": {
            "n": {"type": ["integer", "null"]},
            "m": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "b": {"type": "bool"},
            "e": {"enum": [1, 2, 3]},
            "x": {"type": "number"},
            "o": {"type": "object"},
            "i": {"type": "integer"},
            "s": {"type": "string"}
        }}),
    )];
    let body = "f<arg_key>n</arg_key><arg_value>null</arg_value>\
                <arg_key>m</arg_key><arg_value>NULL</arg_value>\
                <arg_key>b</arg_key><arg_value>1</arg_value>\
                <arg_key>e</arg_key><arg_value>2</arg_value>\
                <arg_key>x</arg_key><arg_value>2.0</arg_value>\
                <arg_key>o</arg_key><arg_value>[1, 2]</arg_value>\
                <arg_key>i</arg_key><arg_value>abc</arg_value>\
                <arg_key>s</arg_key><arg_value>007</arg_value>";
    let (_, a) = accepted(&judge(body, &t));
    assert_eq!(
        a,
        serde_json::json!({
            "n": null, "m": null, "b": true, "e": 2, "x": 2,
            "o": [1, 2], "i": "abc", "s": "007"
        })
    );
}

#[test]
fn a_non_finite_number_stays_a_string() {
    let t = vec![tool(
        "f",
        serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
    )];
    let (_, a) = accepted(&judge(
        "f<arg_key>x</arg_key><arg_value>inf</arg_value>",
        &t,
    ));
    assert_eq!(a["x"], serde_json::json!("inf"));
}

#[test]
fn glm47_shares_poolside_v1_rendering_and_markers() {
    let calls = vec![IncomingToolCall {
        id: None,
        function: IncomingFunction {
            name: "get_weather".into(),
            arguments: r#"{"location":"Paris","days":3}"#.into(),
        },
    }];
    assert_eq!(
        Glm47Parser.format_tool_calls(&calls),
        PoolsideV1Parser.format_tool_calls(&calls)
    );
    let (g, p) = (Glm47Parser.leak_markers(), PoolsideV1Parser.leak_markers());
    assert_eq!(g.orphan_open, p.orphan_open);
    assert_eq!(g.close, p.close);
    assert_eq!(g.envelope_open, p.envelope_open);
    assert_eq!(g.envelope_close, p.envelope_close);
    assert_eq!(Glm47Parser.param_value_close_delim(), Some("</arg_value>"));
    assert!(Glm47Parser.has_tool_grammar());
    assert!(matches!(
        Glm47Parser.call_policy(),
        CallPolicy::FailClosed { .. }
    ));
    assert_eq!(PoolsideV1Parser.call_policy(), CallPolicy::Repairing);
    let fmt: ToolCallFormat = "glm47".parse().expect("glm47 parses");
    assert_eq!(fmt.name(), "glm47");
    assert_eq!(fmt.into_parser().name(), "glm47");
}

#[test]
fn the_glm47_reasoning_split_hands_a_nested_call_to_the_judge_intact() {
    let tool = |name: &str, key: &str| ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(serde_json::json!({"properties": {key: {"type": "string"}}})),
        },
    };
    let tools = [tool("bash", "command"), tool("web_search", "query")];
    let text = "check<tool_call>bash<arg_key>command</arg_key><arg_value>ls\nno, search\
                </think><tool_call>web_search<arg_key>query</arg_key><arg_value>q</arg_value>\
                </tool_call>";
    let (_, answer) = crate::reasoning_parser::ReasoningFormat::Glm47
        .into_parser()
        .extract_thinking(text, true);
    let (_, verdicts) = parse_glm47_answer(&answer, &tools);
    assert_eq!(verdicts.len(), 1);
    let Verdict::Refuse { name, reason, .. } = &verdicts[0] else {
        panic!("the nested call must be refused: {verdicts:#?}");
    };
    assert_eq!(name, "bash");
    assert!(
        reason.contains(r#"web_search with arguments {"query":"q"}"#),
        "{reason}"
    );
}
