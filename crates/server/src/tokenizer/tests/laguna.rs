// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render tests for the Laguna Jinja template.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use serde_json::json;

use crate::tokenizer::normalize_tool_call_arguments;

fn render_laguna_template(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/laguna.jinja"
    ))
    .expect("bundled Laguna template must be present");
    let env = crate::tokenizer::jinja_helpers::build_jinja_env(&raw).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages = normalize_tool_call_arguments(messages);
    tmpl.render(minijinja::context! {
        messages => minijinja::Value::from_serialize(&messages),
        tools => tools.map(minijinja::Value::from_serialize).unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        disable_tool_steering => false,
    })
    .expect("template renders")
}

#[test]
fn laguna_template_renders_native_tool_round_trip() {
    let messages = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "Bash",
                "arguments": "{\"command\":\"pwd\"}"
            }
        }]
    })];
    let rendered = render_laguna_template(&messages, None, true);
    assert!(rendered.contains(
        "<assistant><think></think><tool_call>Bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call></assistant>"
    ));
    assert!(rendered.ends_with("<assistant><think>"));
}

#[test]
fn laguna_template_uses_checkpoint_tool_json_and_reasoning_controls() {
    let messages = vec![json!({"role": "user", "content": "weather"})];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];

    let no_think = render_laguna_template(&messages, Some(&tools), false);
    assert!(no_think.contains(
        r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#
    ));
    assert!(no_think.ends_with("<assistant></think>"));

    let think = render_laguna_template(&messages, Some(&tools), true);
    assert!(think.ends_with("<assistant><think>"));
}

// 2026-10-07: Standalone binaries must carry the same reviewed template as repo launches.
#[test]
fn laguna_standalone_template_matches_reviewed_source() {
    let dir = tempfile::tempdir().unwrap();
    // 2026-10-07: Isolate cwd in a child, never mutate process-global cwd in parallel tests.
    if std::path::Path::new("jinja-templates/laguna.jinja").exists() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tokenizer::tests::laguna::laguna_standalone_template_matches_reviewed_source",
                "--nocapture",
            ])
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"chat_template": "{% generation %}checkpoint{% endgeneration %}"}).to_string(),
    )
    .unwrap();
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "laguna",
        Some(dir.path()),
        false,
    )
    .expect("Laguna starts without a source checkout");
    assert_eq!(
        tokenizer.chat_template,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../jinja-templates/laguna.jinja"
        ))
    );
    assert!(
        crate::tokenizer::ChatTokenizer::from_model_dir(
            dir.path(),
            0,
            true,
            "laguna",
            Some(dir.path()),
            true,
        )
        .is_err(),
        "disabling overrides must still select the checkpoint template"
    );
    let overrides = dir.path().join("jinja-templates");
    std::fs::create_dir(&overrides).unwrap();
    std::fs::write(overrides.join("laguna.jinja"), "explicit override").unwrap();
    let explicit = crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "laguna",
        Some(dir.path()),
        false,
    )
    .unwrap();
    assert_eq!(explicit.chat_template, "explicit override");
}
