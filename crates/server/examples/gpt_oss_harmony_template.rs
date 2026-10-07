// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: CPU-only checkpoint-template oracle check; no weights or inference.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    messages: Vec<serde_json::Value>,
    reasoning_effort: Option<String>,
    input_ids: Vec<u32>,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: gpt_oss_harmony_template CHECKPOINT_METADATA ORACLE_JSON"
    );
    let tokenizer = metrale_server::tokenizer::ChatTokenizer::from_model_dir(
        std::path::Path::new(&args[1]),
        200002,
        true,
        "gpt_oss",
        None,
        true,
    )?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    ensure!(!cases.is_empty(), "empty oracle");
    for (index, case) in cases.iter().enumerate() {
        let actual = tokenizer
            .apply_chat_template_jinja_with_effort(
                &case.messages,
                None,
                true,
                false,
                case.reasoning_effort.as_deref(),
                None,
            )
            .with_context(|| format!("case {index} template render"))?;
        ensure!(
            actual == case.input_ids,
            "case {index}: template IDs differ: actual={actual:?}, expected={:?}",
            case.input_ids
        );
    }
    println!(
        "{}",
        serde_json::json!({"passed": true, "cases": cases.len(), "scope": "checkpoint template/tokenizer only; no model execution"})
    );
    Ok(())
}
