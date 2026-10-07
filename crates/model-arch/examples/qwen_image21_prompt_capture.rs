// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Actual pinned tokenizer comparison without model weights or GPU use.
use metrale_model_arch::qwen_image21::prompt::TextPromptEncoder;
fn main() -> anyhow::Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("processor directory required"))?,
    );
    let tokenizer = std::fs::read(directory.join("tokenizer.json"))?;
    let template = std::fs::read(directory.join("chat_template.jinja"))?;
    let encoder = TextPromptEncoder::from_bytes(&tokenizer, &template)?;
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/qwen_image21_prompt.json"))?;
    let mut rows = Vec::new();
    // 2026-10-06: Known-bad assets and unsupported input precede the valid cases.
    let mut bad = tokenizer.clone();
    bad[0] ^= 1;
    anyhow::ensure!(
        TextPromptEncoder::from_bytes(&bad, &template).is_err(),
        "bad asset accepted"
    );
    anyhow::ensure!(encoder.encode("hello", 1).is_err(), "truncation accepted");
    anyhow::ensure!(
        encoder.encode("<|image_pad|>", 512).is_err(),
        "vision input accepted"
    );
    for case in reference["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let actual = encoder.encode(prompt, 512)?;
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone())?;
        anyhow::ensure!(actual.input_ids == expected, "prompt token mismatch");
        anyhow::ensure!(
            actual.drop_prefix == reference["system_ids"].as_array().unwrap().len(),
            "prefix mismatch"
        );
        rows.push(serde_json::json!({"prompt":prompt,"ids":actual.input_ids,"drop_prefix":actual.drop_prefix}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"cases":rows,"controls":3,"scope":"actual pinned tokenizer; no encoder or image execution"})
        )?
    );
    Ok(())
}
