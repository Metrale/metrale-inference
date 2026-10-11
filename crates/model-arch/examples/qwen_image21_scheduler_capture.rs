// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Constructed host scheduler receipts; no model weights or GPU execution.
use half::bf16;
use metrale_model_arch::qwen_image21::scheduler::step_bf16_host;
fn main() -> anyhow::Result<()> {
    let mut cases = Vec::new();
    for delta in [-0.02000004f32, -0.028576255, -0.1234567] {
        let mut witnesses = Vec::new();
        for bits in 0..=u16::MAX {
            let prediction = bf16::from_bits(bits);
            if !prediction.is_finite() {
                continue;
            }
            let sample = bf16::from_f32([0., 1., -1., 0.5][bits as usize % 4]);
            let actual = step_bf16_host(&[sample], &[prediction], delta)?[0];
            let wrong = bf16::from_f32(sample.to_f32() + delta * prediction.to_f32());
            if actual != wrong {
                witnesses.push(serde_json::json!({"sample_bf16_bits":sample.to_bits(),
                    "prediction_bf16_bits":bits,"delta_f32_bits":delta.to_bits(),
                    "observed_bf16_bits":actual.to_bits(),"wrong_fp32_product_bits":wrong.to_bits()}));
            }
        }
        anyhow::ensure!(
            witnesses.len() >= 32,
            "missing known-bad detection witnesses"
        );
        for i in 0..32 {
            cases.push(witnesses[i * (witnesses.len() - 1) / 31].clone());
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope":"constructed native Rust host scheduler; no model weights or GPU execution",
            "cases":cases
        }))?
    );
    Ok(())
}
