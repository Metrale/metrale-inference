// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Export production native RoPE geometry/math for numerical replay.
use anyhow::{Result, ensure};
use metrale_model_arch::qwen_image21::rope::ImageRopeLayout;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 2, "expected layout JSON and new output JSON");
    let input: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let mask: Vec<bool> = serde_json::from_value(input["mask"].clone())?;
    let shapes: Vec<[usize; 3]> = serde_json::from_value(input["shapes"].clone())?;
    let layout = ImageRopeLayout::new(&mask, &shapes)?;
    let data = serde_json::json!({"positions":layout.positions(),"cis_f32_bits":layout.frequencies().iter().map(|f|f.to_bits()).collect::<Vec<_>>(),"mask":mask,"shapes":shapes});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    file.write_all(serde_json::to_string(&data)?.as_bytes())?;
    Ok(())
}
