// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Export actual native host-frequency bits for independent replay.
use anyhow::{Context, Result};
fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("new output JSON")?;
    let bits: Vec<u32> = metrale_model_arch::qwen_image21::conditioning::timestep_frequencies()
        .iter()
        .map(|f| f.to_bits())
        .collect();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&serde_json::to_vec(&bits)?)?;
    Ok(())
}
