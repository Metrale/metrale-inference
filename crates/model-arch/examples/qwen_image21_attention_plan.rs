// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Export production packed visibility for bounded native CUDA replay.
use anyhow::{Context, Result};
use metrale_circuit::image_attention::{Layout, Token};
use serde::Deserialize;
#[derive(Deserialize)]
struct Input {
    image_ids: Vec<i32>,
    key_valid: Vec<Vec<bool>>,
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input: Input = serde_json::from_slice(&std::fs::read(args.next().context("input JSON")?)?)?;
    let path = args.next().context("new output JSON")?;
    let mut tokens = Vec::new();
    for (sample, validity) in input.key_valid.iter().enumerate() {
        anyhow::ensure!(validity.len() == input.image_ids.len(), "validity shape");
        for (&id, &key_valid) in input.image_ids.iter().zip(validity) {
            anyhow::ensure!(id >= -1, "invalid ID");
            tokens.push(Token {
                sample: u32::try_from(sample)?,
                image: (id >= 0).then_some(id as u32),
                key_valid,
            });
        }
    }
    let plan = Layout::new(tokens)?.packed_prefixes()?;
    let output = serde_json::json!({"image_ids":input.image_ids,"key_valid":input.key_valid,"gather":plan.gather,"spans":plan.spans});
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&serde_json::to_vec_pretty(&output)?)?;
    Ok(())
}
