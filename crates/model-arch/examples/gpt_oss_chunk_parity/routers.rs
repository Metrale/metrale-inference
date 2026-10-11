// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded read-only decision evidence, separate from mathematical gates.
use anyhow::Result;
use metrale_model_arch::weight_loader::gpt_oss::runtime::DiagnosticTensor;
use std::{io::Write, path::Path};
pub fn save(
    output: &Path,
    width: usize,
    start: usize,
    rows: usize,
    layer: usize,
    snapshots: Vec<DiagnosticTensor>,
) -> Result<()> {
    let dir = output.join(format!("width{width}.routers"));
    std::fs::create_dir_all(&dir)?;
    for tensor in snapshots {
        if ![
            "router_logits",
            "router_ids",
            "router_scores",
            "post_attention_norm",
        ]
        .contains(&tensor.name)
        {
            continue;
        }
        let stride = tensor.bytes.len() / rows;
        for row in 0..rows {
            let position = start + row;
            if tensor.name == "post_attention_norm" && ![49, 215, 248].contains(&position) {
                continue;
            }
            let path = dir.join(format!("p{position}-l{layer}-{}.bin", tensor.name));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?;
            file.write_all(&tensor.bytes[row * stride..(row + 1) * stride])?;
        }
    }
    Ok(())
}

/// 2026-10-07: Bounded selected-expert operands and observed BF16 boundaries, not learned weights.
pub fn save_experts(
    output: &Path,
    width: usize,
    position: usize,
    layer: usize,
    snapshots: Vec<DiagnosticTensor>,
) -> Result<()> {
    let dir = output.join(format!("width{width}.experts"));
    std::fs::create_dir_all(&dir)?;
    for tensor in snapshots {
        let path = dir.join(format!("p{position}-l{layer}-{}.bin", tensor.name));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&tensor.bytes)?;
    }
    Ok(())
}
