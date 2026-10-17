// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Read one tensor of a safetensors checkpoint directory by name (through
//! `model.safetensors.index.json` and the shard's header offsets), for the GLM-5.3 examples
//! that run kernels on real weights.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.
#![allow(dead_code)]

use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, bail};
use half::bf16;

/// 2026-10-09: A tensor's dtype string, shape and raw little-endian bytes.
pub struct RawTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// 2026-10-09: The tensor `name` of the checkpoint in `dir`.
pub fn tensor_raw(dir: &str, name: &str) -> Result<RawTensor> {
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
        "{dir}/model.safetensors.index.json"
    ))?)?;
    let file = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} not in the index"))?;
    let mut f = std::fs::File::open(format!("{dir}/{file}"))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let hn = u64::from_le_bytes(n);
    let mut hdr = vec![0u8; hn as usize];
    f.read_exact(&mut hdr)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hdr)?;
    let t = &hdr[name];
    let shape: Vec<usize> = t["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let (a, b) = (
        t["data_offsets"][0].as_u64().context("off")?,
        t["data_offsets"][1].as_u64().context("off")?,
    );
    f.seek(SeekFrom::Start(8 + hn + a))?;
    let mut bytes = vec![0u8; (b - a) as usize];
    f.read_exact(&mut bytes)?;
    Ok(RawTensor {
        dtype: t["dtype"].as_str().unwrap_or_default().to_string(),
        shape,
        bytes,
    })
}

/// 2026-10-09: One BF16 tensor of the checkpoint as f32.
pub fn tensor(dir: &str, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let t = tensor_raw(dir, name)?;
    if t.dtype != "BF16" {
        bail!("{name}: {} not BF16", t.dtype);
    }
    Ok((
        t.shape,
        t.bytes
            .chunks(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    ))
}
