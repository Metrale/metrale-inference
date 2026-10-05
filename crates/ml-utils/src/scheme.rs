// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The storage scheme of each quantized weight, read off the tensor index: the
//! weight tensor and the scale tensors stored beside it. A mock writes the same tensors with the
//! same dtypes and shapes, so a loader sees the layout the checkpoint has.
//!
//! | weight | companions | scheme |
//! |---|---|---|
//! | U8/I8 `[N, K/2]` | `weight_scale` F8 `[N, K/16]`, `weight_scale_2` | NVFP4, ModelOpt global (`amax / (6*448)`) |
//! | U8/I8 `[N, K/2]` | `weight_scale` F8 `[N, K/16]`, `weight_global_scale` | NVFP4, compressed-tensors global (`(6*448) / amax`) |
//! | F8 `[N, K]` | `weight_scale_inv` `[ceil(N/bn), ceil(K/bk)]` | FP8 block (`weight_block_size`) |
//! | F8 `[N, K]` | `weight_scale` `[N, 1]` or `[N]` | FP8 per channel |
//! | F8 `[N, K]` | `weight_scale` `[]` or `[1]` | FP8 per tensor |
//!
//! Either may carry `input_scale` / `input_global_scale` (a static activation scale).
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A quantized weight whose companions do not fit exactly one row of the table is refused,
//!   as is a scale-like tensor that belongs to no weight.

use std::collections::BTreeSet;

use crate::error::{MlError, Result};
use crate::index::{Dtype, TensorEntry, TensorIndex};

/// 2026-10-03: Where an NVFP4 tensor's per-tensor scale lives and what it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Global {
    /// 2026-10-03: `weight_scale_2 = amax / (6 * 448)` (ModelOpt).
    ModelOpt,
    /// 2026-10-03: `weight_global_scale = (6 * 448) / amax` (compressed-tensors).
    CompressedTensors,
}

/// 2026-10-03: A quantized weight's storage scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// 2026-10-03: E2M1 pairs, E4M3 scales per 16, one global.
    Nvfp4(Nvfp4Global),
    /// 2026-10-03: E4M3 with one scale per `[bn, bk]` block.
    Fp8Block {
        /// 2026-10-03: Block rows.
        bn: u64,
        /// 2026-10-03: Block columns.
        bk: u64,
    },
    /// 2026-10-03: E4M3 with one scale per output row.
    Fp8Channel,
    /// 2026-10-03: E4M3 with one scale for the tensor.
    Fp8Tensor,
}

/// 2026-10-03: One quantized weight and the tensors stored with it (source names).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantGroup {
    /// 2026-10-03: Scheme.
    pub scheme: Scheme,
    /// 2026-10-03: The module (the weight name without its last segment).
    pub module: String,
    /// 2026-10-03: The weight tensor.
    pub weight: String,
    /// 2026-10-03: The block, channel or tensor scale.
    pub scale: String,
    /// 2026-10-03: The NVFP4 global scale.
    pub global: Option<String>,
    /// 2026-10-03: A static activation scale.
    pub input: Option<String>,
    /// 2026-10-03: Logical rows (output features).
    pub rows: u64,
    /// 2026-10-03: Logical columns (input features).
    pub cols: u64,
}

impl QuantGroup {
    /// 2026-10-03: Every tensor of the group.
    pub fn tensors(&self) -> Vec<&str> {
        let mut v = vec![self.weight.as_str(), self.scale.as_str()];
        v.extend(self.global.as_deref());
        v.extend(self.input.as_deref());
        v
    }
}

/// 2026-10-03: The last segment of names that are scales of some weight.
const SCALE_SUFFIXES: [&str; 6] = [
    "weight_scale",
    "weight_scale_2",
    "weight_scale_inv",
    "weight_global_scale",
    "input_scale",
    "input_global_scale",
];

/// 2026-10-03: Whether `name` is a scale tensor by its last segment.
pub fn is_scale_name(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, last)| SCALE_SUFFIXES.contains(&last))
}

fn scalar(e: &TensorEntry) -> bool {
    e.numel() == 1 && e.shape.len() <= 1
}

fn refuse(name: &str, why: String) -> MlError {
    MlError::Tensor {
        name: name.to_string(),
        why,
    }
}

/// 2026-10-03: The quantized weights of `index`. `block` is the FP8 block size the metadata
/// declares (`weight_block_size`), needed only by a `weight_scale_inv` weight.
pub fn find_groups(index: &TensorIndex, block: Option<(u64, u64)>) -> Result<Vec<QuantGroup>> {
    let mut groups = Vec::new();
    let mut consumed = BTreeSet::new();
    for e in index.iter() {
        let Some((module, last)) = e.name.rsplit_once('.') else {
            continue;
        };
        let quantized = matches!(e.dtype, Dtype::U8 | Dtype::I8 | Dtype::F8E4m3);
        if !quantized || !(last == "weight" || last == "weight_packed") {
            continue;
        }
        let g = group_of(index, e, module, block)?;
        consumed.extend(g.tensors().into_iter().map(str::to_string));
        groups.push(g);
    }
    for e in index.iter() {
        if is_scale_name(&e.name) && !consumed.contains(&e.name) {
            return Err(refuse(
                &e.name,
                "a scale tensor that belongs to no quantized weight".into(),
            ));
        }
    }
    Ok(groups)
}

fn group_of(
    index: &TensorIndex,
    w: &TensorEntry,
    module: &str,
    block: Option<(u64, u64)>,
) -> Result<QuantGroup> {
    let sib = |s: &str| index.get(&format!("{module}.{s}"));
    let name_of = |e: &TensorEntry| e.name.clone();
    let [rows, stored_cols] = w.shape[..] else {
        return Err(refuse(
            &w.name,
            format!("a quantized weight of rank {}", w.shape.len()),
        ));
    };
    let input = match (sib("input_scale"), sib("input_global_scale")) {
        (Some(e), None) | (None, Some(e)) if scalar(e) => Some(name_of(e)),
        (None, None) => None,
        _ => return Err(refuse(&w.name, "malformed or doubled input scales".into())),
    };
    if matches!(w.dtype, Dtype::U8 | Dtype::I8) {
        let cols = stored_cols * 2;
        let scale = sib("weight_scale")
            .filter(|s| s.dtype == Dtype::F8E4m3 && s.shape == [rows, cols / 16])
            .ok_or_else(|| {
                refuse(
                    &w.name,
                    format!(
                        "NVFP4 needs an F8_E4M3 weight_scale [{rows}, {}]",
                        cols / 16
                    ),
                )
            })?;
        let (global, kind) = match (sib("weight_scale_2"), sib("weight_global_scale")) {
            (Some(g), None) if scalar(g) => (g, Nvfp4Global::ModelOpt),
            (None, Some(g)) if scalar(g) => (g, Nvfp4Global::CompressedTensors),
            _ => {
                return Err(refuse(
                    &w.name,
                    "NVFP4 needs exactly one scalar weight_scale_2 or weight_global_scale".into(),
                ));
            }
        };
        if !cols.is_multiple_of(16) {
            return Err(refuse(
                &w.name,
                format!("{cols} columns is not a multiple of 16"),
            ));
        }
        return Ok(QuantGroup {
            scheme: Scheme::Nvfp4(kind),
            module: module.to_string(),
            weight: w.name.clone(),
            scale: name_of(scale),
            global: Some(name_of(global)),
            input,
            rows,
            cols,
        });
    }
    let cols = stored_cols;
    let (scale, scheme) = match (sib("weight_scale_inv"), sib("weight_scale")) {
        (Some(s), None) => {
            let (bn, bk) = block.ok_or_else(|| {
                refuse(
                    &w.name,
                    "a weight_scale_inv but the metadata states no weight_block_size".into(),
                )
            })?;
            if s.shape != [rows.div_ceil(bn), cols.div_ceil(bk)] {
                return Err(refuse(
                    &w.name,
                    format!(
                        "weight_scale_inv {:?} is not the [{bn}, {bk}] block grid",
                        s.shape
                    ),
                ));
            }
            (s, Scheme::Fp8Block { bn, bk })
        }
        (None, Some(s)) if scalar(s) => (s, Scheme::Fp8Tensor),
        (None, Some(s)) if s.shape == [rows, 1] || s.shape == [rows] => (s, Scheme::Fp8Channel),
        _ => {
            return Err(refuse(
                &w.name,
                "FP8 needs exactly one of weight_scale_inv (block) or weight_scale (channel or \
                 tensor)"
                    .into(),
            ));
        }
    };
    if !matches!(scale.dtype, Dtype::F32 | Dtype::Bf16) {
        return Err(refuse(
            &scale.name,
            format!("a {} FP8 scale", scale.dtype.name()),
        ));
    }
    Ok(QuantGroup {
        scheme,
        module: module.to_string(),
        weight: w.name.clone(),
        scale: name_of(scale),
        global: None,
        input,
        rows,
        cols,
    })
}

#[cfg(test)]
#[path = "scheme_tests.rs"]
mod scheme_tests;
