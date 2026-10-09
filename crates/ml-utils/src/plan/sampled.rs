// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Units under `values.mode = "stats"`: every kept tensor, quantized or not, has its
//! stored bit patterns sampled from its class's statistics (`stats.rs`), so weights, codes and
//! scales carry the real model's bit-pattern distributions. Histogram routing then writes the
//! bias channel over the sample: the embedding's channel `c` holds `K`, each main-layer router's
//! column `c` holds the fitted bias, each main-layer residual writer's row `c` is zero, and each
//! main-layer pre-FFN norm's entry `c` is its class RMS.
//!
//! `K` is `sqrt(hidden) / 2` times the embedding's RMS, as in the analytic mode. A router column
//! is the unit-noise bias times twice the router's RMS (the noise a token's other channels give,
//! for an embedding-dominated residual), times the layer's calibration gain: real-value weights
//! make the residual depend on context, so the true noise is measured on a first mock
//! (`met ml-utils calibrate-routing`) and corrected.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - No shape, dtype or format changes; only values, and only through an [`Edit`].
//! - An edit that does not fit its tensor (a column of a non-BF16 tensor, a row past the end) is
//!   refused.

use std::collections::BTreeMap;

use metrale_core::numeric::f32_to_bf16_rne;

use super::units::{Built, Ctx, fit_router, is_router, routing};
use super::{OutTensor, Unit};
use crate::error::{MlError, Result};
use crate::index::Dtype;
use crate::routing::BiasChannel;
use crate::stats::{Sampler, ValueStats, class_key};
use crate::values::is_residual_writer;

/// 2026-10-04: The norm in front of the router in the layouts M1 covers. Its weight at the bias
/// channel scales the bias; a sampled value there could be near zero, so it is set to the class
/// RMS, a typical magnitude, and the calibration gain absorbs what remains.
const ROUTER_NORM: &str = "post_attention_layernorm.weight";

/// 2026-10-04: What is written over a sampled tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum Edit {
    /// 2026-10-04: Nothing.
    None,
    /// 2026-10-04: Row `row` (of `row_bytes` stored bytes) is zero.
    ZeroRow {
        /// 2026-10-04: The row.
        row: u64,
        /// 2026-10-04: Stored bytes per row.
        row_bytes: u64,
    },
    /// 2026-10-04: BF16 column `col` of rows of `row_len` elements holds `values` (one per row,
    /// or one for every row).
    Column {
        /// 2026-10-04: The column.
        col: u64,
        /// 2026-10-04: Elements per row.
        row_len: u64,
        /// 2026-10-04: The values.
        values: Vec<f32>,
    },
}

impl Edit {
    /// 2026-10-04: Apply to a tensor's stored bytes.
    pub fn apply(&self, bytes: &mut [u8]) -> Result<()> {
        let bad = |w: String| MlError::Spec(format!("edit: {w}"));
        match self {
            Edit::None => Ok(()),
            Edit::ZeroRow { row, row_bytes } => {
                let (a, b) = ((row * row_bytes) as usize, ((row + 1) * row_bytes) as usize);
                let len = bytes.len();
                bytes
                    .get_mut(a..b)
                    .ok_or_else(|| bad(format!("row {row} past {len} bytes")))?
                    .fill(0);
                Ok(())
            }
            Edit::Column {
                col,
                row_len,
                values,
            } => {
                let rows = bytes.len() as u64 / (row_len * 2);
                for r in 0..rows {
                    let v = match values.as_slice() {
                        [one] => *one,
                        many => *many
                            .get(r as usize)
                            .ok_or_else(|| bad(format!("{} values for {rows} rows", many.len())))?,
                    };
                    let at = ((r * row_len + col) * 2) as usize;
                    bytes[at..at + 2].copy_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
                }
                Ok(())
            }
        }
    }
}

fn rms_of(stats: &ValueStats, class: &str) -> Result<f32> {
    stats
        .classes
        .get(class)
        .and_then(|c| c.bf16_rms())
        .filter(|r| *r > 0.0)
        .ok_or_else(|| MlError::Spec(format!("value statistics: no BF16 RMS for `{class}`")))
}

/// 2026-10-04: The units, their tensors and the samplers they draw from.
pub(super) fn build(c: &Ctx<'_>, stats: &ValueStats) -> Result<(Built, BTreeMap<String, Sampler>)> {
    let mut r = routing(c)?;
    let mut b = Built {
        tensors: Vec::new(),
        units: Vec::new(),
        routers: Vec::new(),
        bias_channel: None,
    };
    let mut samplers = BTreeMap::new();
    if let Some(rt) = r.as_mut() {
        let emb = c
            .index
            .iter()
            .find(|e| e.name.ends_with("embed_tokens.weight"))
            .ok_or_else(|| MlError::Checkpoint("histogram routing needs an embedding".into()))?;
        let k = (rt.hidden as f32).sqrt() / 2.0
            * rms_of(stats, &class_key(c.schedule, &emb.name, emb.dtype))?;
        rt.channel = BiasChannel { k, ..rt.channel };
        b.bias_channel = Some(rt.channel);
    }
    for e in c.index.iter() {
        let Some(mock) = c.renamer.name(&e.name) else {
            continue;
        };
        let class = class_key(c.schedule, &e.name, e.dtype);
        if !samplers.contains_key(&class) {
            samplers.insert(class.clone(), stats.sampler(&class)?);
        }
        let main = c.schedule.layer_of(&e.name);
        let (module, last) = e.name.rsplit_once('.').unwrap_or(("", e.name.as_str()));
        let edit = match (&r, main) {
            (Some(rt), Some(layer)) if is_router(e) => {
                if e.dtype != Dtype::Bf16 || e.shape != [rt.experts, rt.hidden] {
                    return Err(MlError::Tensor {
                        name: e.name.clone(),
                        why: "histogram routing needs a BF16 [experts, hidden] router".into(),
                    });
                }
                let scale = 2.0 * rms_of(stats, &class)?;
                let values = fit_router(c, rt, e, layer, &mock, &mut b, scale)?;
                Edit::Column {
                    col: rt.channel.channel,
                    row_len: rt.hidden,
                    values,
                }
            }
            (Some(rt), Some(_))
                if is_residual_writer(module) && (last == "weight" || last == "weight_packed") =>
            {
                if e.shape.first() != Some(&rt.hidden) || e.shape.len() != 2 {
                    return Err(MlError::Tensor {
                        name: e.name.clone(),
                        why: format!("a residual writer of shape {:?}, not [hidden, _]", e.shape),
                    });
                }
                Edit::ZeroRow {
                    row: rt.channel.channel,
                    row_bytes: e.shape[1] * e.dtype.bytes(),
                }
            }
            (Some(rt), Some(_)) if e.name.ends_with(ROUTER_NORM) => {
                if e.dtype != Dtype::Bf16 || e.shape != [rt.hidden] {
                    return Err(MlError::Tensor {
                        name: e.name.clone(),
                        why: "the bias channel needs a BF16 [hidden] pre-FFN norm".into(),
                    });
                }
                Edit::Column {
                    col: rt.channel.channel,
                    row_len: rt.hidden,
                    values: vec![rms_of(stats, &class)?],
                }
            }
            (Some(rt), _) if e.name.ends_with("embed_tokens.weight") => {
                if e.dtype != Dtype::Bf16 {
                    return Err(MlError::Tensor {
                        name: e.name.clone(),
                        why: "the bias channel needs a BF16 embedding".into(),
                    });
                }
                Edit::Column {
                    col: rt.channel.channel,
                    row_len: e.shape[1],
                    values: vec![rt.channel.k],
                }
            }
            _ => Edit::None,
        };
        b.tensors.push(OutTensor {
            source: e.name.clone(),
            name: mock,
            dtype: e.dtype,
            shape: e.shape.clone(),
        });
        b.units.push(Unit::Sampled {
            tensor: b.tensors.len() - 1,
            class,
            edit,
        });
    }
    Ok((b, samplers))
}

#[cfg(test)]
#[path = "sampled_tests.rs"]
mod sampled_tests;
