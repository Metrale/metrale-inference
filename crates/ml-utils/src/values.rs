// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: What values a synthetic tensor holds, by tensor class, and their generation.
//!
//! | class | rule (first match) | values |
//! |---|---|---|
//! | norm weight / bias | the module's last segment contains `norm` | 1 / 0 |
//! | bias | last segment `bias` | 0 |
//! | `A_log` | last segment `A_log` | `ln(1..=16)`, uniformly chosen |
//! | `dt_bias` | last segment `dt_bias` | inverse softplus of dt in [0.001, 0.1] |
//! | KV scale | `k_scale`, `v_scale` | `ACT_AMAX / 448` |
//! | embedding | `embed_tokens.weight` | unit normal (bias channel: a constant column) |
//! | linear | rank >= 2 | normal, std `gain / sqrt(fan_in)` |
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Norm weights are 1 whatever the norm's weight form: `x * w` and `x * (1 + w)` then both
//!   scale by a constant, so no checkpoint's norm is degenerate.
//! - Residual writers (`o_proj`, `out_proj`, `down_proj` in a main layer) use
//!   [`residual_gain`], so the residual stream stays dominated by the embedding and the
//!   activations do not grow across layers.
//! - Any tensor no rule classifies is refused by name.

use crate::error::{MlError, Result};
use crate::index::TensorEntry;
use crate::rng::Stream;

/// 2026-10-03: The activation magnitude static scales assume: an RMS-normalized activation
/// stays within +-8, so `input_scale` and KV scales cover it without clipping.
pub const ACT_AMAX: f32 = 8.0;

/// 2026-10-03: `ln(k)` for k = 1..=16, written out (no runtime logarithm, so every platform
/// draws the same bits): `A = -exp(A_log)` spans the reference initialisation range [1, 16].
const A_LOG: [f32; 16] = [
    0.0,
    std::f32::consts::LN_2,
    1.098_612_3,
    1.386_294_4,
    1.609_438,
    1.791_759_5,
    1.945_910_1,
    2.079_441_5,
    2.197_224_5,
    std::f32::consts::LN_10,
    2.397_895_3,
    2.484_906_6,
    2.564_949_4,
    2.639_057_3,
    2.708_050_2,
    2.772_588_7,
];

/// 2026-10-03: `ln(exp(dt) - 1)` for dt in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1},
/// the reference initialisation's dt range, written out.
const DT_BIAS: [f32; 8] = [
    -6.907_255,
    -6.213_608,
    -5.295_816_4,
    -4.600_166,
    -3.902_006_4,
    -2.970_628,
    -2.624_056,
    -2.252_168_5,
];

/// 2026-10-03: The values of one tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueClass {
    /// 2026-10-03: All ones.
    Ones,
    /// 2026-10-03: All zeros.
    Zeros,
    /// 2026-10-03: GatedDeltaNet `A_log`.
    ALog,
    /// 2026-10-03: GatedDeltaNet `dt_bias`.
    DtBias,
    /// 2026-10-03: A KV-cache scale.
    KvScale,
    /// 2026-10-03: `rows x row_len` normal values of standard deviation `std`; row `zero_row`
    /// (the bias channel's row of a residual writer) is zero.
    Normal {
        /// 2026-10-03: Standard deviation.
        std: f32,
        /// 2026-10-03: Elements per row (the last logical dimension's span).
        row_len: u64,
        /// 2026-10-03: A row held at zero.
        zero_row: Option<u64>,
    },
    /// 2026-10-03: Unit normal rows; with a bias channel, column `channel` holds `value`.
    Embedding {
        /// 2026-10-03: Elements per row (hidden).
        row_len: u64,
        /// 2026-10-03: `(channel, value)`.
        bias: Option<(u64, f32)>,
    },
    /// 2026-10-03: A router whose logits reproduce an expert-load profile: normal of std
    /// `sigma` except column `channel`, which holds `column[e]` in row `e`.
    Router {
        /// 2026-10-03: Elements per row (hidden).
        row_len: u64,
        /// 2026-10-03: The bias channel.
        channel: u64,
        /// 2026-10-03: Standard deviation of the other columns.
        sigma: f32,
        /// 2026-10-03: Per expert, the bias column's value.
        column: Vec<f32>,
    },
}

/// 2026-10-03: The gain of residual writers in a model of `layers` layers: `0.1 / sqrt(layers)`.
pub fn residual_gain(layers: usize) -> f32 {
    0.1 / (layers as f32).sqrt()
}

/// 2026-10-03: Whether `module` writes the residual stream.
pub fn is_residual_writer(module: &str) -> bool {
    module
        .rsplit_once('.')
        .is_some_and(|(_, last)| matches!(last, "o_proj" | "out_proj" | "down_proj"))
}

/// 2026-10-03: `(rows, fan_in)` of a weight: fused experts `[E, N, K]` read K; any other rank
/// reads everything after the first dimension (a convolution's `[O, C, ...]`).
pub fn rows_and_fan_in(e: &TensorEntry) -> Result<(u64, u64)> {
    if e.shape.len() < 2 {
        return Err(MlError::Tensor {
            name: e.name.clone(),
            why: format!("rank {} is not a weight", e.shape.len()),
        });
    }
    if e.shape.len() == 3 && e.name.contains(".experts.") {
        return Ok((e.shape[0] * e.shape[1], e.shape[2]));
    }
    let fan_in: u64 = e.shape[1..].iter().product();
    Ok((e.shape[0], fan_in))
}

/// 2026-10-03: The class of a tensor that is neither a quantized group nor a router or
/// embedding (`plan` decides those), with `gain` for a linear weight.
pub fn classify_plain(e: &TensorEntry, gain: f32) -> Result<ValueClass> {
    let (module, last) = e.name.rsplit_once('.').unwrap_or(("", e.name.as_str()));
    let module_last = module.rsplit_once('.').map_or(module, |(_, l)| l);
    if module_last.contains("norm") && last == "weight" {
        return Ok(ValueClass::Ones);
    }
    if last == "bias" {
        return Ok(ValueClass::Zeros);
    }
    match last {
        "A_log" => return Ok(ValueClass::ALog),
        "dt_bias" => return Ok(ValueClass::DtBias),
        "k_scale" | "v_scale" => return Ok(ValueClass::KvScale),
        _ => {}
    }
    if last == "weight" || e.shape.len() >= 2 {
        let (_, fan_in) = rows_and_fan_in(e)?;
        return Ok(ValueClass::Normal {
            std: gain / (fan_in as f32).sqrt(),
            row_len: fan_in,
            zero_row: None,
        });
    }
    Err(MlError::Tensor {
        name: e.name.clone(),
        why: "no value class (not a norm, bias, A_log, dt_bias, KV scale or weight)".into(),
    })
}

/// 2026-10-03: Elements per chunk when generation is split over threads.
const CHUNK: usize = 1 << 20;

/// 2026-10-03: The `n` values of `class` drawn from `stream`. Element `i` depends only on
/// `(stream, i)`, so the split over threads does not change the result.
pub fn fill(class: &ValueClass, stream: Stream, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    if n <= CHUNK || threads == 1 {
        fill_range(class, stream, 0, &mut out);
        return out;
    }
    let per = n.div_ceil(threads).max(CHUNK);
    std::thread::scope(|s| {
        for (k, part) in out.chunks_mut(per).enumerate() {
            s.spawn(move || fill_range(class, stream, k * per, part));
        }
    });
    out
}

fn fill_range(class: &ValueClass, stream: Stream, start: usize, out: &mut [f32]) {
    for (j, v) in out.iter_mut().enumerate() {
        let i = (start + j) as u64;
        *v = match class {
            ValueClass::Ones => 1.0,
            ValueClass::Zeros => 0.0,
            ValueClass::ALog => A_LOG[stream.below(i, 16) as usize],
            ValueClass::DtBias => DT_BIAS[stream.below(i, 8) as usize],
            ValueClass::KvScale => ACT_AMAX / 448.0,
            ValueClass::Normal {
                std,
                row_len,
                zero_row,
            } => {
                if *zero_row == Some(i / row_len) {
                    0.0
                } else {
                    stream.normal4(i) * std
                }
            }
            ValueClass::Embedding { row_len, bias } => match bias {
                Some((c, value)) if i % row_len == *c => *value,
                _ => stream.normal4(i),
            },
            ValueClass::Router {
                row_len,
                channel,
                sigma,
                column,
            } => {
                if i % row_len == *channel {
                    column[(i / row_len) as usize]
                } else {
                    stream.normal4(i) * sigma
                }
            }
        };
    }
}

#[cfg(test)]
#[path = "values_tests.rs"]
mod values_tests;
