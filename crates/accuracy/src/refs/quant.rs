// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Quantized operand layouts, read from the circuit's `Format`: element encoding,
//! scale granularity, and the host quantizers that turn generated real values into valid
//! stored operands (what a checkpoint or an activation quantizer would hold). The quantizer
//! only has to produce valid, realistically distributed codes: references decode the stored
//! bytes exactly, so its own rounding choices are not part of any bound.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Every produced code decodes to a finite value; scales are positive and finite.

use metrale_circuit::format::{Format, Scale};

use crate::case::{Enc, Tensor};
use crate::elem::{E2M1, E4M3, F32, UE4M3};

/// 2026-10-09: How one operand of a projection is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// 2026-10-09: bf16 values, no scale.
    Bf16,
    /// 2026-10-09: f32 values, no scale.
    F32,
    /// 2026-10-09: E4M3 values with one f32 scale per row of the stored matrix (per token for
    /// an activation, per output channel for a weight).
    Fp8Row,
    /// 2026-10-09: E4M3 values with one f32 scale for the whole tensor.
    Fp8Tensor,
    /// 2026-10-09: E4M3 values with one f32 scale per `g` consecutive K values of a row.
    Fp8Group(usize),
    /// 2026-10-09: E4M3 values with one f32 scale per `[r, c]` block (weights).
    Fp8Block(usize, usize),
    /// 2026-10-09: E2M1 values, one UE4M3 scale per `g` K values, one f32 global per tensor
    /// (weight) or per row (activation).
    Nvfp4(usize),
}

impl Layout {
    /// 2026-10-09: The layout of a stored format; `None` for formats no projection operand has.
    pub fn of(f: Format) -> Option<Layout> {
        Some(match f {
            Format::Bf16 => Layout::Bf16,
            Format::F32 => Layout::F32,
            Format::I32 => return None,
            Format::Fp8E4m3 { scale } => match scale {
                Scale::PerToken | Scale::PerChannel => Layout::Fp8Row,
                Scale::PerTensor => Layout::Fp8Tensor,
                Scale::Group(g) => Layout::Fp8Group(g as usize),
                Scale::Block(r, c) => Layout::Fp8Block(r as usize, c as usize),
            },
            Format::Nvfp4 { group } => Layout::Nvfp4(group as usize),
        })
    }

    /// 2026-10-09: K values sharing one block scale along the reduction, if any.
    pub fn k_group(&self) -> Option<usize> {
        match *self {
            Layout::Fp8Group(g) | Layout::Nvfp4(g) => Some(g),
            Layout::Fp8Block(_, c) => Some(c),
            _ => None,
        }
    }
}

/// 2026-10-09: A stored operand: values, block scales `[rows_b, groups]`, and the per-row and
/// per-tensor f32 scales (each 1.0 where the layout has none).
#[derive(Debug, Clone, PartialEq)]
pub struct Stored {
    /// 2026-10-09: The value tensor `[rows, k]`.
    pub values: Tensor,
    /// 2026-10-09: Block scales, row-major `[ceil(rows/block_rows), k/group]`.
    pub block: Option<Tensor>,
    /// 2026-10-09: Rows of stored values sharing one block-scale row (1 except weight blocks).
    pub block_rows: usize,
    /// 2026-10-09: Per-row f32 scales `[rows]` (token, channel, or NVFP4 activation globals).
    pub row: Option<Tensor>,
    /// 2026-10-09: The per-tensor f32 scale (NVFP4 weight global, FP8 per-tensor).
    pub global: f64,
}

fn f32r(x: f64) -> f64 {
    F32.round(x).expect("a finite scale")
}

/// 2026-10-09: Quantize real values `x` (`[rows, k]`, row-major) into `layout`. `per_row_global`
/// selects one NVFP4 global per row (activations) instead of one per tensor (weights).
pub fn quantize(
    layout: Layout,
    x: &[f64],
    rows: usize,
    k: usize,
    per_row_global: bool,
) -> Result<Stored, String> {
    let amax = |s: &[f64]| s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let nonzero = |a: f64| if a > 0.0 { a } else { 1.0 };
    let plain = |enc: Enc| -> Result<Stored, String> {
        Ok(Stored {
            values: Tensor::encode(enc, vec![rows, k], x)?,
            block: None,
            block_rows: 1,
            row: None,
            global: 1.0,
        })
    };
    match layout {
        Layout::Bf16 => plain(Enc::Bf16),
        Layout::F32 => plain(Enc::F32),
        Layout::Fp8Row => {
            let scales: Vec<f64> = x
                .chunks(k)
                .map(|r| f32r(nonzero(amax(r)) / 448.0))
                .collect();
            let q: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, v)| E4M3.round_saturating(v / scales[i / k]).unwrap_or(0.0))
                .collect();
            Ok(Stored {
                values: Tensor::encode(Enc::E4m3, vec![rows, k], &q)?,
                block: None,
                block_rows: 1,
                row: Some(Tensor::encode(Enc::F32, vec![rows], &scales)?),
                global: 1.0,
            })
        }
        Layout::Fp8Tensor => {
            let s = f32r(nonzero(amax(x)) / 448.0);
            let q: Vec<f64> = x
                .iter()
                .map(|v| E4M3.round_saturating(v / s).unwrap_or(0.0))
                .collect();
            Ok(Stored {
                values: Tensor::encode(Enc::E4m3, vec![rows, k], &q)?,
                block: None,
                block_rows: 1,
                row: None,
                global: s,
            })
        }
        Layout::Fp8Group(g) | Layout::Fp8Block(_, g) => {
            if k % g != 0 {
                return Err(format!("k={k} is not a multiple of the scale group {g}"));
            }
            let br = if let Layout::Fp8Block(r, _) = layout {
                r
            } else {
                1
            };
            let (gk, gr) = (k / g, rows.div_ceil(br));
            let mut scales = vec![0.0; gr * gk];
            for (bi, s) in scales.iter_mut().enumerate() {
                let (rb, kb) = (bi / gk, bi % gk);
                let mut m = 0.0f64;
                for r in rb * br..((rb + 1) * br).min(rows) {
                    m = m.max(amax(&x[r * k + kb * g..r * k + (kb + 1) * g]));
                }
                *s = f32r(nonzero(m) / 448.0);
            }
            let q: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let (r, c) = (i / k, i % k);
                    E4M3.round_saturating(v / scales[(r / br) * gk + c / g])
                        .unwrap_or(0.0)
                })
                .collect();
            Ok(Stored {
                values: Tensor::encode(Enc::E4m3, vec![rows, k], &q)?,
                block: Some(Tensor::encode(Enc::F32, vec![gr, gk], &scales)?),
                block_rows: br,
                row: None,
                global: 1.0,
            })
        }
        Layout::Nvfp4(g) => quantize_nvfp4(x, rows, k, g, per_row_global),
    }
}

fn quantize_nvfp4(
    x: &[f64],
    rows: usize,
    k: usize,
    g: usize,
    per_row_global: bool,
) -> Result<Stored, String> {
    if k % g != 0 {
        return Err(format!("k={k} is not a multiple of the NVFP4 group {g}"));
    }
    let gk = k / g;
    let amax = |s: &[f64]| s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // 2026-10-09: The usual two-level recipe: the global maps the largest group amax to the top
    // of E4M3 x E2M1 (448 * 6); each block scale is its group's amax / 6 over that global.
    let global_of = |s: &[f64]| {
        let a = amax(s);
        f32r(if a > 0.0 { a / (448.0 * 6.0) } else { 1.0 })
    };
    let globals: Vec<f64> = if per_row_global {
        x.chunks(k).map(global_of).collect()
    } else {
        vec![global_of(x); rows]
    };
    let mut scales = vec![0.0; rows * gk];
    let mut q = vec![0.0; rows * k];
    for r in 0..rows {
        for b in 0..gk {
            let grp = &x[r * k + b * g..r * k + (b + 1) * g];
            let s = UE4M3
                .round_saturating(amax(grp) / 6.0 / globals[r])
                .unwrap_or(0.0);
            scales[r * gk + b] = s;
            let unit = s * globals[r];
            for (j, v) in grp.iter().enumerate() {
                q[r * k + b * g + j] = if unit > 0.0 {
                    E2M1.round_saturating(v / unit).unwrap_or(0.0)
                } else {
                    0.0
                };
            }
        }
    }
    Ok(Stored {
        values: Tensor::encode(Enc::E2m1x2, vec![rows, k], &q)?,
        block: Some(Tensor::encode(Enc::Ue4m3, vec![rows, gk], &scales)?),
        block_rows: 1,
        row: if per_row_global {
            Some(Tensor::encode(Enc::F32, vec![rows], &globals)?)
        } else {
            None
        },
        global: if per_row_global { 1.0 } else { globals[0] },
    })
}

impl Stored {
    /// 2026-10-09: The block scale covering element `(r, c)` (1.0 without block scales).
    pub fn block_scale(&self, r: usize, c: usize, group: usize) -> f64 {
        match &self.block {
            None => 1.0,
            Some(t) => t.get((r / self.block_rows) * t.dims[1] + c / group),
        }
    }

    /// 2026-10-09: The per-row scale of row `r` (1.0 without).
    pub fn row_scale(&self, r: usize) -> f64 {
        self.row.as_ref().map_or(1.0, |t| t.get(r))
    }
}
