// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Synthetic values to stored bytes: plain tensors by dtype, quantized weights by
//! their scheme. Every scale is computed from the values it scales, so scales are valid by
//! construction; codes are encoded against the stored (rounded) scale, as a real quantizer does.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - One codec: NVFP4 goes through `metrale_core::numeric::quantize_to_nvfp4`, FP8 through
//!   `f32_to_fp8_e4m3_rne`, BF16 through `f32_to_bf16_rne` (no environment escape hatch).
//! - No NaN or infinity is ever written: values are bounded, and a zero scale encodes zero
//!   codes.

use metrale_core::numeric::{f32_to_bf16_rne, f32_to_fp8_e4m3_rne, quantize_to_nvfp4};

use crate::error::{MlError, Result};
use crate::index::Dtype;
use crate::scheme::{Nvfp4Global, QuantGroup, Scheme};
use crate::values::ACT_AMAX;

/// 2026-10-03: The largest finite E4M3 magnitude.
const E4M3_MAX: f32 = 448.0;
/// 2026-10-03: The largest E2M1 magnitude times the largest E4M3: the NVFP4 global scale's
/// denominator.
const NVFP4_RANGE: f32 = 6.0 * 448.0;

/// 2026-10-03: `values` stored as `dtype`.
pub fn encode(name: &str, values: &[f32], dtype: Dtype) -> Result<Vec<u8>> {
    match dtype {
        Dtype::Bf16 => Ok(values
            .iter()
            .flat_map(|&v| f32_to_bf16_rne(v).to_le_bytes())
            .collect()),
        Dtype::F32 => Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        other => Err(MlError::Tensor {
            name: name.to_string(),
            why: format!("plain values cannot be stored as {}", other.name()),
        }),
    }
}

/// 2026-10-03: The value a scale decodes to once stored as `dtype` (BF16 rounds it).
fn stored(v: f32, dtype: Dtype) -> f32 {
    match dtype {
        Dtype::Bf16 => f32::from_bits((f32_to_bf16_rne(v) as u32) << 16),
        _ => v,
    }
}

fn amax(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

fn fp8_codes(values: &[f32], scale: f32) -> impl Iterator<Item = u8> + '_ {
    values.iter().map(move |&v| {
        if scale > 0.0 {
            f32_to_fp8_e4m3_rne(v / scale)
        } else {
            0
        }
    })
}

/// 2026-10-03: The dtypes of a group's tensors, as the source stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupDtypes {
    /// 2026-10-03: The weight's (packed) dtype.
    pub weight: Dtype,
    /// 2026-10-03: The scale's dtype.
    pub scale: Dtype,
    /// 2026-10-03: The NVFP4 global scale's dtype.
    pub global: Option<Dtype>,
    /// 2026-10-03: The input scale's dtype, when there is one.
    pub input: Option<Dtype>,
}

/// 2026-10-03: The bytes of every tensor of `g`, in [`QuantGroup::tensors`] order, from its
/// row-major `rows x cols` values.
pub fn quantize_group(g: &QuantGroup, values: &[f32], dt: GroupDtypes) -> Result<Vec<Vec<u8>>> {
    let (rows, cols) = (g.rows as usize, g.cols as usize);
    let mut out = Vec::with_capacity(4);
    match g.scheme {
        Scheme::Nvfp4(global) => {
            let blob =
                quantize_to_nvfp4(&g.weight, values, rows, cols).map_err(|e| MlError::Tensor {
                    name: g.weight.clone(),
                    why: format!("{e:#}"),
                })?;
            out.push(blob.packed);
            out.push(blob.scales);
            let gv = match global {
                Nvfp4Global::ModelOpt => blob.scale_2,
                Nvfp4Global::CompressedTensors => 1.0 / blob.scale_2,
            };
            let gd = dt.global.ok_or_else(|| MlError::Tensor {
                name: g.weight.clone(),
                why: "an NVFP4 weight without a global scale dtype".into(),
            })?;
            out.push(encode(&g.weight, &[gv], gd)?);
            if let Some(d) = dt.input {
                let s = match global {
                    Nvfp4Global::ModelOpt => ACT_AMAX / NVFP4_RANGE,
                    Nvfp4Global::CompressedTensors => NVFP4_RANGE / ACT_AMAX,
                };
                out.push(encode(&g.weight, &[s], d)?);
            }
        }
        Scheme::Fp8Block { bn, bk } => {
            let (bn, bk) = (bn as usize, bk as usize);
            let (gr, gc) = (rows.div_ceil(bn), cols.div_ceil(bk));
            let mut scales = vec![0.0f32; gr * gc];
            let mut codes = vec![0u8; rows * cols];
            let mut block = Vec::with_capacity(bn * bk);
            for br in 0..gr {
                for bc in 0..gc {
                    block.clear();
                    for r in br * bn..((br + 1) * bn).min(rows) {
                        block.extend_from_slice(
                            &values[r * cols + bc * bk..r * cols + ((bc + 1) * bk).min(cols)],
                        );
                    }
                    let s = stored(amax(&block) / E4M3_MAX, dt.scale);
                    scales[br * gc + bc] = s;
                    for r in br * bn..((br + 1) * bn).min(rows) {
                        let span = r * cols + bc * bk..r * cols + ((bc + 1) * bk).min(cols);
                        for (c, code) in codes[span.clone()]
                            .iter_mut()
                            .zip(fp8_codes(&values[span], s))
                        {
                            *c = code;
                        }
                    }
                }
            }
            out.push(codes);
            out.push(encode(&g.scale, &scales, dt.scale)?);
        }
        Scheme::Fp8Channel => {
            let mut scales = Vec::with_capacity(rows);
            let mut codes = Vec::with_capacity(rows * cols);
            for row in values.chunks(cols) {
                let s = stored(amax(row) / E4M3_MAX, dt.scale);
                scales.push(s);
                codes.extend(fp8_codes(row, s));
            }
            out.push(codes);
            out.push(encode(&g.scale, &scales, dt.scale)?);
        }
        Scheme::Fp8Tensor => {
            let s = stored(amax(values) / E4M3_MAX, dt.scale);
            out.push(fp8_codes(values, s).collect());
            out.push(encode(&g.scale, &[s], dt.scale)?);
        }
    }
    if let (false, Some(d)) = (matches!(g.scheme, Scheme::Nvfp4(_)), dt.input) {
        out.push(encode(&g.weight, &[ACT_AMAX / E4M3_MAX], d)?);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "synth_tests.rs"]
mod synth_tests;
