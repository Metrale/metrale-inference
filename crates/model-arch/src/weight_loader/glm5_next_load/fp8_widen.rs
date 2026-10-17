// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Block-scaled E4M3 weights widened to BF16 on the host, for checkpoints that store
//! their attention, dense-MLP, shared-expert and head linears in 8 bits.
//!
//! Two layouts are read, told apart by the `.weight_scale` sibling of an F8_E4M3 `.weight`
//! `[n, k]`:
//! - MX (`MXFP8`): one UE8M0 byte per 32 weights of a row, `[n, k/32]` (U8 or F8_E8M0).
//!   `value = e4m3(w) * 2^(e - 127)`; `e = 255` is NaN in the MX encoding and is refused.
//! - Block-128 (`FP8_PB_WO`): one F32 per 128x128 tile, `[ceil(n/128), 1, ceil(k/128), 1]` or
//!   `[ceil(n/128), ceil(k/128)]`. `value = e4m3(w) * s`.
//!
//! Both products are formed in `f32` (exact: an E4M3 significand has 4 bits and a power of two
//! or one `f32` multiply rounds once) and rounded to nearest even into BF16 by
//! `half::bf16::from_f32`. An MX product whose magnitude is a normal BF16 is therefore exact; a
//! block-128 product carries the one rounding of the multiply plus the BF16 rounding.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Any other scale dtype or shape is refused; nothing is guessed.
//! - A NaN weight byte (`0x7F`/`0xFF`), a NaN scale or a non-finite product is refused.

use anyhow::{Result, bail};
use metrale_cache::kv_dequant::e4m3_lut;
use metrale_model_weights::weights::WeightDtype;

/// 2026-10-17: MX block width along a row.
pub(super) const MX_BLOCK: usize = 32;
/// 2026-10-17: Block-128 tile edge.
pub(super) const TILE: usize = 128;
/// 2026-10-17: The UE8M0 NaN encoding.
const E8M0_NAN: u8 = 0xFF;

/// 2026-10-17: `2^(e - 127)` as an exact `f32` (`e = 0` is the subnormal `2^-127`).
fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        0 => f32::from_bits(1 << 22),
        _ => f32::from_bits(u32::from(e) << 23),
    }
}

/// 2026-10-17: The per-weight scale of one stored layout, row `r`, column `c`.
enum Scales<'a> {
    Mx { bytes: &'a [u8], per_row: usize },
    Tile { values: Vec<f32>, per_row: usize },
}

impl Scales<'_> {
    fn at(&self, r: usize, c: usize) -> f32 {
        match self {
            Scales::Mx { bytes, per_row } => e8m0_to_f32(bytes[r * per_row + c / MX_BLOCK]),
            Scales::Tile { values, per_row } => values[(r / TILE) * per_row + c / TILE],
        }
    }
}

/// 2026-10-17: Classify the `.weight_scale` of an E4M3 `[n, k]` weight and check its length.
fn classify<'a>(
    what: &str,
    n: usize,
    k: usize,
    dtype: WeightDtype,
    shape: &[usize],
    bytes: &'a [u8],
) -> Result<Scales<'a>> {
    let (tn, tk) = (n.div_ceil(TILE), k.div_ceil(TILE));
    match dtype {
        WeightDtype::UInt8 | WeightDtype::FP8E8M0 => {
            if !k.is_multiple_of(MX_BLOCK) || shape != [n, k / MX_BLOCK] {
                bail!(
                    "{what}: UE8M0 scale {shape:?} does not fit an MX weight [{n}, {k}] \
                     (expected [{n}, {}])",
                    k / MX_BLOCK
                );
            }
            if bytes.len() != n * (k / MX_BLOCK) {
                bail!("{what}: {} B of UE8M0 scales for {shape:?}", bytes.len());
            }
            if let Some(i) = bytes.iter().position(|&e| e == E8M0_NAN) {
                bail!("{what}: UE8M0 scale {i} is the NaN encoding 0xFF");
            }
            Ok(Scales::Mx {
                bytes,
                per_row: k / MX_BLOCK,
            })
        }
        WeightDtype::FP32 => {
            if shape != [tn, 1, tk, 1] && shape != [tn, tk] {
                bail!(
                    "{what}: F32 scale {shape:?} is not one per {TILE}x{TILE} tile of [{n}, {k}] \
                     (expected [{tn}, 1, {tk}, 1] or [{tn}, {tk}])"
                );
            }
            if bytes.len() != tn * tk * 4 {
                bail!("{what}: {} B of F32 tile scales for {shape:?}", bytes.len());
            }
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if let Some(i) = values.iter().position(|v| v.is_nan()) {
                bail!("{what}: tile scale {i} is NaN");
            }
            Ok(Scales::Tile {
                values,
                per_row: tk,
            })
        }
        other => bail!(
            "{what}: an F8_E4M3 weight with a {other:?} {shape:?} scale is neither MX (UE8M0 \
             per {MX_BLOCK}) nor block-{TILE} (F32 per tile)"
        ),
    }
}

/// 2026-10-17: The row-major BF16 bytes of the F8_E4M3 weight `[n, k]` (`weight`, `shape`)
/// scaled by its `.weight_scale` (`scale_dtype`, `scale_shape`, `scale`).
pub(super) fn widen_fp8_to_bf16(
    what: &str,
    weight: &[u8],
    shape: &[usize],
    scale_dtype: WeightDtype,
    scale_shape: &[usize],
    scale: &[u8],
) -> Result<Vec<u8>> {
    let [n, k] = shape[..] else {
        bail!("{what}: an F8_E4M3 weight must be 2-D, got {shape:?}");
    };
    if weight.len() != n * k {
        bail!("{what}: {} B of E4M3 for [{n}, {k}]", weight.len());
    }
    let scales = classify(what, n, k, scale_dtype, scale_shape, scale)?;
    let lut = e4m3_lut();
    let mut out = Vec::with_capacity(n * k * 2);
    for r in 0..n {
        for c in 0..k {
            let w = weight[r * k + c];
            let v = lut[usize::from(w)] * scales.at(r, c);
            if !v.is_finite() {
                bail!("{what}: weight [{r}, {c}] (byte {w:#04x}) widens to {v}");
            }
            out.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }
    }
    Ok(out)
}

/// 2026-10-17: One layer's host tensors under layer-relative names (`LayerSource`'s map).
pub(super) type HostTensors =
    std::collections::BTreeMap<String, (WeightDtype, Vec<usize>, Vec<u8>)>;

/// 2026-10-17: Replace every F8_E4M3 `.weight` of `tensors` that has a `.weight_scale` sibling
/// by its BF16 widening, and remove the sibling. Returns the removed scale names, which the
/// caller drops from its own lists. A layer without such a pair (every 16-bit or NVFP4 layer)
/// is left untouched, byte for byte.
pub(super) fn widen_layer_fp8(tensors: &mut HostTensors) -> Result<Vec<String>> {
    let pairs: Vec<(String, String)> = tensors
        .iter()
        .filter(|(rel, (dtype, ..))| *dtype == WeightDtype::FP8E4M3 && rel.ends_with(".weight"))
        .map(|(rel, _)| (rel.clone(), format!("{rel}_scale")))
        .filter(|(_, scale)| tensors.contains_key(scale))
        .collect();
    let mut consumed = Vec::with_capacity(pairs.len());
    for (rel, scale) in pairs {
        let (Some((sd, sshape, sbytes)), Some(entry)) = (tensors.remove(&scale), tensors.get(&rel))
        else {
            bail!("{rel}: its pair vanished while widening");
        };
        let bf16 = widen_fp8_to_bf16(&rel, &entry.2, &entry.1, sd, &sshape, &sbytes)?;
        let shape = entry.1.clone();
        tensors.insert(rel, (WeightDtype::BF16, shape, bf16));
        consumed.push(scale);
    }
    Ok(consumed)
}

#[cfg(test)]
#[path = "fp8_widen_tests.rs"]
mod tests;
