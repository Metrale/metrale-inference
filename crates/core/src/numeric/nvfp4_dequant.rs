// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Packed NVFP4 to `f32` on the host. 2026-10-03: moved here from the GLM-5.3 loader
//! (`model-arch/src/weight_loader/glm5_next_load/`), which re-exports it, so the loader and
//! `metrale-ml-utils` share one host codec.
//!
//! `value = E2M1[nibble] * e4m3(block_scale) * weight_scale_2`, multiplied in
//! that order, as `kernels/gb10/common/moe_w4a16_grouped_gemm.cu` does. The
//! codebook and the block width are [`NVFP4_E2M1_LUT`] and [`NVFP4_GROUP_SIZE`]; a scale byte
//! decodes through [`FP8_E4M3_LUT`]. Within a byte, the even column is the low nibble and the
//! odd column the high nibble.
//!
//! Owner: metrale-core.
//! Invariants: none beyond the types.

use super::{FP8_E4M3_LUT, NVFP4_E2M1_LUT, NVFP4_GROUP_SIZE};
use anyhow::{Result, bail};

/// 2026-09-25: Packed NVFP4 `[rows, cols/2]`, E4M3 block scales
/// `[rows, cols/16]` and one global `f32` to row-major `f32 [rows, cols]`.
///
/// `packed_shape` is the stored shape, so the logical column count is
/// `2 * packed_shape[1]`. A packed or scale length that does not match it is
/// an error.
pub fn dequant_nvfp4_to_f32(
    what: &str,
    packed: &[u8],
    packed_shape: &[usize],
    scales: &[u8],
    scale_2: f32,
) -> Result<Vec<f32>> {
    let [rows, packed_cols] = packed_shape[..] else {
        bail!("{what}: packed NVFP4 must be 2-D, got shape {packed_shape:?}");
    };
    if packed.len() != rows * packed_cols {
        bail!(
            "{what}: {} B of packed NVFP4 for shape [{rows}, {packed_cols}]",
            packed.len()
        );
    }
    let cols = packed_cols * 2;
    if !cols.is_multiple_of(NVFP4_GROUP_SIZE) {
        bail!(
            "{what}: {cols} logical columns is not a whole number of \
             {NVFP4_GROUP_SIZE}-element NVFP4 blocks"
        );
    }
    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    if scales.len() != rows * groups_per_row {
        bail!(
            "{what}: {} block scales, expected {rows} x {groups_per_row} = {}",
            scales.len(),
            rows * groups_per_row
        );
    }
    let e4m3 = &FP8_E4M3_LUT;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
        let srow = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let orow = &mut out[r * cols..(r + 1) * cols];
        for (g, &sb) in srow.iter().enumerate() {
            // 2026-09-25: Every byte indexes the 256-entry E4M3 table, where
            // `0x7F` is NaN and `0x80..` are negative. A block scale must be
            // finite and non-negative (`0x00..=0x7E`), so any other byte is an
            // error rather than a NaN or sign-flipped block.
            if sb >= 0x7F {
                bail!(
                    "{what}: block scale byte 0x{sb:02X} at row {r}, block {g} is not a \
                     finite non-negative E4M3 value (0x7F is NaN, 0x80.. are negative). \
                     The `.weight_scale` sibling is not ModelOpt E4M3 block scales — check \
                     that it is F8_E4M3 of shape [{rows}, {groups_per_row}] and not, say, a \
                     compressed-tensors `weight_global_scale`."
                );
            }
            // 2026-09-25: `(E2M1 * scale) * scale_2`, in the kernel's order;
            // folding `scale * scale_2` first rounds differently in f32.
            let s = e4m3[sb as usize];
            let base = g * NVFP4_GROUP_SIZE;
            for i in 0..NVFP4_GROUP_SIZE {
                let c = base + i;
                let byte = prow[c / 2];
                let nibble = if c.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                orow[c] = NVFP4_E2M1_LUT[nibble as usize] * s * scale_2;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "nvfp4_dequant_tests.rs"]
mod tests;
