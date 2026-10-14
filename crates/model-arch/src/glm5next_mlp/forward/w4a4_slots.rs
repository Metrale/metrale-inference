// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The launch of the W4A4 slot GEMV (`w4a4_gemv_mx8_moe_slots` and its `_k64` twin,
//! `kernels/gb10/common/w4a4_gemv_mx_moe.cu`), which puts one slot on each `blockIdx.y`. CUDA
//! caps grid Y at 65535, so 8192 rows x top 8 cannot go in one launch; the slots are split into
//! pieces of at most that many, each launched on pointers advanced to its first slot. Up to
//! 65535 slots it is the one launch it always was, on the same arguments.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every slot is in exactly one piece, in order; each piece is at most [`CUDA_MAX_GRID_Y`]
//!   slots and starts at a multiple of `act_div`, so a slot's activation row inside its piece
//!   is its global row minus the piece's first row.
//! - Nothing launches before every check passes.

use std::ops::Range;

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::w4a4::{W4A4_BLOCK, W4A4_ROWS_PER_CTA};

/// 2026-10-14: The largest grid Y (and Z) of a CUDA launch.
pub const CUDA_MAX_GRID_Y: usize = 65_535;

/// 2026-10-14: One slot-GEMV call: `out[s, n]` for every slot `s < slots` from activation row
/// `s / act_div` of the quantized `aq`/`as_`/`ag` and the weights of expert `ids[s]`.
#[derive(Clone, Copy, Debug)]
pub struct W4a4SlotsCall {
    pub aq: DevicePtr,
    pub as_: DevicePtr,
    pub ag: DevicePtr,
    pub ids: DevicePtr,
    pub packed_ptrs: DevicePtr,
    pub scale_ptrs: DevicePtr,
    pub scale2_vals: DevicePtr,
    pub out: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub slots: usize,
    pub act_div: usize,
    pub num_experts: usize,
}

/// 2026-10-14: The slot ranges of the launches covering `slots` slots read with row divisor
/// `act_div`: one range up to [`CUDA_MAX_GRID_Y`] slots, else ranges of the largest multiple of
/// `act_div` that fits, the last one ragged.
pub fn slot_pieces(slots: usize, act_div: usize) -> Result<Vec<Range<usize>>> {
    ensure!(
        slots >= 1 && (1..=CUDA_MAX_GRID_Y).contains(&act_div),
        "W4A4 slot GEMV: {slots} slots with row divisor {act_div} (need >= 1 slot and a divisor \
         in 1..={CUDA_MAX_GRID_Y})"
    );
    let step = if slots <= CUDA_MAX_GRID_Y {
        slots
    } else {
        CUDA_MAX_GRID_Y - CUDA_MAX_GRID_Y % act_div
    };
    Ok((0..slots)
        .step_by(step)
        .map(|s0| s0..(s0 + step).min(slots))
        .collect())
}

/// 2026-10-14: Launch `kern` over every slot of `c`, one launch per [`slot_pieces`] range. A later
/// piece's activation pointers advance by whole rows of the width both entries store, `K` rounded
/// up to 128 (`ka` in the kernel: `K` itself for the plain entry, whose K is a multiple of 128).
/// K must be a positive multiple of 64, the wider of the two entries' contracts.
pub fn launch_w4a4_slots(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    c: &W4a4SlotsCall,
    stream: u64,
) -> Result<()> {
    ensure!(
        c.k >= 64 && c.k.is_multiple_of(64),
        "W4A4 slot GEMV: K = {} is not a positive multiple of 64",
        c.k
    );
    let ka = c.k.next_multiple_of(128);
    for p in slot_pieces(c.slots, c.act_div)? {
        let row0 = p.start / c.act_div;
        KernelLaunch::new(gpu, kern)
            .grid([div_ceil(c.n as u32, W4A4_ROWS_PER_CTA), p.len() as u32, 1])
            .block([W4A4_BLOCK, 1, 1])
            .arg_ptr(c.aq.offset(row0 * (ka / 2)))
            .arg_ptr(c.as_.offset(row0 * (ka / 16)))
            .arg_ptr(c.ag.offset(row0 * 4))
            .arg_ptr(c.ids.offset(p.start * 4))
            .arg_ptr(c.packed_ptrs)
            .arg_ptr(c.scale_ptrs)
            .arg_ptr(c.scale2_vals)
            .arg_ptr(c.out.offset(p.start * c.n * 2))
            .arg_u32(c.n as u32)
            .arg_u32(c.k as u32)
            .arg_u32(c.act_div as u32)
            .arg_u32(c.num_experts as u32)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "w4a4_slots_tests.rs"]
mod tests;
