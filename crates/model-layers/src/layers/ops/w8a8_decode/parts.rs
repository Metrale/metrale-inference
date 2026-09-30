// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The parts of a W8A8 projection the circuit executor launches one group at a
//! time: a stacked weight's segments, the kernel each launch takes, and a scratch view over
//! one activation buffer. The launches themselves stay [`super::w8a8_act_quant`],
//! [`super::w8a8_act_quant_silu`] and [`super::w8a8_gemv`].
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`W8a8Kernels::gemv_entry`] is the entry [`super::w8a8_gemv`] launches for a launch of
//!   that many rows; [`W8a8Kernels::quant`] and [`W8a8Kernels::quant_silu`] are the ones the
//!   quantizers launch.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::{W8A8_LAUNCH_ROWS, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight, entry_index};

impl W8a8Weight {
    /// 2026-09-30: Segment `i` as a weight of its own. A segment's output rows read only its
    /// own weight rows and scales, so its launch writes the bits the stacked launch writes
    /// for those rows.
    pub fn segment(&self, i: usize) -> Result<W8a8Weight> {
        ensure!(i < self.count, "W8a8Weight: segment {i} of {}", self.count);
        W8a8Weight::new(&[self.segs[i]])
    }

    /// 2026-09-30: Stacked segments.
    pub fn segments(&self) -> usize {
        self.count
    }
}

impl W8a8Kernels {
    /// 2026-09-30: The quantizer of `scale`'s layout.
    pub fn quant(&self, scale: W8a8Scale) -> KernelHandle {
        match scale {
            W8a8Scale::PerRow => self.quant_row,
            W8a8Scale::Block128 => self.quant_g128,
        }
    }

    /// 2026-09-30: The SiLU·mul quantizer of `scale`'s layout.
    pub fn quant_silu(&self, scale: W8a8Scale) -> KernelHandle {
        match scale {
            W8a8Scale::PerRow => self.quant_silu_row,
            W8a8Scale::Block128 => self.quant_silu_g128,
        }
    }

    /// 2026-09-30: The GEMV entry of one launch of `rows` rows; `None` above one launch.
    pub fn gemv_entry(&self, scale: W8a8Scale, rows: usize) -> Option<KernelHandle> {
        if !(1..=W8A8_LAUNCH_ROWS).contains(&rows) {
            return None;
        }
        let e = entry_index(rows);
        Some(match scale {
            W8a8Scale::PerRow => self.rowscale[e],
            W8a8Scale::Block128 => self.blk128[e],
        })
    }
}

impl W8a8Scratch {
    /// 2026-09-30: One activation of `rows` rows packed from `base`: `[rows, k]` E4M3, then
    /// its FP32 scales (the circuit's `fp8/token` edge, `Format::bytes`).
    pub fn packed(base: DevicePtr, rows: usize, k: u32, scale: W8a8Scale) -> Self {
        let q_bytes = rows * k as usize;
        Self {
            q: base,
            q_bytes,
            scale: base.offset(q_bytes),
            scale_bytes: rows * scale.act_scales_per_row(k) * 4,
        }
    }
}
