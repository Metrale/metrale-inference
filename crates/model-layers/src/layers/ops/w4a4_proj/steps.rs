// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The two launches of a W4A4 projection, quantize and MX GEMV, over an NVFP4
//! activation the caller places: `proj` runs both on the shared scratch, and the circuit
//! executor runs them as its `act_quant` and projection groups over the activation edge.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`W4a4Proj::gemv`] launches the entry `mx_plan` picks at the process's levers, and
//!   [`W4a4Proj::mx_kernel`] names that same entry, so a caller that checks a planned kernel
//!   against it checks the launch.
//! - A [`W4a4Proj`] exists only for a launch `w4a4_state_for` admits, or (from
//!   [`W4a4Proj::prepared`]) for the quantize alone once `prepare` found the kernels.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{MxLaunch, W4a4State, mx_nt, mx_plan, mx_ps, state, w4a4_state_for};
use crate::weight_map::QuantizedWeight;

/// 2026-09-30: One NVFP4 activation of `m` rows: `[m, k / 2]` E2M1 and `[m, k / 16]` E4M3
/// group scales, both in the GEMV's fragment order, and `[m]` FP32 per-row global scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4ActBuf {
    pub aq: DevicePtr,
    pub a_scale: DevicePtr,
    pub a_gs: DevicePtr,
}

impl Nvfp4ActBuf {
    /// 2026-09-30: The three parts packed in that order from `base` (the circuit's `nvfp4/g16`
    /// edge, `Format::bytes`). Every part stays 8-byte aligned when `k % 128 == 0`.
    pub fn packed(base: DevicePtr, m: u32, k: u32) -> Self {
        let (m, k) = (m as usize, k as usize);
        let a_scale = base.offset(m * k / 2);
        Self {
            aq: base,
            a_scale,
            a_gs: a_scale.offset(m * k / 16),
        }
    }
}

/// 2026-09-30: The W4A4 kernels of one backend (see the module header).
#[derive(Clone, Copy)]
pub struct W4a4Proj(pub(super) W4a4State);

impl W4a4Proj {
    /// 2026-09-30: The kernels for an `m`-row `[n, k]` projection over `weight`, when the tier
    /// admits it there (`nvfp4_proj_small_m` then runs W4A4); `None` when it runs W4A16.
    pub fn for_launch(
        gpu: &dyn GpuBackend,
        weight: &QuantizedWeight,
        m: u32,
        n: u32,
        k: u32,
    ) -> Option<Self> {
        w4a4_state_for(gpu, weight, m, n, k).map(Self)
    }

    /// 2026-09-30: The kernels once `prepare` found them, for the quantize launch alone.
    pub fn prepared(gpu: &dyn GpuBackend) -> Option<Self> {
        state(gpu).map(Self)
    }

    /// 2026-09-30: The shared scratch `proj` quantizes into.
    pub(super) fn scratch(&self) -> Nvfp4ActBuf {
        Nvfp4ActBuf {
            aq: self.0.aq,
            a_scale: self.0.a_scale,
            a_gs: self.0.a_gs,
        }
    }

    /// 2026-09-30: `w4a4_quant_rows`.
    pub fn quant_kernel(&self) -> KernelHandle {
        self.0.quant
    }

    /// 2026-09-30: The MX entry an `m`-row `[n, k]` launch takes.
    pub fn mx_kernel(&self, m: u32, n: u32, k: u32) -> KernelHandle {
        match mx_plan(&self.0, m, n, k, mx_nt(), mx_ps()) {
            MxLaunch::Tiles { kernel, .. } | MxLaunch::Persistent { kernel, .. } => kernel,
        }
    }

    /// 2026-09-30: Quantize `m` rows of the BF16 `input` (`[m, k]`) into `act`.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        act: Nvfp4ActBuf,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.0.quant)
            .grid([m, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(act.aq)
            .arg_ptr(act.a_scale)
            .arg_ptr(act.a_gs)
            .arg_u32(k)
            .launch(stream)
    }

    /// 2026-09-30: `output[m, n]` (BF16) = the W4A4 product of `act` and `weight`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        act: Nvfp4ActBuf,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let s = &self.0;
        let (mx, grid, smem, sst) = match mx_plan(s, m, n, k, mx_nt(), mx_ps()) {
            MxLaunch::Tiles {
                kernel,
                rows_per_cta,
            } => (kernel, div_ceil(n, rows_per_cta), 0, None),
            MxLaunch::Persistent { kernel, sst, smem } => (kernel, s.sms, smem, Some(sst)),
        };
        ensure!(mx.0 != 0, "w4a4: no kernel for {m} rows");
        let launch = KernelLaunch::new(gpu, mx)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .shared_mem(smem)
            .arg_ptr(act.aq)
            .arg_ptr(act.a_scale)
            .arg_ptr(act.a_gs)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k);
        match sst {
            Some(sst) => launch.arg_u32(sst).launch(stream),
            None => launch.launch(stream),
        }
    }
}
