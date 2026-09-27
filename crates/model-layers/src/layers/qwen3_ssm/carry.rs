// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Carried-state GDN verify (`gdn_carry_wy{2,3,4}` and `gdn_carry_flush`,
//! kernels/gb10/common/gated_delta_rule_carry.cu): the layer-side handles and binding, the
//! engage decision as a pure function, the verify launch, and the fold a declining layer
//! runs before its parent kernels.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - The carry kernels run only when `carry_decision` holds, which requires the caller's
//!   request (`ForwardContext::gdn_write_on_accept` with a carry binding present).
//! - A layer asked to carry that declines folds the batch's pending rows into H before
//!   any parent kernel reads H (`carry_flush_run`).
//!
//! ## What the model owns
//!
//! The model allocates the stash (`[layers][slots][seq_floats]` f32), the pending counts
//! (`[layers][slots]` u32), the slot table (`u32` per batch position) and one engaged
//! word per layer, binds them once before any capture, stages the slot table per verify
//! and sets the pending counts after each verdict (`model-engine gdn_carry.rs`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::GdnCarryBinding;
use crate::layers::ops;

/// 2026-09-26: The kernel handles, each 0 when the module is absent: the verify for
/// K = 2, 3, 4 and the fold.
pub(super) struct CarryKernels {
    pub wy: [KernelHandle; 3],
    pub flush: KernelHandle,
    pub conv: KernelHandle,
    pub conv_flush: KernelHandle,
}

pub(super) fn carry_kernels(gpu: &dyn GpuBackend) -> CarryKernels {
    let m = "gated_delta_rule_carry";
    CarryKernels {
        wy: [
            crate::layers::try_kernel(gpu, m, "gdn_carry_wy2"),
            crate::layers::try_kernel(gpu, m, "gdn_carry_wy3"),
            crate::layers::try_kernel(gpu, m, "gdn_carry_wy4"),
        ],
        flush: crate::layers::try_kernel(gpu, m, "gdn_carry_flush"),
        conv: crate::layers::try_kernel(gpu, m, "gdn_carry_conv"),
        conv_flush: crate::layers::try_kernel(gpu, m, "gdn_carry_conv_flush"),
    }
}

/// 2026-09-26: The per-layer carry state: handles, head dims and the binding.
pub(super) struct CarryState {
    pub kernels: CarryKernels,
    /// 2026-09-26: `[nk, nv, kd, vd]`.
    pub dims: [usize; 4],
    pub binding: std::sync::OnceLock<GdnCarryBinding>,
}

/// 2026-09-26: Everything the engage decision depends on, so tests can pin it without a
/// GPU.
#[derive(Clone, Copy, Debug)]
pub(super) struct CarryRequest {
    /// 2026-09-26: `ForwardContext::gdn_write_on_accept` in a serve that bound carry.
    pub requested: bool,
    pub kk: usize,
    pub h_f16: bool,
    pub kd: usize,
    pub vd: usize,
    pub kernel_linked: bool,
    pub bound: bool,
}

/// 2026-09-26: The one place that decides whether a carry kernel runs.
pub(super) const fn carry_decision(r: CarryRequest) -> bool {
    r.requested
        && r.kk >= 2
        && r.kk <= 4
        && !r.h_f16
        && r.kd == 128
        && r.vd == 128
        && r.kernel_linked
        && r.bound
}

impl CarryState {
    pub(super) fn new(gpu: &dyn GpuBackend, dims: [usize; 4]) -> Self {
        Self {
            kernels: carry_kernels(gpu),
            dims,
            binding: std::sync::OnceLock::new(),
        }
    }

    /// 2026-09-26: Stash width when this layer can carry: every kernel linked and
    /// kd == vd == 128. `None` otherwise.
    pub(super) fn seq_floats(&self) -> Option<usize> {
        let [_, nv, kd, vd] = self.dims;
        let k = &self.kernels;
        let linked = k.wy.iter().all(|h| h.0 != 0)
            && [k.flush, k.conv, k.conv_flush].iter().all(|h| h.0 != 0);
        (linked && kd == 128 && vd == 128).then(|| ops::gdn_carry_seq_floats(nv, kd, vd))
    }

    fn kernel_for(&self, kk: usize) -> KernelHandle {
        match kk {
            2..=4 => self.kernels.wy[kk - 2],
            _ => KernelHandle(0),
        }
    }
}

impl Qwen3SsmLayer {
    /// 2026-09-26: The binding when this call asked for carry and a binding exists.
    pub(super) fn carry_requested(&self, write_on_accept: bool) -> Option<GdnCarryBinding> {
        if !write_on_accept {
            return None;
        }
        self.carry.binding.get().copied()
    }

    /// 2026-09-26: Whether the carry kernel for verify width `kk` runs on this call.
    pub(super) fn carry_now(&self, write_on_accept: bool, kk: usize) -> bool {
        let [_, _, kd, vd] = self.carry.dims;
        carry_decision(CarryRequest {
            requested: write_on_accept,
            kk,
            h_f16: super::ssm_h_fp16_enabled(),
            kd,
            vd,
            kernel_linked: self.carry.kernel_for(kk).0 != 0,
            bound: self.carry.binding.get().is_some(),
        })
    }

    /// 2026-09-26: Launch the carry verify for one run of `n` sequences at width `kk`.
    /// `slot_tab` and `flag` are the run's slices of the shared slot table and of this
    /// layer's engaged words.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn carry_launch(
        &self,
        gpu: &dyn GpuBackend,
        kk: usize,
        h_table: DevicePtr,
        slot_tab: DevicePtr,
        flag: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gate_ptr: DevicePtr,
        beta_ptr: DevicePtr,
        gdn_out_buf: DevicePtr,
        n: usize,
        conv_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let b = self
            .carry
            .binding
            .get()
            .ok_or_else(|| anyhow::anyhow!("carry_launch: no carry binding"))?;
        let [nk, nv, kd, _] = self.carry.dims;
        ops::gdn_carry_wy(
            gpu,
            self.carry.kernel_for(kk),
            h_table,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            b.stash,
            slot_tab,
            b.pend,
            b.seq_floats as u32,
            n as u32,
            nk as u32,
            nv as u32,
            conv_dim as u32,
            conv_dim as u32,
            (nv * 2) as u32,
            kd as u32,
            flag,
            stream,
        )?;
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let bit = 1u32 << kk;
        if LOGGED.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
            tracing::info!(
                "batched-verify GDN CARRY ENGAGED (n={n}, k={kk}): pending rows folded in the \
                 verify, no h intermediates written (kill switch METRALE_NO_GDN_CARRY)"
            );
        }
        Ok(())
    }

    /// 2026-09-26: The carried conv launch for one run (`gdn_carry_conv`), with the batched
    /// parent's arguments: `conv_base` is the run's first conv state, the rest follow at
    /// `conv_bytes` strides; `slot_tab` is the run's slice of the slot table.
    pub(super) fn carry_conv_launch(
        &self,
        gpu: &dyn GpuBackend,
        conv_base: DevicePtr,
        slot_tab: DevicePtr,
        n: usize,
        a: &super::trait_decode_batched_conv_gdn::ConvGdnArgs,
    ) -> Result<()> {
        let b = self
            .carry
            .binding
            .get()
            .ok_or_else(|| anyhow::anyhow!("carry_conv_launch: no carry binding"))?;
        ops::gdn_carry_conv(
            gpu,
            self.carry.kernels.conv,
            conv_base,
            a.deinterleaved,
            &self.ssm.conv1d,
            a.conv_out_buf,
            b.conv_stash,
            slot_tab,
            b.pend,
            b.conv_seq_elems as u32,
            a.num_tokens as u32,
            a.conv_dim as u32,
            a.d_conv as u32,
            a.qk_ch,
            a.kd as u32,
            a.qkvz_size as u32,
            a.conv_dim as u32,
            1e-6,
            n as u32,
            (self.conv_state_bytes / 4) as u32,
            (a.num_tokens * a.qkvz_size) as u32,
            (a.num_tokens * a.conv_dim) as u32,
            a.stream,
        )
    }

    /// 2026-09-26: Fold the pending rows of `n` sequences into this layer's H and conv
    /// window before a parent kernel reads them. `h_table`, `slot_tab` and `run_first`
    /// locate the run in the WY table slab 0, the slot table and the conv-state table. A
    /// no-op without a binding.
    pub(super) fn carry_flush_run(
        &self,
        gpu: &dyn GpuBackend,
        h_table: DevicePtr,
        run_first: usize,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(b) = self.carry.binding.get().copied() else {
            return Ok(());
        };
        anyhow::ensure!(
            !h_table.is_null(),
            "carry_flush_run: a layer asked to carry declined without WY tables"
        );
        let [nk, nv, kd, vd] = self.carry.dims;
        let slot_tab = b.slot_tab.offset(run_first * 4);
        ops::gdn_carry_flush(
            gpu,
            self.carry.kernels.flush,
            h_table,
            0,
            b.stash,
            0,
            slot_tab,
            b.pend,
            0,
            b.seq_floats as u32,
            n as u32,
            nv as u32,
            1,
            stream,
        )?;
        ops::gdn_carry_conv_flush(
            gpu,
            self.carry.kernels.conv_flush,
            b.conv_tab.offset(run_first * 8),
            0,
            b.conv_stash,
            0,
            slot_tab,
            b.pend,
            0,
            b.conv_seq_elems as u32,
            n as u32,
            (nk * kd * 2 + nv * vd) as u32,
            (self.conv_state_bytes / 4 / (nk * kd * 2 + nv * vd)) as u32,
            1,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{CarryRequest, carry_decision};

    const GO: CarryRequest = CarryRequest {
        requested: true,
        kk: 2,
        h_f16: false,
        kd: 128,
        vd: 128,
        kernel_linked: true,
        bound: true,
    };

    #[test]
    fn widths_two_to_four_engage() {
        for kk in 2..=4 {
            assert!(carry_decision(CarryRequest { kk, ..GO }));
        }
    }

    #[test]
    fn each_gate_declines_alone() {
        assert!(!carry_decision(CarryRequest {
            requested: false,
            ..GO
        }));
        assert!(!carry_decision(CarryRequest { kk: 1, ..GO }));
        assert!(!carry_decision(CarryRequest { kk: 5, ..GO }));
        assert!(!carry_decision(CarryRequest { h_f16: true, ..GO }));
        assert!(!carry_decision(CarryRequest { kd: 64, ..GO }));
        assert!(!carry_decision(CarryRequest { vd: 256, ..GO }));
        assert!(!carry_decision(CarryRequest {
            kernel_linked: false,
            ..GO
        }));
        assert!(!carry_decision(CarryRequest { bound: false, ..GO }));
    }
}
