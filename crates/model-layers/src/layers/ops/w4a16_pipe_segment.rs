// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The launch of a persistent W4A16 MLP segment (`kernels/gb10/common/
//! w4a16_pipe_segment.cuh` over `segment.cuh`): gate|up GEMV, clamped SwiGLU, down GEMV as the
//! three dependent phases of ONE launch, each phase's tasks the blocks the separate launches would
//! run (`w4a16_gemv_pipe_cap` / `w4a16_gemv_pipe_blocks`), so every output's bytes are theirs.
//!
//! Owner: model-layers ops (model-neutral; the entries are named by the caller).
//! Invariants:
//! - `ctl` is a [`SegmentCtl`] slot, zero before the launch; the kernel leaves it zero.
//! - Refuses a shape either GEMV phase's contract refuses (`w4a16_gemv_pipe_ok`).

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{
    W4A16_GEMV_PIPE_WARPS, W4a16Seg, w4a16_gemv_pipe_blocks, w4a16_gemv_pipe_cap,
    w4a16_gemv_pipe_ok, w4a16_gemv_seg_smem, w4a16_gemv_seg_tiles,
};

/// 2026-10-23: Bytes of one control slot (`SegCtl`: ticket, exited, gated, done[4]).
pub const SEGMENT_CTL_BYTES: usize = 28;

/// 2026-10-23: A ring of zeroed control slots. A launch takes the next slot; two launches in
/// flight at once take different slots as long as fewer than `slots` launches are issued between
/// them (a decode step issues at most a few concurrently).
pub struct SegmentCtl {
    base: DevicePtr,
    slots: usize,
    next: std::sync::atomic::AtomicUsize,
}

impl SegmentCtl {
    /// 2026-10-23: `slots` zeroed slots on `gpu`.
    pub fn new(gpu: &dyn GpuBackend, slots: usize) -> Result<Self> {
        ensure!(slots > 0, "a segment control ring needs a slot");
        let base = gpu.alloc(slots * SEGMENT_CTL_BYTES)?;
        gpu.memset(base, 0, slots * SEGMENT_CTL_BYTES)?;
        Ok(Self {
            base,
            slots,
            next: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// 2026-10-23: The next slot.
    pub fn take(&self) -> DevicePtr {
        let i = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.slots;
        self.base.offset(i * SEGMENT_CTL_BYTES)
    }
}

/// 2026-10-23: One dense segment: `x [m, hidden]` -> gate, up `[m, inter]` -> `act` -> `out [m,
/// hidden]`, by `kernel` (an entry of `m` rows), on `sms` SMs, with control slot `ctl`. Only the
/// CTAs on SMs whose id is below `sm_budget` take tasks (`segment.cuh`; `u32::MAX`: every CTA);
/// the tasks, so the bytes, are those of `sms` whatever the budget.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_dense_segment(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    gate: W4a16Seg,
    up: W4a16Seg,
    act: DevicePtr,
    limit: f32,
    down: W4a16Seg,
    m: u32,
    hidden: u32,
    sms: u32,
    ctl: DevicePtr,
    sm_budget: u32,
    stream: u64,
) -> Result<()> {
    let inter = gate.n;
    ensure!(
        up.n == inter
            && w4a16_gemv_pipe_ok(hidden, &[inter, inter], m)
            && w4a16_gemv_pipe_ok(inter, &[down.n], m)
            && down.n == hidden,
        "w4a16_dense_segment: hidden={hidden} inter={inter} down={} rows={m} outside the contract",
        down.n
    );
    let gu = [inter, inter];
    let b0 = w4a16_gemv_pipe_blocks(
        w4a16_gemv_seg_tiles(&gu, 1),
        w4a16_gemv_pipe_cap(&gu, hidden, m, sms),
    );
    let b2 = w4a16_gemv_pipe_blocks(
        w4a16_gemv_seg_tiles(&[hidden], 1),
        w4a16_gemv_pipe_cap(&[hidden], inter, m, sms),
    );
    let smem = w4a16_gemv_seg_smem(hidden, m).max(w4a16_gemv_seg_smem(inter, m));
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([b0.max(b2), 1, 1])
        .block([W4A16_GEMV_PIPE_WARPS * 32, 1, 1])
        .shared_mem(smem)
        .arg_ptr(x);
    for g in [gate, up, down] {
        l = l
            .arg_ptr(g.weight.weight)
            .arg_ptr(g.weight.weight_scale)
            .arg_f32(g.weight.weight_scale_2)
            .arg_ptr(g.output)
            .arg_u32(g.n);
    }
    l.arg_ptr(act)
        .arg_f32(limit)
        .arg_u32(hidden)
        .arg_u32(inter)
        .arg_u32(b0)
        .arg_u32(b2)
        .arg_ptr(ctl)
        .arg_u32(sm_budget)
        .launch(stream)
}
