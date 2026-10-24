// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `METRALE_GLM_DENSE_SEGMENT`: the P3 prototype of the persistent-segment executor
//! (DESIGN-fuser-megakernel.md §4.2): a GLM-5.3 dense MLP (the shared expert, under
//! `--dense-quantization w4a16`) whose gate|up and down both run on `w4a16_gemv_pipe` launches
//! `glm5next_dense_seg_pipe_m<M>` (`kernels/gb10/common/glm5next_dense_seg.cu`, an instantiation
//! of the pipe template through `w4a16_pipe_segment.cuh`): the three launches (gate|up, the
//! clamped SwiGLU, down) as the dependent phases of one persistent launch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`dense_segment`] launches only where the three launches it replaces would run exactly as
//!   the segment's phases: gate|up as one `w4a16_gemv_pipe` group
//!   ([`group_on_pipe`]), the down as a lone pipe projection
//!   (`SegPlan::lone_pipe`), every weight registered; otherwise it launches nothing and returns
//!   `Ok(false)`. Its bytes are theirs (gate, up, act and out: the kernel's invariant).
//! - The control ring is resolved once per process at load ([`prepare`]), never inside a capture.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{SegmentCtl, W4a16Seg, w4a16_dense_segment};

use crate::glm5next_dense_levers::DenseLevers;
use crate::glm5next_w4a16_dense::{lookup, registry_in_use, seg_gemv_group};
use crate::glm5next_w4a16_seg as seg;

/// 2026-10-23: The kernel module.
pub const MODULE: &str = "glm5next_dense_seg";

/// 2026-10-23: Control slots: launches a step can have in flight at once are far fewer.
const CTL_SLOTS: usize = 128;

/// 2026-10-23: The process's control ring (one model per process).
static CTL: OnceLock<SegmentCtl> = OnceLock::new();

/// 2026-10-23: The entry of `m` rows.
pub fn entry(m: usize) -> Option<&'static str> {
    match m {
        1 => Some("glm5next_dense_seg_pipe_m1"),
        2 => Some("glm5next_dense_seg_pipe_m2"),
        _ => None,
    }
}

/// 2026-10-23: The dense MLP's buffers and weights for one segment.
pub struct DenseSegIo {
    pub x: DevicePtr,
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub down_w: DevicePtr,
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub act: DevicePtr,
    pub out: DevicePtr,
}

/// 2026-10-23: `out = down(swiglu_clamp(gate(x), up(x)))` over `m` rows as one segment when the
/// module invariants allow; `Ok(false)` when they do not (nothing launched).
/// 2026-10-23: Whether a registered group of `m` rows runs as one `w4a16_gemv_pipe` launch
/// under `plan` (`proj_group_registered`'s first branch on the pipe): the segment's gate|up phase
/// stands in for exactly that launch.
fn group_on_pipe(plan: seg::SegPlan, m: usize) -> bool {
    seg_gemv_group(plan, m) && plan.pipe_rows(m)
}

#[allow(clippy::too_many_arguments)]
pub fn dense_segment(
    gpu: &dyn GpuBackend,
    l: &DenseLevers,
    io: &DenseSegIo,
    limit: f32,
    m: usize,
    hidden: usize,
    inter: usize,
    stream: u64,
) -> Result<bool> {
    if !l.dense_seg || !registry_in_use() {
        return Ok(false);
    }
    let plan = seg::plan(l)?;
    let Some(name) = entry(m).filter(|_| group_on_pipe(plan, m) && plan.lone_pipe(m)) else {
        return Ok(false);
    };
    let (Some(g), Some(u), Some(d)) = (
        lookup(io.gate_w, inter, hidden)?,
        lookup(io.up_w, inter, hidden)?,
        lookup(io.down_w, hidden, inter)?,
    ) else {
        return Ok(false);
    };
    let ctl = CTL
        .get()
        .context("METRALE_GLM_DENSE_SEGMENT: the control ring was not prepared at load")?;
    let kernel = gpu
        .op_cache()
        .kernel(gpu, MODULE, name)
        .with_context(|| format!("METRALE_GLM_DENSE_SEGMENT: {MODULE}::{name}"))?;
    let budget = match l.dense_seg_sms {
        None => u32::MAX,
        Some(k) if k.get() <= plan.sms => k.get(),
        Some(k) => bail!(
            "METRALE_GLM_DENSE_SEGMENT: an SM budget of {k} on a {}-SM device",
            plan.sms
        ),
    };
    let (n_i, n_h) = (u32::try_from(inter)?, u32::try_from(hidden)?);
    let s = |weight, output, n| W4a16Seg { weight, output, n };
    w4a16_dense_segment(
        gpu,
        kernel,
        io.x,
        s(g.w, io.gate, n_i),
        s(u.w, io.up, n_i),
        io.act,
        limit,
        s(d.w, io.out, n_h),
        u32::try_from(m)?,
        n_h,
        plan.sms,
        ctl.take(),
        budget,
        stream,
    )?;
    Ok(true)
}

/// 2026-10-23: At load, wherever the pipe is on (`glm5next_w4a16_seg::prepare`), whether or not
/// the lever is (the paired A/B instrument switches it per step): resolve every entry and the
/// control ring, so nothing is refused or allocated inside a capture.
pub fn prepare(gpu: &dyn GpuBackend) -> Result<()> {
    for name in (1..=2).filter_map(entry) {
        gpu.op_cache()
            .kernel(gpu, MODULE, name)
            .with_context(|| format!("METRALE_GLM_DENSE_SEGMENT: {MODULE}::{name}"))?;
    }
    if CTL.get().is_none() {
        let _ = CTL.set(SegmentCtl::new(gpu, CTL_SLOTS)?);
    }
    Ok(())
}
