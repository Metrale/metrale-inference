// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `METRALE_GLM_SWIGLU_DOWN_FOLD`: a GLM-5.3 dense MLP (the shared expert, under
//! `--dense-quantization w4a16`) whose down projection runs on `w4a16_gemv_pipe` launches
//! `glm5next_swiglu_down_pipe_m<M>` (`kernels/gb10/common/glm5next_swiglu_down.cu`) instead of
//! `glm5next_swiglu_clamp` then the down: the pipe template staging the clamped SwiGLU of gate
//! and up, so one launch and the activation buffer's write and read go.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`swiglu_down`] launches only where the unfolded down would run on `w4a16_gemv_pipe`
//!   (`SegPlan::lone_pipe`, the predicate `glm5next_w4a16_dense::proj_registered` routes by) on a
//!   registered weight; otherwise it launches nothing and returns `Ok(false)`, and the caller
//!   runs the unfolded pair. The output bytes are the unfolded pair's (the kernel's invariant).
//! - The activation buffer is not written when it folds; nothing after the down reads it.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{
    PipeInput, W4a16Seg, w4a16_gemv_pipe_cap, w4a16_gemv_pipe_staged,
};

use crate::glm5next_dense_levers::DenseLevers;
use crate::glm5next_w4a16_dense::{lookup, registry_in_use};
use crate::glm5next_w4a16_seg as seg;

/// 2026-10-23: The kernel module.
pub const MODULE: &str = "glm5next_swiglu_down";

/// 2026-10-23: The entry of `m` rows (the pipe template's row counts).
pub fn entry(m: usize) -> Option<&'static str> {
    match m {
        1 => Some("glm5next_swiglu_down_pipe_m1"),
        2 => Some("glm5next_swiglu_down_pipe_m2"),
        _ => None,
    }
}

/// 2026-10-23: `out [m, n] = swiglu_clamp(gate, up, limit) [m, k] @ W^T` for the down weight
/// registered under `down`, in one launch, when the module invariants allow; `Ok(false)` when
/// they do not (nothing launched).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_down(
    gpu: &dyn GpuBackend,
    l: &DenseLevers,
    gate: DevicePtr,
    up: DevicePtr,
    limit: f32,
    down: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !l.swiglu_down || !registry_in_use() {
        return Ok(false);
    }
    let plan = seg::plan(l)?;
    let Some(name) = entry(m).filter(|_| plan.lone_pipe(m)) else {
        return Ok(false);
    };
    let Some(e) = lookup(down, n, k)? else {
        return Ok(false);
    };
    let kernel = gpu
        .op_cache()
        .kernel(gpu, MODULE, name)
        .with_context(|| format!("METRALE_GLM_SWIGLU_DOWN_FOLD: {MODULE}::{name}"))?;
    let (n32, k32, m32) = (u32::try_from(n)?, u32::try_from(k)?, u32::try_from(m)?);
    let segs = [W4a16Seg {
        weight: e.w,
        output: out,
        n: n32,
    }];
    let cap = w4a16_gemv_pipe_cap(&[n32], k32, m32, plan.sms);
    let input = PipeInput::Pair(gate, up, limit);
    w4a16_gemv_pipe_staged(gpu, kernel, input, &segs, m32, k32, cap, stream)?;
    Ok(true)
}

/// 2026-10-23: At load, wherever the pipe is on (`glm5next_w4a16_seg::prepare`), whether or not
/// the lever is (the paired A/B instrument switches it per step): resolve every entry, so a
/// missing one is refused here, never inside a capture.
pub fn prepare(gpu: &dyn GpuBackend) -> Result<()> {
    for name in (1..=2).filter_map(entry) {
        gpu.op_cache()
            .kernel(gpu, MODULE, name)
            .with_context(|| format!("METRALE_GLM_SWIGLU_DOWN_FOLD: {MODULE}::{name}"))?;
    }
    Ok(())
}
