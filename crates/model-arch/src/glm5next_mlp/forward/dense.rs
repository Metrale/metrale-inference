// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The GLM-5.3 dense SwiGLU MLP forward, used by the dense layers and the shared
//! expert.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_dense` leaves a partial sum in `out` when `inter` is a TP shard; the caller
//!   all-reduces.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpWorkspace;
use super::launch::{gemm, swiglu};
use crate::glm5next_mlp::precision::MlpKernel;
use crate::glm5next_mlp::weights::{Glm5NextDenseMlpWeights, Glm5NextDenseSite};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-09-25: A BF16 SwiGLU MLP of width `inter`, `down(clamped_swiglu(gate(x), up(x)))`, for a
/// dense layer or the shared expert. Errors when `inter` or `m` does not fit the workspace.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseMlpWeights,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if inter == 0 || inter > ws.max_inter {
        bail!(
            "GLM dense MLP: width {inter} does not fit a workspace built for {}",
            ws.max_inter
        );
    }
    if m == 0 || m > ws.max_rows {
        bail!(
            "GLM dense MLP: {m} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    // 2026-10-09: gate and up read `x` unchanged: one FP8 quantization under the FP8 tier.
    let stable = crate::glm5next_fp8_dense::stable_input(x);
    // 2026-10-09: One launch under `METRALE_GLM_W4A16_SEG` (`proj_group`), else one each.
    let group = [(w.gate_proj, ws.a_gate, inter), (w.up_proj, ws.a_up, inter)];
    if !crate::glm5next_w4a16_dense::proj_group(gpu, &group, x, m, cfg.hidden, stream)? {
        gemm(
            gpu,
            k.gemm,
            k.gemv,
            k.batchm(),
            x,
            w.gate_proj,
            ws.a_gate,
            m,
            inter,
            cfg.hidden,
            stream,
        )?;
        gemm(
            gpu,
            k.gemm,
            k.gemv,
            k.batchm(),
            x,
            w.up_proj,
            ws.a_up,
            m,
            inter,
            cfg.hidden,
            stream,
        )?;
    }
    drop(stable);
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        // 2026-09-25: Elementwise over all `m * inter` values.
        m * inter,
        cfg.swiglu_limit,
        stream,
    )?;
    gemm(
        gpu,
        k.gemm,
        k.gemv,
        k.batchm(),
        ws.a_act,
        w.down_proj,
        out,
        m,
        cfg.hidden,
        inter,
        stream,
    )
}

/// 2026-10-08: A dense MLP site (`Glm5NextDenseSite`) over `m` rows: the kernel its precision
/// plan gives for `m` rows, W4A4 on the packed weights or BF16 on the dequantized ones. Errors
/// when the plan reaches a form the loader did not build (a loader bug, never a fallback).
#[allow(clippy::too_many_arguments)]
pub fn forward_dense_site(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    site: &Glm5NextDenseSite,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    match site.precision.kernel(m) {
        MlpKernel::W4a4Static => {
            let Some((w, scales)) = &site.nvfp4 else {
                bail!(
                    "GLM dense MLP: the plan runs W4A4 at {m} rows but no packed weights were bound"
                );
            };
            super::w4a4::forward_dense_w4a4(gpu, k, cfg, w, *scales, inter, x, out, m, ws, stream)
        }
        MlpKernel::Bf16 | MlpKernel::W4a16 => {
            let Some(w) = &site.bf16 else {
                bail!(
                    "GLM dense MLP: the plan runs BF16 at {m} rows but no BF16 weights were bound"
                );
            };
            forward_dense(gpu, k, cfg, w, inter, x, out, m, ws, stream)
        }
    }
}
