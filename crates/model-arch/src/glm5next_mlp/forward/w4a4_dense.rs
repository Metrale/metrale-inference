// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The dense SwiGLU MLP of GLM-5.3 at the declared W4A4 (`forward_dense_w4a4`) on the
//! mx GEMVs in chunks of [`DENSE_W4A4_CHUNK_ROWS`] rows, moved here unchanged from `w4a4.rs`
//! (which keeps the routed experts and the shared static quantizer). 2026-10-12: At or above
//! `METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS` rows, one group of the grouped GEMM per projection
//! (`w4a4_dense_grouped`), byte-identical to the chunks.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: as `w4a4.rs`'s: a row's output does not depend on the row count or on the other
//! rows; launches only after the shape checks pass.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Glm5NextMlpWorkspace;
use super::launch::swiglu;
use super::w4a4::{W4A4_BLOCK, W4A4_ROWS_PER_CTA, quant_static};
use crate::glm5next_mlp::build_w4a4::check_w4a4_k;
use crate::glm5next_mlp::weights::{Glm5NextDenseNvfp4Weights, Nvfp4Proj, W4a4ActScales};
use crate::glm5next_mlp::{DENSE_W4A4_CHUNK_ROWS, Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-08: The mx entry for `rows` (1..=32): mx8, mx16 or mx32.
fn mx_kernel(k: &Glm5NextMlpKernels, rows: usize) -> KernelHandle {
    match rows {
        0..=8 => k.w4a4_mx[0],
        9..=16 => k.w4a4_mx[1],
        _ => k.w4a4_mx[2],
    }
}

/// 2026-10-08: `out[rows, n]` = the quantized scratch times `w` (`[n, k]` NVFP4).
#[allow(clippy::too_many_arguments)]
fn mx_proj(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    ws: &Glm5NextMlpWorkspace,
    w: &Nvfp4Proj,
    out: DevicePtr,
    rows: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, mx_kernel(k, rows))
        .grid([div_ceil(n as u32, W4A4_ROWS_PER_CTA), 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(ws.w4a4_aq)
        .arg_ptr(ws.w4a4_as)
        .arg_ptr(ws.w4a4_ag)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        .arg_f32(w.scale_2)
        .arg_ptr(out)
        .arg_u32(rows as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)
}

/// 2026-10-08: The dense SwiGLU MLP of width `inter` at W4A4, `m` rows in chunks of
/// [`DENSE_W4A4_CHUNK_ROWS`]: gate and up per chunk from one quantization, the clamped SwiGLU
/// over all rows, then down per chunk. `out` is a partial sum under TP, as in `forward_dense`.
/// 2026-10-12: With `grouped` (`METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS`) taking `m`, each
/// projection runs as one group of the grouped GEMM over all rows instead, the same bytes.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense_w4a4(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseNvfp4Weights,
    scales: W4a4ActScales,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    grouped: Option<&super::w4a4_dense_grouped::DenseGroupedTables>,
    stream: u64,
) -> Result<()> {
    let h = cfg.hidden;
    ensure!(
        m >= 1 && m <= ws.max_rows() && inter <= ws.max_inter,
        "GLM dense W4A4: {m} rows x {inter} do not fit the workspace"
    );
    check_w4a4_k(h, "dense gate/up")?;
    check_w4a4_k(inter, "dense down")?;
    // 2026-10-12: `METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS`: each projection as one group of the
    // grouped GEMM, one quantization per input (`w4a4_dense_grouped`; the same bytes).
    if let Some(t) = grouped.filter(|t| t.takes(m)) {
        use super::w4a4_dense_grouped as g;
        g::prepare(gpu, k, ws, t, m, h, inter, stream)?;
        quant_static(
            gpu,
            k.w4a4_quant_static,
            ws,
            x,
            m,
            h,
            h,
            scales.gate_up,
            stream,
        )?;
        g::proj(gpu, k, ws, t, &t.gate, ws.a_gate, m, inter, h, stream)?;
        g::proj(gpu, k, ws, t, &t.up, ws.a_up, m, inter, h, stream)?;
        swiglu(
            gpu,
            k.swiglu,
            ws.a_gate,
            ws.a_up,
            ws.a_act,
            m * inter,
            cfg.swiglu_limit,
            stream,
        )?;
        quant_static(
            gpu,
            k.w4a4_quant_static,
            ws,
            ws.a_act,
            m,
            inter,
            inter,
            scales.down,
            stream,
        )?;
        return g::proj(gpu, k, ws, t, &t.down, out, m, h, inter, stream);
    }
    let chunks = (0..m)
        .step_by(DENSE_W4A4_CHUNK_ROWS)
        .map(|r0| (r0, (m - r0).min(DENSE_W4A4_CHUNK_ROWS)));
    for (r0, n) in chunks.clone() {
        quant_static(
            gpu,
            k.w4a4_quant_static,
            ws,
            x.offset(r0 * h * 2),
            n,
            h,
            h,
            scales.gate_up,
            stream,
        )?;
        mx_proj(
            gpu,
            k,
            ws,
            &w.gate_proj,
            ws.a_gate.offset(r0 * inter * 2),
            n,
            inter,
            h,
            stream,
        )?;
        mx_proj(
            gpu,
            k,
            ws,
            &w.up_proj,
            ws.a_up.offset(r0 * inter * 2),
            n,
            inter,
            h,
            stream,
        )?;
    }
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        m * inter,
        cfg.swiglu_limit,
        stream,
    )?;
    for (r0, n) in chunks {
        quant_static(
            gpu,
            k.w4a4_quant_static,
            ws,
            ws.a_act.offset(r0 * inter * 2),
            n,
            inter,
            inter,
            scales.down,
            stream,
        )?;
        mx_proj(
            gpu,
            k,
            ws,
            &w.down_proj,
            out.offset(r0 * h * 2),
            n,
            h,
            inter,
            stream,
        )?;
    }
    Ok(())
}
