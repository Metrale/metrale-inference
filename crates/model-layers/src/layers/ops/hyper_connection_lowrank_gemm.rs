// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GEMM formulation of the low-rank hyper-connection collapse, on either `dense_gemm_bf16_pipelined` or cuBLASLt.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types. `hyper_connection_lowrank.rs` passes `use_cublas = true`
//! for `T <= 64` and, unless `METRALE_HC_PREFILL_CUBLAS=1` (2026-10-14), `false` for larger
//! `T`. With `use_cublas` and a T the decode GEMVs admit
//! (2026-10-14, `hyper_connection_lowrank_gemv.rs`), those two launches replace the projections
//! and `hc_silu_scale` / `hc_pre_mix`; with `METRALE_HC_SITE_FUSED=1` and a shape it admits
//! (2026-10-14, `hyper_connection_lowrank_site.rs`), one launch replaces the stage and both GEMVs.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::gemv::{HcGemv, HcGemvArgs};
use super::site::{HcSite, HcSiteArgs};
use super::{HC_ELEMENTWISE_BLOCK, hc_hidden_split};
use crate::layers::qwen3_attention::HcLowRank;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Stage `normed` in BF16, run the down, up and (when `inject`) injection projections
/// as GEMMs, and finish with the kernels `hc_silu_scale` and `hc_pre_mix`. Tokens
/// are processed in slabs of at most 2048 so the scratch region stays bounded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_gemm(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    use_cublas: bool,
    stream: u64,
) -> Result<()> {
    const SLAB: u32 = 2048;
    let hc_dim = (hc_mult * hidden_size) as usize;
    let rank = w.rank as u32;
    // 2026-09-25: Scratch layout (BF16): normed [L, hc_dim], up_pre [L, hc_dim], low [L, rank],
    // inj_pre [L, hc], with L = min(T, 2048). gpu-runtime `buffers/sizes.rs` reserves this layout
    // at `m.min(2048)` rows, so it fits whenever T <= m.
    let lay = num_tokens.min(SLAB) as usize;
    let normed = scratch;
    let up_pre = scratch.offset(lay * hc_dim * 2);
    let low = scratch.offset(2 * lay * hc_dim * 2);
    let inj_pre = scratch.offset(2 * lay * hc_dim * 2 + lay * w.rank * 2);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?;
    let k_silu = gpu.kernel("hyper_connection", "hc_silu_scale")?;
    let k_mix = gpu.kernel("hyper_connection", "hc_pre_mix")?;
    let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let inv_hc = 1.0f32 / hc_mult as f32;
    let gemv = use_cublas.then(|| HcGemv::resolve(gpu));
    let site = use_cublas.then(|| HcSite::resolve(gpu));

    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = SLAB.min(num_tokens - t0);
        let streams_s = streams.offset(t0 as usize * hc_dim * 4);
        let y_s = y_out.offset(t0 as usize * hidden_size as usize * 2);
        let inj_s = inj_out.offset(t0 as usize * hc_mult as usize * 4);

        // 2026-10-14: Decode widths: stage, down + injection and up + mix in one launch, the down
        // partials in the unused `up_pre` rows.
        let site_args = HcSiteArgs {
            streams: streams_s,
            norm_w: w.norm_w,
            down_w: w.down_w,
            inject_w: if inject { w.inject_w } else { DevicePtr::NULL },
            up_w: w.up_w,
            normed,
            partial: up_pre,
            low,
            y_out: y_s,
            inj_out: inj_s,
        };
        let up_pre_bytes = lay * hc_dim * 2;
        if let Some(s) =
            site.filter(|s| s.admits(&site_args, ts, hidden_size, hc_mult, rank, up_pre_bytes))
        {
            s.launch(
                gpu,
                &site_args,
                ts,
                hidden_size,
                hc_mult,
                rank,
                norm_eps,
                stream,
            )?;
            t0 += ts;
            continue;
        }

        // 2026-10-13: One block per (token, stream): a stream's RMS needs only its own slice.
        KernelLaunch::new(gpu, k_stage)
            .grid([ts, hc_mult, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams_s)
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;

        // 2026-10-14: Decode widths: down + injection, then up + mix, on the Hopper GEMVs.
        let args = HcGemvArgs {
            normed,
            down_w: w.down_w,
            up_w: w.up_w,
            inject_w: if inject { w.inject_w } else { DevicePtr::NULL },
            low,
            y_out: y_s,
            inj_out: inj_s,
        };
        if let Some(g) = gemv.filter(|g| g.admits(&args, ts, hidden_size, hc_mult, rank)) {
            g.launch(gpu, &args, ts, hidden_size, hc_mult, rank, stream)?;
            t0 += ts;
            continue;
        }

        // 2026-09-25: low = normed x down_w^T   [ts, rank]
        if use_cublas {
            crate::layers::ops::cublas_bf16_proj_dense(
                normed,
                w.down_w,
                low,
                ts,
                rank,
                hc_dim as u32,
                stream,
            )?;
        } else {
            gemm_raw(
                gpu,
                k_gemm,
                normed,
                w.down_w,
                low,
                ts,
                rank,
                hc_dim as u32,
                stream,
            )?;
        }
        let n_low = ts * rank;
        KernelLaunch::new(gpu, k_silu)
            .grid([n_low.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(low)
            .arg_u32(n_low)
            .arg_f32(inv_hc)
            .launch(stream)?;

        // 2026-09-25: up_pre = low x up_w^T   [ts, hc_dim]
        if use_cublas {
            crate::layers::ops::cublas_bf16_proj_dense(
                low,
                w.up_w,
                up_pre,
                ts,
                hc_dim as u32,
                rank,
                stream,
            )?;
        } else {
            gemm_raw(
                gpu,
                k_gemm,
                low,
                w.up_w,
                up_pre,
                ts,
                hc_dim as u32,
                rank,
                stream,
            )?;
        }
        if inject {
            // 2026-09-25: inj_pre = normed x inject_w^T   [ts, hc]
            if use_cublas {
                crate::layers::ops::cublas_bf16_proj_dense(
                    normed,
                    w.inject_w,
                    inj_pre,
                    ts,
                    hc_mult,
                    hc_dim as u32,
                    stream,
                )?;
            } else {
                gemm_raw(
                    gpu,
                    k_gemm,
                    normed,
                    w.inject_w,
                    inj_pre,
                    ts,
                    hc_mult,
                    hc_dim as u32,
                    stream,
                )?;
            }
        }

        KernelLaunch::new(gpu, k_mix)
            .grid([ts, hc_hidden_split(hidden_size), 1])
            .block([HC_ELEMENTWISE_BLOCK, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(up_pre)
            .arg_ptr(if inject { inj_pre } else { DevicePtr::NULL })
            .arg_ptr(y_out.offset(t0 as usize * hidden_size as usize * 2))
            .arg_ptr(inj_out.offset(t0 as usize * hc_mult as usize * 4))
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(inv_hc)
            .launch(stream)?;

        t0 += ts;
    }
    Ok(())
}

fn gemm_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
