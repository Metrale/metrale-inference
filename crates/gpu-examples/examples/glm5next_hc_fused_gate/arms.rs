// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The two arms of the mHC pre-norm gate, their timing and the byte comparison.
//!
//! Owner: model-arch examples (GLM-5.3 kernels).
//! Invariants: none beyond the types.

use super::*;

pub(super) fn site(g: &dyn GpuBackend, hid: usize, hc: usize, t: usize, salt: u64) -> Result<Site> {
    let m = metrale_model_arch::glm5next_mhc::mix_hc(hc);
    let hc_dim = hc * hid;
    let mut seed = 0x6c6d_7f00_u64 ^ (hid as u64) << 20 ^ (hc as u64) << 8 ^ t as u64 ^ salt << 40;
    let streams: Vec<f32> = (0..t * hc_dim).map(|_| lcg(&mut seed) * 3.0).collect();
    // 2026-10-09: `hc_fn` is scaled down so the mixes stay in the sigmoids' and the softmax's
    // working range; the byte comparison holds at any scale.
    let hc_fn: Vec<f32> = (0..m * hc_dim).map(|_| lcg(&mut seed) * 0.02).collect();
    let hc_base: Vec<f32> = (0..m).map(|_| lcg(&mut seed)).collect();
    let norm_w: Vec<f32> = (0..hid).map(|_| 1.0 + 0.5 * lcg(&mut seed)).collect();
    let done = vec![0u8; metrale_gpu_examples::mhc_mix_tokens()? * 4];
    Ok(Site {
        streams: up(g, &f32_bytes(&streams))?,
        norm_w: up(g, &bf16_bytes(&norm_w))?,
        w: Glm5NextMhcSiteWeights {
            hc_fn: up(g, &bf16_bytes(&hc_fn))?,
            hc_fn_bf16: true,
            hc_scale: up(g, &f32_bytes(&[0.7, 1.3, 0.9]))?,
            hc_base: up(g, &f32_bytes(&hc_base))?,
            mix: g.alloc(metrale_gpu_examples::mhc_mix_tokens()? * m * 4)?,
            mix_tokens: metrale_gpu_examples::mhc_mix_tokens()?,
            done: up(g, &done)?,
        },
    })
}

/// 2026-10-09: The unfused arm: `glm_hc_pre`, then `rms_norm_vanilla` with the block width the
/// GLM layer's `norm` launches (`min(hidden, 1024)`).
#[allow(clippy::too_many_arguments)]
pub(super) fn unfused(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    norm_k: KernelHandle,
    s: &Site,
    o: &Outs,
    t: usize,
    hid: usize,
    hc: usize,
    stream: u64,
) -> Result<()> {
    glm_hc_pre(
        g,
        k,
        s.streams,
        &s.w,
        o.p[0],
        o.p[1],
        o.p[2],
        t as u32,
        hid as u32,
        hc as u32,
        SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
        stream,
    )?;
    KernelLaunch::new(g, norm_k)
        .grid([t as u32, 1, 1])
        .block([hid.min(1024) as u32, 1, 1])
        .arg_ptr(o.p[0])
        .arg_ptr(s.norm_w)
        .arg_ptr(o.p[3])
        .arg_u32(hid as u32)
        .arg_f32(RMS_EPS)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    s: &Site,
    o: &Outs,
    t: usize,
    hid: usize,
    hc: usize,
    iters: u32,
    rms_eps: f32,
    stream: u64,
) -> Result<()> {
    fused_arm(g, k, s, o, t, hid, hc, iters, rms_eps, false, false, stream)
}

/// 2026-10-11: The fused launch with the prefetching kernel's choice explicit (`prefetch`), so
/// neither arm depends on `METRALE_GLM_MHC_PRE_PREFETCH`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_arm(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    s: &Site,
    o: &Outs,
    t: usize,
    hid: usize,
    hc: usize,
    iters: u32,
    rms_eps: f32,
    prefetch: bool,
    splitk: bool,
    stream: u64,
) -> Result<()> {
    glm_hc_pre_norm_with(
        g,
        k,
        s.streams,
        &s.w,
        s.norm_w,
        o.p[0],
        o.p[3],
        o.p[1],
        o.p[2],
        t as u32,
        hid as u32,
        hc as u32,
        iters,
        NORM_EPS,
        HC_EPS,
        rms_eps,
        PreNormChoice {
            prefetch,
            splitk,
            tok: false,
            c1: false,
        },
        stream,
    )
}

/// 2026-10-11: Whether the prefetching kernel runs at this case (`pre_norm_kernel` picks it).
pub(super) fn pf_case(k: &Glm5NextMhcKernels, t: usize, hid: usize, hc: usize) -> bool {
    pre_norm_kernel(k, t as u32, hid as u32, hc as u32, true)
        .0
        .0
        == k.hc_pre_norm_pf.0
        && k.hc_pre_norm_pf.0 != 0
}

/// 2026-10-09: Per-site microseconds of `f(stream, i)` for i in 0..REPS captured into one graph, replayed
/// `TRIALS` times: (median, min, max).
pub(super) fn graph_us(
    g: &dyn GpuBackend,
    stream: u64,
    mut f: impl FnMut(u64, usize) -> Result<()>,
) -> Result<(f64, f64, f64)> {
    g.begin_capture(stream)?;
    for i in 0..REPS {
        if let Err(e) = f(stream, i) {
            g.abort_capture_if_active(stream);
            return Err(e);
        }
    }
    let graph = g.end_capture(stream)?;
    g.launch_graph(graph, stream)?;
    g.synchronize(stream)?;
    let mut v = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let t0 = std::time::Instant::now();
        g.launch_graph(graph, stream)?;
        g.synchronize(stream)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    g.destroy_graph(graph)?;
    v.sort_by(|a, b| a.total_cmp(b));
    Ok((v[TRIALS / 2], v[0], v[TRIALS - 1]))
}

/// 2026-10-11: Largest relative error of `y` against `x` (BF16 when `bf16`, else FP32), each
/// element measured against max(|x|, 1e-3).
pub(super) fn max_rel(x: &[u8], y: &[u8], bf16: bool) -> f64 {
    let dec = |v: &[u8]| -> Vec<f64> {
        if bf16 {
            v.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
                .collect()
        } else {
            v.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()
        }
    };
    dec(x)
        .iter()
        .zip(dec(y))
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
        .fold(0.0, f64::max)
}
