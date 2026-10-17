// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Bit-identity gate and microbench: `glm5next_hc_pre_norm` (one launch) against the
//! unfused `glm_hc_pre` (`hc_mix_bf16` or `hc_mix_bf16_rows`, then `hc_finish`) followed by
//! `rms_norm_vanilla`, the kernels the GLM layer launches without `METRALE_GLM_MHC_FUSED`.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails unless `y`, `post`, `comb` and the normed row are byte-identical between the
//!   two arms for every case, on each of three back-to-back fused launches (the per-token
//!   arrival counters must come back to zero), and fails if the unfused arm left an output
//!   unwritten.
//! - 2026-10-11: Where `pre_norm_kernel` takes `glm5next_hc_pre_norm_pf64` (H 4096, hc 4, fewer
//!   than 64 tokens) it is held to the same byte identity, three launches, and its own Sinkhorn
//!   control must differ; its time is printed under the fused arm's.
//! - Two negative controls must DIFFER: the fused arm with `CONTROL_ITERS` Sinkhorn iterations
//!   (`comb`), and with a different norm epsilon (the normed row).
//!
//!   cargo run -p metrale-gpu-examples --release --example glm5next_hc_fused_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (GLM-5.3 shape, H 4096, hc 4): each arm is captured as a CUDA graph over `REPS`
//! distinct sites and replayed `TRIALS` times on an otherwise idle stream; the median, min and max per-site
//! microseconds are printed. `--no-timing` skips it.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_pre, mix_hc,
};
use metrale_model_arch::glm5next_mhc_fused::{
    glm_hc_pre_norm_with, pre_norm_kernel, splitk_kernel,
};

/// 2026-10-09: `(hidden, hc_mult, tokens)`. Rows 1, 4 and 16 are the decode widths; at 32 the
/// unfused arm takes its 8-row mix and the fused arm its per-row grid, from 64 both take 8-row
/// grids. The rest vary hidden and `hc_mult`.
const CASES: [(usize, usize, usize); 11] = [
    (4096, 4, 1),
    (4096, 4, 4),
    (4096, 4, 16),
    (4096, 4, 32),
    (4096, 4, 64),
    (4096, 4, 128),
    (5120, 4, 7),
    (4096, 3, 2),
    (1024, 2, 3),
    (256, 4, 1),
    (512, 1, 2),
];
const TIMED_TOKENS: [usize; 5] = [1, 4, 16, 32, 64];
const SINKHORN_ITERS: u32 = 20;
/// 2026-10-09: The Sinkhorn has converged to its fixed point in FP32 by iteration 19 on these
/// inputs, so 19 iterations give the same bytes as 20; the control stops far earlier.
const CONTROL_ITERS: u32 = 3;
/// 2026-10-09: A norm epsilon large enough against mean(y^2) to move every BF16 output.
const CONTROL_RMS_EPS: f32 = 1.0;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;
const RMS_EPS: f32 = 1e-5;
const REPS: usize = 90;
const TRIALS: usize = 9;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn f32_bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_bytes(d: &[f32]) -> Vec<u8> {
    d.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect()
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-09: The four outputs of one site: `y`, `post`, `comb`, normed row.
struct Outs {
    p: [DevicePtr; 4],
    bytes: [usize; 4],
}

const NAMES: [&str; 4] = ["y", "post", "comb", "normed"];

impl Outs {
    fn new(g: &dyn GpuBackend, t: usize, hid: usize, hc: usize) -> Result<Self> {
        let bytes = [t * hid * 2, t * hc * 4, t * hc * hc * 4, t * hid * 2];
        let mut p = [DevicePtr(0); 4];
        for (q, &n) in p.iter_mut().zip(&bytes) {
            *q = g.alloc(n)?;
        }
        Ok(Self { p, bytes })
    }
    /// 2026-10-09: Fill with 0xAB so an unwritten output cannot match the other arm.
    fn poison(&self, g: &dyn GpuBackend) -> Result<()> {
        for (&p, &n) in self.p.iter().zip(&self.bytes) {
            g.copy_h2d(&vec![0xABu8; n], p)?;
        }
        Ok(())
    }
    fn read(&self, g: &dyn GpuBackend) -> Result<Vec<Vec<u8>>> {
        self.p
            .iter()
            .zip(&self.bytes)
            .map(|(&p, &n)| dn(g, p, n))
            .collect()
    }
}

/// 2026-10-09: One site's inputs on the device.
struct Site {
    streams: DevicePtr,
    norm_w: DevicePtr,
    w: Glm5NextMhcSiteWeights,
}

fn site(g: &dyn GpuBackend, hid: usize, hc: usize, t: usize, salt: u64) -> Result<Site> {
    let m = mix_hc(hc);
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
fn unfused(
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
fn fused(
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
fn fused_arm(
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
        g, k, s.streams, &s.w, s.norm_w, o.p[0], o.p[3], o.p[1], o.p[2], t as u32, hid as u32,
        hc as u32, iters, NORM_EPS, HC_EPS, rms_eps, prefetch, splitk, false, stream,
    )
}

/// 2026-10-11: Whether the prefetching kernel runs at this case (`pre_norm_kernel` picks it).
fn pf_case(k: &Glm5NextMhcKernels, t: usize, hid: usize, hc: usize) -> bool {
    pre_norm_kernel(k, t as u32, hid as u32, hc as u32, true)
        .0
        .0
        == k.hc_pre_norm_pf.0
        && k.hc_pre_norm_pf.0 != 0
}

/// 2026-10-09: Per-site microseconds of `f(stream, i)` for i in 0..REPS captured into one graph, replayed
/// `TRIALS` times: (median, min, max).
fn graph_us(
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
fn max_rel(x: &[u8], y: &[u8], bf16: bool) -> f64 {
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

fn main() -> Result<()> {
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Glm5NextMhcKernels::resolve(g, metrale_gpu_examples::glm_levers()?.mhc)?;
    if k.hc_pre_norm.0 == 0 || k.hc_pre_norm_rows.0 == 0 {
        bail!("glm5next_mhc_fused kernels missing from this build");
    }
    let norm_k = g.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    println!(
        "glm5next hc fused gate: unfused hc_mix + hc_finish + rms_norm_vanilla is the ORACLE\n"
    );

    for (hid, hc, t) in CASES {
        let s = site(g, hid, hc, t, 0)?;
        let a = Outs::new(g, t, hid, hc)?;
        a.poison(g)?;
        unfused(g, &k, norm_k, &s, &a, t, hid, hc, 0)?;
        g.synchronize(0)?;
        let ah = a.read(g)?;
        for (n, v) in NAMES.iter().zip(&ah) {
            if v.iter().all(|&x| x == 0xAB) {
                bail!("H={hid} hc={hc} T={t}: oracle `{n}` is still poison");
            }
        }
        let b = Outs::new(g, t, hid, hc)?;
        for run in 0..3 {
            b.poison(g)?;
            fused(g, &k, &s, &b, t, hid, hc, SINKHORN_ITERS, RMS_EPS, 0)?;
            g.synchronize(0)?;
            for ((n, x), y) in NAMES.iter().zip(&ah).zip(&b.read(g)?) {
                if x != y {
                    let i = x.iter().zip(y).position(|(p, q)| p != q).unwrap();
                    bail!(
                        "H={hid} hc={hc} T={t} run {run}: `{n}` differs at byte {i}: \
                         unfused {:#04x} fused {:#04x}",
                        x[i],
                        y[i]
                    );
                }
            }
        }
        // 2026-10-11: The prefetching kernel against the same oracle, three launches, and its
        // own Sinkhorn control, where `pre_norm_kernel` takes it.
        let pf = pf_case(&k, t, hid, hc);
        if pf {
            for run in 0..3 {
                b.poison(g)?;
                fused_arm(
                    g,
                    &k,
                    &s,
                    &b,
                    t,
                    hid,
                    hc,
                    SINKHORN_ITERS,
                    RMS_EPS,
                    true,
                    false,
                    0,
                )?;
                g.synchronize(0)?;
                for ((n, x), y) in NAMES.iter().zip(&ah).zip(&b.read(g)?) {
                    if x != y {
                        let i = x.iter().zip(y).position(|(p, q)| p != q).unwrap();
                        bail!(
                            "H={hid} hc={hc} T={t} run {run}: prefetching `{n}` differs at byte \
                             {i}: unfused {:#04x} prefetching {:#04x}",
                            x[i],
                            y[i]
                        );
                    }
                }
            }
            b.poison(g)?;
            fused_arm(
                g,
                &k,
                &s,
                &b,
                t,
                hid,
                hc,
                CONTROL_ITERS,
                RMS_EPS,
                true,
                false,
                0,
            )?;
            g.synchronize(0)?;
            if b.read(g)?[2] == ah[2] {
                bail!("H={hid} hc={hc} T={t}: prefetching control did not change `comb`");
            }
        }
        // 2026-10-11: The split-k kernel (accuracy class RO): within tolerance of the oracle and
        // byte-identical across three launches (run-to-run identity).
        let sk = splitk_kernel(
            &k,
            t as u32,
            hid as u32,
            hc as u32,
            true,
            metrale_gpu_examples::mhc_mix_tokens()?,
        )
        .is_some();
        let mut sk_err = [0f64; 4];
        if sk {
            let mut first: Option<Vec<Vec<u8>>> = None;
            for run in 0..3 {
                b.poison(g)?;
                fused_arm(
                    g,
                    &k,
                    &s,
                    &b,
                    t,
                    hid,
                    hc,
                    SINKHORN_ITERS,
                    RMS_EPS,
                    false,
                    true,
                    0,
                )?;
                g.synchronize(0)?;
                let got = b.read(g)?;
                for (j, (x, y)) in ah.iter().zip(&got).enumerate() {
                    sk_err[j] = sk_err[j].max(max_rel(x, y, j == 0 || j == 3));
                }
                if let Some(f) = &first {
                    if f != &got {
                        bail!(
                            "H={hid} hc={hc} T={t} run {run}: split-k is not run-to-run identical"
                        );
                    }
                } else {
                    first = Some(got);
                }
            }
            if sk_err[0] > 2e-2 || sk_err[3] > 2e-2 || sk_err[1] > 1e-4 || sk_err[2] > 1e-4 {
                bail!("H={hid} hc={hc} T={t}: split-k outside tolerance {sk_err:?}");
            }
        }
        // 2026-10-09: Negative controls: each must change the bytes it governs.
        b.poison(g)?;
        fused(g, &k, &s, &b, t, hid, hc, CONTROL_ITERS, RMS_EPS, 0)?;
        g.synchronize(0)?;
        // 2026-10-09: At hc_mult 1 comb is the 1x1 matrix 1 whatever the iteration count.
        if hc > 1 && b.read(g)?[2] == ah[2] {
            bail!(
                "H={hid} hc={hc} T={t}: control (Sinkhorn iterations {CONTROL_ITERS}) did not change `comb`"
            );
        }
        b.poison(g)?;
        fused(
            g,
            &k,
            &s,
            &b,
            t,
            hid,
            hc,
            SINKHORN_ITERS,
            CONTROL_RMS_EPS,
            0,
        )?;
        g.synchronize(0)?;
        if b.read(g)?[3] == ah[3] {
            bail!(
                "H={hid} hc={hc} T={t}: control (rms_eps {CONTROL_RMS_EPS}) did not change the normed row"
            );
        }
        println!(
            "  H={hid:>5} hc={hc} T={t:>3}  y/post/comb/normed BYTE-IDENTICAL x3; controls differ{}",
            if pf {
                "; prefetching kernel BYTE-IDENTICAL x3"
            } else {
                ""
            }
        );
        if sk {
            println!(
                "             split-k: max rel err y {:.2e} post {:.2e} comb {:.2e} normed {:.2e}; identical x3",
                sk_err[0], sk_err[1], sk_err[2], sk_err[3]
            );
        }
    }

    if timing {
        let stream = g.create_stream()?;
        let (hid, hc) = (4096usize, 4usize);
        println!(
            "\n  TIMING H={hid} hc={hc}: graph of {REPS} sites x {TRIALS} replays, us/site median [min, max]"
        );
        for t in TIMED_TOKENS {
            // 2026-10-09: REPS distinct sites, as a decode step walks 90 sites whose `hc_fn`
            // (786 KB each at this shape) cannot all stay in L2; one site repeated would time
            // L2-resident weights.
            let sites: Vec<Site> = (0..REPS)
                .map(|i| site(g, hid, hc, t, i as u64 + 1))
                .collect::<Result<_>>()?;
            let o = Outs::new(g, t, hid, hc)?;
            let old = graph_us(g, stream, |st, i| {
                unfused(g, &k, norm_k, &sites[i], &o, t, hid, hc, st)
            })?;
            let pre_only = graph_us(g, stream, |st, i| {
                let s = &sites[i];
                glm_hc_pre(
                    g,
                    &k,
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
                    st,
                )
            })?;
            let new = graph_us(g, stream, |st, i| {
                fused(
                    g,
                    &k,
                    &sites[i],
                    &o,
                    t,
                    hid,
                    hc,
                    SINKHORN_ITERS,
                    RMS_EPS,
                    st,
                )
            })?;
            let new1 = graph_us(g, stream, |st, i| {
                fused(g, &k, &sites[i], &o, t, hid, hc, 1, RMS_EPS, st)
            })?;
            println!(
                "    T={t:>3}  unfused {:6.2} [{:6.2}, {:6.2}] (mix+finish {:6.2})  fused {:6.2} \
                 [{:6.2}, {:6.2}] (1 Sinkhorn iter {:6.2})  x{:.2}  saves {:.3} ms/step at 90 sites",
                old.0,
                old.1,
                old.2,
                pre_only.0,
                new.0,
                new.1,
                new.2,
                new1.0,
                old.0 / new.0,
                (old.0 - new.0) * 90.0 / 1000.0
            );
            if pf_case(&k, t, hid, hc) {
                let pfu = graph_us(g, stream, |st, i| {
                    fused_arm(
                        g,
                        &k,
                        &sites[i],
                        &o,
                        t,
                        hid,
                        hc,
                        SINKHORN_ITERS,
                        RMS_EPS,
                        true,
                        false,
                        st,
                    )
                })?;
                println!(
                    "           prefetching {:6.2} [{:6.2}, {:6.2}]  saves {:.3} ms/step at 90 sites vs fused",
                    pfu.0,
                    pfu.1,
                    pfu.2,
                    (new.0 - pfu.0) * 90.0 / 1000.0
                );
            }
            if splitk_kernel(
                &k,
                t as u32,
                hid as u32,
                hc as u32,
                true,
                metrale_gpu_examples::mhc_mix_tokens()?,
            )
            .is_some()
            {
                let sku = graph_us(g, stream, |st, i| {
                    fused_arm(
                        g,
                        &k,
                        &sites[i],
                        &o,
                        t,
                        hid,
                        hc,
                        SINKHORN_ITERS,
                        RMS_EPS,
                        false,
                        true,
                        st,
                    )
                })?;
                println!(
                    "           split-k     {:6.2} [{:6.2}, {:6.2}]  saves {:.3} ms/step at 90 sites vs fused",
                    sku.0,
                    sku.1,
                    sku.2,
                    (new.0 - sku.0) * 90.0 / 1000.0
                );
            }
        }
    }
    println!("\nfused gate PASS");
    Ok(())
}
