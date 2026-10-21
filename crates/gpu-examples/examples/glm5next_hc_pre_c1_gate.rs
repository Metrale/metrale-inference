// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Bit-identity gate and microbench: `glm5next_hc_pre_norm_c1` (the decode-width
//! pre-norm, `glm5next_mhc_c1.cu`) against `glm5next_hc_pre_norm` (`glm5next_mhc_fused.cu`, itself
//! gated byte-identical to the unfused kernels by `glm5next_hc_fused_gate`).
//!
//! Owner: gpu examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails unless `y`, `post`, `comb` and the normed row are byte-identical between the
//!   two arms for every token count and input family, on three back-to-back launches of the new
//!   kernel (the arrival counters must come back to zero), and fails if the reference left an
//!   output unwritten.
//! - Two negative controls must DIFFER: the new kernel with `CONTROL_ITERS` Sinkhorn iterations
//!   (`comb`) and with a different norm epsilon (the normed row).
//!
//!   cargo build -p metrale-gpu-examples --release --example glm5next_hc_pre_c1_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (H 4096, hc 4): each arm is captured as a CUDA graph over `REPS` distinct sites and
//! replayed `TRIALS` times; the median, min and max per-site microseconds are printed.
//! `--no-timing` skips it.

use anyhow::{Result, bail};
use metrale_gpu_examples::graph_arms::{Plan, time_arms};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, mix_hc};
use metrale_model_arch::glm5next_mhc_c1::c1_kernel;
use metrale_model_arch::glm5next_mhc_fused::{PreNormChoice, glm_hc_pre_norm_with};

const HID: usize = 4096;
const HC: usize = 4;
/// 2026-10-21: The decode widths the kernel serves (the serve's verify is 2), the edges of its
/// range (1, 63) and odd counts.
const TOKENS: [usize; 9] = [1, 2, 3, 4, 7, 8, 16, 31, 63];
const TIMED_TOKENS: [usize; 4] = [1, 2, 4, 16];
const SINKHORN_ITERS: u32 = 20;
const CONTROL_ITERS: u32 = 3;
const CONTROL_RMS_EPS: f32 = 1.0;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;
const RMS_EPS: f32 = 1e-5;
const REPS: usize = 90;
const TRIALS: usize = 9;
const ROUNDS: usize = 5;
const WARM: std::time::Duration = std::time::Duration::from_millis(300);

/// 2026-10-21: Input families. `Random`: uniform streams and small `hc_fn`. `Wide`: streams
/// spanning 2^-20 .. 2^20 with random signs and +-0.0 (the RMS sum and each mix chain see very
/// different magnitudes, so any change of add order shows). `Ties`: streams that are BF16 values
/// plus half an ulp, so the collapse sits on BF16 rounding boundaries. `Peaked`: one huge mix
/// row per comb row, so the softmax max and the Sinkhorn divisions run at their extremes.
#[derive(Clone, Copy, Debug)]
enum Family {
    Random,
    Wide,
    Ties,
    Peaked,
}

const FAMILIES: [Family; 4] = [Family::Random, Family::Wide, Family::Ties, Family::Peaked];

struct Lcg(u64);

impl Lcg {
    /// 2026-10-21: Uniform in [-1, 1). (`glm5next_hc_fused_gate`'s `lcg` spans [-1, 0) only: its
    /// streams and `hc_fn` share a sign, every mix lands near +142 and every sigmoid saturates,
    /// so a wrong `pre` or `post` operand cannot show there.)
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
    fn u(&mut self) -> u32 {
        self.f();
        (self.0 >> 40) as u32
    }
}

fn stream_value(fam: Family, r: &mut Lcg) -> f32 {
    match fam {
        Family::Random | Family::Peaked => r.f() * 3.0,
        Family::Wide => {
            let k = r.u();
            if k.is_multiple_of(61) {
                if k.is_multiple_of(2) { 0.0 } else { -0.0 }
            } else {
                let e = (k % 41) as i32 - 20;
                r.f() * 2f32.powi(e)
            }
        }
        Family::Ties => {
            let b = half::bf16::from_f32(r.f() * 4.0).to_f32();
            let bits = b.to_bits() | 0x8000;
            f32::from_bits(bits)
        }
    }
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

const NAMES: [&str; 4] = ["y", "post", "comb", "normed"];

/// 2026-10-21: One arm's outputs: `y`, `post`, `comb`, normed row.
struct Outs {
    p: [DevicePtr; 4],
    bytes: [usize; 4],
}

impl Outs {
    fn new(g: &dyn GpuBackend, t: usize) -> Result<Self> {
        let bytes = [t * HID * 2, t * HC * 4, t * HC * HC * 4, t * HID * 2];
        let mut p = [DevicePtr(0); 4];
        for (q, &n) in p.iter_mut().zip(&bytes) {
            *q = g.alloc(n)?;
        }
        Ok(Self { p, bytes })
    }
    /// 2026-10-21: 0xAB everywhere, so an unwritten output cannot match the other arm.
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

struct Site {
    streams: DevicePtr,
    norm_w: DevicePtr,
    w: Glm5NextMhcSiteWeights,
}

fn site(g: &dyn GpuBackend, t: usize, fam: Family, salt: u64) -> Result<Site> {
    let m = mix_hc(HC);
    let hc_dim = HC * HID;
    let mut r = Lcg(0x6c31_7f00 ^ (t as u64) << 8 ^ salt << 40 ^ fam as u64);
    let streams: Vec<f32> = (0..t * hc_dim).map(|_| stream_value(fam, &mut r)).collect();
    let hc_fn: Vec<f32> = (0..m * hc_dim).map(|_| r.f() * 0.02).collect();
    let mut hc_base: Vec<f32> = (0..m).map(|_| r.f()).collect();
    if let Family::Peaked = fam {
        for i in 0..HC {
            hc_base[2 * HC + i * HC + (i + 1) % HC] = 60.0;
        }
    }
    let norm_w: Vec<f32> = (0..HID).map(|_| 1.0 + 0.5 * r.f()).collect();
    let mix_tokens = metrale_gpu_examples::mhc_mix_tokens()?;
    Ok(Site {
        streams: up(g, &f32_bytes(&streams))?,
        norm_w: up(g, &bf16_bytes(&norm_w))?,
        w: Glm5NextMhcSiteWeights {
            hc_fn: up(g, &bf16_bytes(&hc_fn))?,
            hc_fn_bf16: true,
            hc_scale: up(g, &f32_bytes(&[0.7, 1.3, 0.9]))?,
            hc_base: up(g, &f32_bytes(&hc_base))?,
            mix: g.alloc(mix_tokens * m * 4)?,
            mix_tokens,
            done: up(g, &vec![0u8; mix_tokens * 4])?,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    s: &Site,
    o: &Outs,
    t: usize,
    iters: u32,
    rms_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, kernel)
        .grid([t as u32, mix_hc(HC) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(s.streams)
        .arg_ptr(s.w.hc_fn)
        .arg_ptr(s.w.hc_scale)
        .arg_ptr(s.w.hc_base)
        .arg_ptr(s.w.mix)
        .arg_ptr(s.w.done)
        .arg_ptr(s.norm_w)
        .arg_ptr(o.p[0])
        .arg_ptr(o.p[1])
        .arg_ptr(o.p[2])
        .arg_ptr(o.p[3])
        .arg_u32(HID as u32)
        .arg_u32(HC as u32)
        .arg_u32(iters)
        .arg_f32(NORM_EPS)
        .arg_f32(HC_EPS)
        .arg_f32(rms_eps)
        .launch(stream)
}

/// 2026-10-21: The production launcher with only the decode-step kernel's lever varied.
fn production(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    s: &Site,
    o: &Outs,
    t: usize,
    c1: bool,
    stream: u64,
) -> Result<()> {
    let choice = PreNormChoice {
        prefetch: false,
        splitk: false,
        tok: false,
        c1,
    };
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
        HID as u32,
        HC as u32,
        SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
        RMS_EPS,
        choice,
        stream,
    )
}

fn first_diff(x: &[u8], y: &[u8]) -> Option<usize> {
    x.iter().zip(y).position(|(p, q)| p != q)
}

fn main() -> Result<()> {
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Glm5NextMhcKernels::resolve(g, metrale_gpu_examples::glm_levers()?.mhc)?;
    let (old, pf, new) = (k.hc_pre_norm, k.hc_pre_norm_pf, k.hc_pre_norm_c1);
    if old.0 == 0 || pf.0 == 0 || new.0 == 0 {
        bail!("glm5next_hc_pre_norm / _pf64 / _c1 missing from this build");
    }
    println!("glm5next hc pre c1 gate: glm5next_hc_pre_norm is the ORACLE (H {HID}, hc {HC})\n");

    let mut fails = 0usize;
    for fam in FAMILIES {
        for t in TOKENS {
            let s = site(g, t, fam, 0)?;
            let a = Outs::new(g, t)?;
            a.poison(g)?;
            launch(g, old, &s, &a, t, SINKHORN_ITERS, RMS_EPS, 0)?;
            g.synchronize(0)?;
            let ah = a.read(g)?;
            // 2026-10-21: The inputs must keep the sigmoids off their rails (not every post at
            // 0 or 2), or the pre and post operands are not exercised.
            let post: Vec<f32> = ah[1]
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if post.iter().all(|&p| !(0.001..1.999).contains(&p)) {
                bail!(
                    "{fam:?} T={t}: post {post:?} sits on a sigmoid rail; the inputs exercise nothing"
                );
            }
            for (n, v) in NAMES.iter().zip(&ah) {
                if v.iter().all(|&x| x == 0xAB) {
                    bail!("{fam:?} T={t}: oracle `{n}` is still poison");
                }
            }
            let b = Outs::new(g, t)?;
            let mut bad = Vec::new();
            for run in 0..3 {
                b.poison(g)?;
                launch(g, new, &s, &b, t, SINKHORN_ITERS, RMS_EPS, 0)?;
                g.synchronize(0)?;
                for ((n, x), y) in NAMES.iter().zip(&ah).zip(&b.read(g)?) {
                    if let Some(i) = first_diff(x, y) {
                        bad.push(format!("run {run} `{n}` byte {i}"));
                    }
                }
            }
            // 2026-10-21: Where the launcher takes the kernel (the decode widths), its output too:
            // the dispatch must reach it and keep the bytes.
            if c1_kernel(&k, t as u32, HID as u32, HC as u32, true).is_some() {
                b.poison(g)?;
                production(g, &k, &s, &b, t, true, 0)?;
                g.synchronize(0)?;
                for ((n, x), y) in NAMES.iter().zip(&ah).zip(&b.read(g)?) {
                    if let Some(i) = first_diff(x, y) {
                        bad.push(format!("glm_hc_pre_norm_with `{n}` byte {i}"));
                    }
                }
            }
            // 2026-10-21: The Sinkhorn of `Peaked` reaches its fixed point in three iterations,
            // and `Wide`'s mean square dwarfs the control epsilon: those controls cannot move.
            let controls = [
                (CONTROL_ITERS, RMS_EPS, 2, !matches!(fam, Family::Peaked)),
                (
                    SINKHORN_ITERS,
                    CONTROL_RMS_EPS,
                    3,
                    !matches!(fam, Family::Wide),
                ),
            ];
            for (iters, eps, out, _) in controls.into_iter().filter(|c| c.3) {
                b.poison(g)?;
                launch(g, new, &s, &b, t, iters, eps, 0)?;
                g.synchronize(0)?;
                if b.read(g)?[out] == ah[out] {
                    bad.push(format!(
                        "control (iters {iters}, rms eps {eps}) `{}` did not differ",
                        NAMES[out]
                    ));
                }
            }
            if bad.is_empty() {
                println!(
                    "  {fam:>7?} T={t:>2}  y/post/comb/normed BYTE-IDENTICAL x3; controls differ"
                );
            } else {
                fails += 1;
                println!("  {fam:>7?} T={t:>2}  FAIL: {}", bad.join("; "));
            }
        }
    }
    if fails > 0 {
        bail!("glm5next hc pre c1 gate FAILED in {fails} cases");
    }

    if timing {
        let stream = g.create_stream()?;
        let plan = Plan {
            calls: REPS,
            rounds: ROUNDS,
            trials: TRIALS,
            warm: WARM,
        };
        println!(
            "\n  TIMING: graph of {REPS} sites, {ROUNDS} interleaved rounds x {TRIALS} replays after \
             {WARM:?} warm-up; us/site median [min, max] over rounds"
        );
        let arms: [(&str, KernelHandle); 3] = [("pre_norm", old), ("pf64", pf), ("c1", new)];
        for t in TIMED_TOKENS {
            let sites: Vec<Site> = (0..REPS)
                .map(|i| site(g, t, Family::Random, i as u64 + 1))
                .collect::<Result<_>>()?;
            let o = Outs::new(g, t)?;
            let fs: Vec<Box<dyn Fn(u64, usize) -> Result<()>>> = arms
                .iter()
                .map(|&(_, kh)| {
                    let (sites, o) = (&sites, &o);
                    Box::new(move |st: u64, i: usize| {
                        launch(g, kh, &sites[i], o, t, SINKHORN_ITERS, RMS_EPS, st)
                    }) as Box<dyn Fn(u64, usize) -> Result<()>>
                })
                .collect();
            let refs: Vec<&dyn Fn(u64, usize) -> Result<()>> =
                fs.iter().map(|f| f.as_ref()).collect();
            let r = time_arms(g, stream, plan, &refs)?;
            print!("    T={t:>2}");
            for ((name, _), u) in arms.iter().zip(&r) {
                print!("  {name} {:6.2} [{:6.2}, {:6.2}]", u.0, u.1, u.2);
            }
            println!(
                "  c1 saves {:.3} ms/step at 90 sites",
                (r[0].0 - r[2].0) * 90.0 / 1000.0
            );
        }
    }
    println!("\nglm5next hc pre c1 gate PASS");
    Ok(())
}
