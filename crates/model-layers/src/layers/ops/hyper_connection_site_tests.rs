// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate of the one-launch hyper-connection site (`hc_lowrank_site`,
//! METRALE_HC_SITE_FUSED): through `hc_pre_gemm`, its `normed`, `low`, `inj` and `y` are the
//! bytes of the three-launch route (`hc_pre_stage_bf16`, then the two decode GEMVs) at T in
//! {1, 2, 3, 8, 13, 16}, with and without the injection, on the model's shape and smaller ones;
//! a token's bytes alone equal its bytes inside a 16- or 13-token launch; a graph replayed many
//! times (so the in-kernel counters are reset between launches) gives the same bytes; and the
//! admission refuses the shapes the kernel traps on. Timing: `hyper_connection_site_bench_tests.rs`.
//!
//! GPU test: run with
//! ```text
//! METRALE_HC_SITE_FUSED=1 cargo test -p metrale-model-layers --release hyper_connection_site \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//! on a Hopper host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank::gemv::{HcGemv, HcGemvArgs};
use super::hyper_connection_lowrank::hc_pre_gemm;
use super::hyper_connection_lowrank::site::{HC_SITE_MAX_T, HcSite, HcSiteArgs};
use crate::layers::qwen3_attention::HcLowRank;

const SENTINEL: u8 = 0xAB;
pub(super) const EPS: f32 = 1e-6;
/// 2026-10-14: (H, hc, rank): the model's site, then smaller ones (3 streams, 2 streams).
const SHAPES: [(u32, u32, u32); 3] = [(2560, 4, 320), (2048, 3, 128), (1024, 2, 64)];
const WIDTHS: [u32; 6] = [1, 2, 3, 8, 13, 16];

/// 2026-10-14: Deterministic values in [-scale, scale); BF16-representable with `bf16`.
pub(super) fn values(n: usize, seed: u32, scale: f32, bf16: bool) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = ((x % 1_000_003) as f32 / 500_001.5 - 1.0) * scale;
            if bf16 { bf16_round(v) } else { v }
        })
        .collect()
}

fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub(super) fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

pub(super) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub(super) fn upload(g: &dyn GpuBackend, data: &[u8]) -> DevicePtr {
    let p = g.alloc(data.len()).unwrap();
    g.copy_h2d(data, p).unwrap();
    p
}

pub(super) fn sentinel(g: &dyn GpuBackend, n: usize) -> DevicePtr {
    let p = g.alloc(n).unwrap();
    g.memset(p, SENTINEL, n).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

/// 2026-10-14: One site's weights on the device, at the model's magnitudes: per-stream norm
/// scales near zero (the checkpoint stores `1 + w`), down and injection rows small, up rows larger.
pub(super) struct Weights {
    pub h: u32,
    pub hc: u32,
    pub rank: u32,
    pub w: HcLowRank,
}

pub(super) fn weights(g: &dyn GpuBackend, (h, hc, rank): (u32, u32, u32), seed: u32) -> Weights {
    let k = (hc * h) as usize;
    let up =
        |n: usize, s: u32, scale: f32| upload(g, &bf16_bytes(&values(n, seed + s, scale, true)));
    Weights {
        h,
        hc,
        rank,
        w: HcLowRank {
            norm_w: up(k, 1, 0.1),
            down_w: up(rank as usize * k, 3, 0.04),
            up_w: up(k * rank as usize, 5, 0.25),
            inject_w: up(hc as usize * k, 7, 0.04),
            rank: rank as usize,
        },
    }
}

/// 2026-10-14: FP32 residual streams [T, hc, H]; each stream at its own magnitude, as the
/// hyper-connection streams drift apart over the layers.
pub(super) fn streams(g: &dyn GpuBackend, w: &Weights, t: u32, seed: u32) -> DevicePtr {
    let h = w.h as usize;
    let mut v = values(t as usize * w.hc as usize * h, seed, 1.0, false);
    for (i, x) in v.iter_mut().enumerate() {
        *x *= 0.5 + 1.5 * ((i / h) % w.hc as usize) as f32;
    }
    upload(g, &f32_bytes(&v))
}

/// 2026-10-14: Output bytes: normed [T, hc*H] BF16, low [T, rank] BF16, y [T, H] BF16,
/// inj [T, hc] FP32.
#[derive(PartialEq, Debug)]
struct Out {
    normed: Vec<u8>,
    low: Vec<u8>,
    y: Vec<u8>,
    inj: Vec<u8>,
}

/// 2026-10-14: Device outputs of one route for T tokens, sentinel-filled. `scratch` has
/// `hc_pre_gemm`'s layout at L = T: normed, up_pre, low, inj_pre.
pub(super) struct Bufs {
    pub scratch: DevicePtr,
    pub y: DevicePtr,
    pub inj: DevicePtr,
    t: u32,
}

impl Bufs {
    pub(super) fn new(g: &dyn GpuBackend, w: &Weights, t: u32) -> Self {
        let kd = (w.hc * w.h * t) as usize;
        Self {
            scratch: sentinel(
                g,
                4 * kd + (t * w.rank) as usize * 2 + (t * w.hc) as usize * 2,
            ),
            y: sentinel(g, (t * w.h) as usize * 2),
            inj: sentinel(g, (t * w.hc) as usize * 4),
            t,
        }
    }

    pub(super) fn low(&self, w: &Weights) -> DevicePtr {
        self.scratch.offset(4 * (w.hc * w.h * self.t) as usize)
    }

    fn read(&self, g: &dyn GpuBackend, w: &Weights) -> Out {
        let t = self.t;
        Out {
            normed: download(g, self.scratch, (t * w.hc * w.h) as usize * 2),
            low: download(g, self.low(w), (t * w.rank) as usize * 2),
            y: download(g, self.y, (t * w.h) as usize * 2),
            inj: download(g, self.inj, (t * w.hc) as usize * 4),
        }
    }
}

/// 2026-10-14: The three-launch route of `hc_pre_gemm` at decode widths.
pub(super) fn launch_current(
    g: &dyn GpuBackend,
    gemv: &HcGemv,
    w: &Weights,
    streams: DevicePtr,
    b: &Bufs,
    inject: bool,
    stream: u64,
) {
    let stage = g.kernel("hyper_connection", "hc_pre_stage_bf16").unwrap();
    KernelLaunch::new(g, stage)
        .grid([b.t, w.hc, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.w.norm_w)
        .arg_ptr(b.scratch)
        .arg_u32(w.h)
        .arg_u32(w.hc)
        .arg_f32(EPS)
        .launch(stream)
        .unwrap();
    let a = HcGemvArgs {
        normed: b.scratch,
        down_w: w.w.down_w,
        up_w: w.w.up_w,
        inject_w: if inject {
            w.w.inject_w
        } else {
            DevicePtr::NULL
        },
        low: b.low(w),
        y_out: b.y,
        inj_out: b.inj,
    };
    assert!(
        gemv.admits(&a, b.t, w.h, w.hc, w.rank),
        "GEMVs admit T={}",
        b.t
    );
    gemv.launch(g, &a, b.t, w.h, w.hc, w.rank, stream).unwrap();
}

/// 2026-10-14: The site kernel, as `hc_pre_gemm` launches it (partials in the `up_pre` rows).
pub(super) fn site_args(w: &Weights, streams: DevicePtr, b: &Bufs, inject: bool) -> HcSiteArgs {
    HcSiteArgs {
        streams,
        norm_w: w.w.norm_w,
        down_w: w.w.down_w,
        inject_w: if inject {
            w.w.inject_w
        } else {
            DevicePtr::NULL
        },
        up_w: w.w.up_w,
        normed: b.scratch,
        partial: b.scratch.offset(2 * (w.hc * w.h * b.t) as usize),
        low: b.low(w),
        y_out: b.y,
        inj_out: b.inj,
    }
}

fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

/// 2026-10-14: The site kernel resolved; fails loudly when the switch is off, since the fused
/// route would then silently be the current one and every comparison would pass.
pub(super) fn resolved_site(g: &dyn GpuBackend) -> HcSite {
    assert_eq!(
        std::env::var("METRALE_HC_SITE_FUSED").as_deref(),
        Ok("1"),
        "run with METRALE_HC_SITE_FUSED=1"
    );
    HcSite::resolve(g)
}

/// 2026-10-14: The fused route through the production entry point, `hc_pre_gemm`.
fn run_fused(g: &dyn GpuBackend, w: &Weights, streams: DevicePtr, t: u32, inject: bool) -> Out {
    let b = Bufs::new(g, w, t);
    let stream = g.create_stream().unwrap();
    hc_pre_gemm(
        g, streams, &w.w, b.y, b.inj, b.scratch, t, w.h, w.hc, EPS, inject, true, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    b.read(g, w)
}

fn run_current(
    g: &dyn GpuBackend,
    gemv: &HcGemv,
    w: &Weights,
    streams: DevicePtr,
    t: u32,
    inject: bool,
) -> Out {
    let b = Bufs::new(g, w, t);
    let stream = g.create_stream().unwrap();
    launch_current(g, gemv, w, streams, &b, inject, stream);
    g.synchronize(stream).unwrap();
    b.read(g, w)
}

fn assert_written(label: &str, o: &Out, inject: bool) {
    for (name, bytes, width) in [("normed", &o.normed, 2), ("low", &o.low, 2), ("y", &o.y, 2)] {
        let unwritten = bytes
            .chunks_exact(width)
            .filter(|c| c.iter().all(|&b| b == SENTINEL))
            .count();
        assert_eq!(unwritten, 0, "{label} {name}: unwritten elements");
    }
    let inj_left = o.inj.iter().filter(|&&b| b == SENTINEL).count();
    if inject {
        assert!(inj_left < o.inj.len(), "{label}: inj unwritten");
    } else {
        assert_eq!(
            inj_left,
            o.inj.len(),
            "{label}: inj written without an injection weight"
        );
    }
}

#[test]
#[ignore]
fn hc_site_fused_is_byte_identical_to_the_three_launch_route() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let site = resolved_site(g);
    let gemv = HcGemv::resolve(g);
    for (i, shape) in SHAPES.into_iter().enumerate() {
        let w = weights(g, shape, 100 * i as u32);
        for t in WIDTHS {
            let x = streams(g, &w, t, 1000 + t);
            for inject in [true, false] {
                let label = format!("{shape:?} T={t} inject={inject}");
                let b = Bufs::new(g, &w, t);
                let a = site_args(&w, x, &b, inject);
                let cap = 2 * (w.hc * w.h * t) as usize;
                assert!(
                    site.admits(&a, t, w.h, w.hc, w.rank, cap),
                    "{label}: admits"
                );
                let want = run_current(g, &gemv, &w, x, t, inject);
                let got = run_fused(g, &w, x, t, inject);
                assert_written(&label, &got, inject);
                assert!(got.normed == want.normed, "{label}: normed differs");
                assert!(got.low == want.low, "{label}: low differs");
                assert!(got.inj == want.inj, "{label}: inj differs");
                assert!(got.y == want.y, "{label}: y differs");
            }
        }
    }
}

#[test]
#[ignore]
fn hc_site_fused_rows_are_batch_invariant_and_replay_identically() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    resolved_site(g);
    let w = weights(g, SHAPES[0], 7);
    let (kd, h, hc, rank) = (
        (w.hc * w.h) as usize,
        w.h as usize,
        w.hc as usize,
        w.rank as usize,
    );
    for t in [HC_SITE_MAX_T, 13] {
        let x = streams(g, &w, t, 77 + t);
        let full = run_fused(g, &w, x, t, true);
        for r in 0..t as usize {
            let one = run_fused(g, &w, x.offset(r * kd * 4), 1, true);
            assert_eq!(
                one.normed,
                full.normed[r * kd * 2..][..kd * 2],
                "T={t} r={r}: normed"
            );
            assert_eq!(
                one.low,
                full.low[r * rank * 2..][..rank * 2],
                "T={t} r={r}: low"
            );
            assert_eq!(one.y, full.y[r * h * 2..][..h * 2], "T={t} r={r}: y");
            assert_eq!(
                one.inj,
                full.inj[r * hc * 4..][..hc * 4],
                "T={t} r={r}: inj"
            );
        }

        // 2026-10-14: 50 launches in one graph, the graph replayed twice: every launch must leave
        // the counters at zero for the next, or a later launch traps, hangs or differs.
        let b = Bufs::new(g, &w, t);
        let stream = g.create_stream().unwrap();
        g.begin_capture(stream).unwrap();
        for _ in 0..50 {
            hc_pre_gemm(
                g, x, &w.w, b.y, b.inj, b.scratch, t, w.h, w.hc, EPS, true, true, stream,
            )
            .unwrap();
        }
        let graph = g.end_capture(stream).unwrap();
        for _ in 0..2 {
            g.launch_graph(graph, stream).unwrap();
        }
        g.synchronize(stream).unwrap();
        g.destroy_graph(graph).unwrap();
        assert!(b.read(g, &w) == full, "T={t}: graph replay differs");
    }
}

#[test]
#[ignore]
fn hc_site_admission_refuses_what_the_kernel_traps_on() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let site = resolved_site(g);
    let w = weights(g, SHAPES[0], 3);
    let b = Bufs::new(g, &w, HC_SITE_MAX_T);
    let a = site_args(&w, DevicePtr::NULL, &b, true);
    let cap = 2 * (w.hc * w.h * HC_SITE_MAX_T) as usize;
    assert!(site.admits(&a, HC_SITE_MAX_T, 2560, 4, 320, cap));
    assert!(
        !site.admits(&a, HC_SITE_MAX_T + 1, 2560, 4, 320, cap),
        "T cap"
    );
    assert!(!site.admits(&a, 0, 2560, 4, 320, cap), "T = 0");
    assert!(!site.admits(&a, 1, 2560, 5, 320, cap), "hc > 4");
    assert!(!site.admits(&a, 1, 2576, 4, 320, cap), "hc * H % 2048");
    assert!(!site.admits(&a, 1, 8192, 4, 320, cap), "hc * H > 16384");
    assert!(!site.admits(&a, 1, 2560, 4, 520, cap), "rank > 512");
    assert!(!site.admits(&a, 1, 2560, 4, 324, cap), "rank % 8");
    assert!(
        !site.admits(&a, 4, 2560, 4, 320, 8 * 324 * 4 * 4 - 1),
        "partials overflow"
    );
    let mut m = site_args(&w, DevicePtr::NULL, &b, true);
    m.low = m.low.offset(2);
    assert!(!site.admits(&m, 1, 2560, 4, 320, cap), "misaligned low");
}
