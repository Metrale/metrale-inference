// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The Hopper decode GEMVs of the low-rank hyper-connection collapse
//! (`hc_lowrank_down_inject`, `hc_lowrank_up_mix`) against a CPU reference that rounds where the
//! kernels round (projections to BF16, `low` and `y` to BF16), within a tolerance for their
//! summation order and `__expf`; each token's outputs are the same bytes at T = 1 as inside a
//! T = 13 batch (batch invariance, partial token groups of both kernels included) and on a second run; the
//! injection-free launch writes the same `low` and `y` and leaves `inj_out` alone; and the
//! admission refuses the shapes the kernels trap on.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release hyper_connection_gemv_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set and the Hopper kernels.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::hyper_connection_lowrank::gemv::{HC_GEMV_MAX_T, HcGemv, HcGemvArgs};

const SENTINEL: u8 = 0xAB;
const T: u32 = 13;
/// 2026-10-14: (H, hc, rank): the model's site, and a smaller one with 3 streams.
const SHAPES: [(u32, u32, u32); 2] = [(2560, 4, 320), (2000, 3, 64)];

/// 2026-10-14: Deterministic BF16-representable values in [-scale, scale].
fn values(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            bf16_round(((x % 4001) as f32 / 2000.0 - 1.0) * scale)
        })
        .collect()
}

/// 2026-10-14: FP32 to BF16, round to nearest even (the device's `__float2bfloat16`).
fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn from_bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn from_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn upload(g: &dyn GpuBackend, data: &[u8]) -> DevicePtr {
    let p = g.alloc(data.len()).unwrap();
    g.copy_h2d(data, p).unwrap();
    p
}

fn sentinel(g: &dyn GpuBackend, n: usize) -> DevicePtr {
    let p = g.alloc(n).unwrap();
    g.memset(p, SENTINEL, n).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum::<f64>() as f32
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// 2026-10-14: |got - want| <= 2% of |want| + `abs`, element by element. The slack covers one
/// BF16 step of a projection rounded the other way (the summation orders differ), carried
/// through the activation; a wrong row, stream, scale or activation is off by far more.
fn assert_close(label: &str, got: &[f32], want: &[f32], abs: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let bad = got
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (g, w))| !g.is_finite() || (*g - *w).abs() > 0.02 * w.abs() + abs);
    assert_eq!(bad, None, "{label}: first element out of tolerance");
    assert!(
        want.iter().any(|w| w.abs() > 0.05),
        "{label}: reference is near zero everywhere"
    );
}

/// 2026-10-14: One site's weights and inputs, host copies and device buffers.
struct Site {
    h: u32,
    hc: u32,
    rank: u32,
    normed: Vec<f32>,
    down: Vec<f32>,
    up: Vec<f32>,
    inject: Vec<f32>,
    d_normed: DevicePtr,
    d_down: DevicePtr,
    d_up: DevicePtr,
    d_inject: DevicePtr,
}

/// 2026-10-14: Output bytes of one launch: low [T, rank] BF16, y [T, H] BF16, inj [T, hc] FP32.
struct Out {
    low: Vec<u8>,
    y: Vec<u8>,
    inj: Vec<u8>,
}

fn site(g: &dyn GpuBackend, (h, hc, rank): (u32, u32, u32)) -> Site {
    let k = (hc * h) as usize;
    let normed = values(T as usize * k, 3, 2.0);
    let down = values(rank as usize * k, 5, 0.04);
    let up = values(k * rank as usize, 7, 0.25);
    let inject = values(hc as usize * k, 11, 0.04);
    Site {
        h,
        hc,
        rank,
        d_normed: upload(g, &bf16_bytes(&normed)),
        d_down: upload(g, &bf16_bytes(&down)),
        d_up: upload(g, &bf16_bytes(&up)),
        d_inject: upload(g, &bf16_bytes(&inject)),
        normed,
        down,
        up,
        inject,
    }
}

/// 2026-10-14: Launch tokens [t0, t0 + nt) into sentinel-filled outputs.
fn run(g: &dyn GpuBackend, k: &HcGemv, s: &Site, t0: u32, nt: u32, inject: bool) -> Out {
    let kd = (s.hc * s.h) as usize;
    let (n_low, n_y, n_inj) = (
        (nt * s.rank) as usize * 2,
        (nt * s.h) as usize * 2,
        (nt * s.hc) as usize * 4,
    );
    let a = HcGemvArgs {
        normed: s.d_normed.offset(t0 as usize * kd * 2),
        down_w: s.d_down,
        up_w: s.d_up,
        inject_w: if inject { s.d_inject } else { DevicePtr::NULL },
        low: sentinel(g, n_low),
        y_out: sentinel(g, n_y),
        inj_out: sentinel(g, n_inj),
    };
    assert!(k.admits(&a, nt, s.h, s.hc, s.rank), "admits {nt} tokens");
    let stream = g.create_stream().unwrap();
    k.launch(g, &a, nt, s.h, s.hc, s.rank, stream).unwrap();
    g.synchronize(stream).unwrap();
    Out {
        low: download(g, a.low, n_low),
        y: download(g, a.y_out, n_y),
        inj: download(g, a.inj_out, n_inj),
    }
}

/// 2026-10-14: The reference for `low` and `inj` from the inputs, and for `y` from the kernel's
/// own `low` (so the up GEMV is checked on its actual input).
fn check_reference(s: &Site, o: &Out) {
    let (kd, rank, h, hc) = (
        (s.hc * s.h) as usize,
        s.rank as usize,
        s.h as usize,
        s.hc as usize,
    );
    let inv_hc = 1.0f32 / s.hc as f32;
    let low_gpu = from_bf16(&o.low);
    let (mut low, mut inj, mut y) = (vec![], vec![], vec![]);
    for t in 0..T as usize {
        let x = &s.normed[t * kd..][..kd];
        for r in 0..rank {
            let p = bf16_round(dot(x, &s.down[r * kd..][..kd])) * inv_hc;
            low.push(bf16_round(p * sigmoid(p)));
        }
        for i in 0..hc {
            let p = bf16_round(dot(x, &s.inject[i * kd..][..kd])) * inv_hc;
            inj.push(2.0 * sigmoid(p));
        }
        let lt = &low_gpu[t * rank..][..rank];
        for d in 0..h {
            let mixed = (0..hc)
                .map(|st| {
                    let u = bf16_round(dot(lt, &s.up[(st * h + d) * rank..][..rank]));
                    sigmoid(u) * x[st * h + d]
                })
                .fold(0.0f32, |acc, v| acc + v);
            y.push(bf16_round(mixed * inv_hc));
        }
    }
    assert_close("low", &low_gpu, &low, 1e-3);
    assert_close("inj", &from_f32(&o.inj), &inj, 1e-3);
    assert_close("y", &from_bf16(&o.y), &y, 2e-3);
}

#[test]
#[ignore]
fn hc_lowrank_gemvs_match_the_reference_and_are_batch_invariant() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let k = HcGemv::resolve(g);
    for shape in SHAPES {
        let s = site(g, shape);
        let full = run(g, &k, &s, 0, T, true);
        for (label, bytes, width) in [
            ("low", &full.low, 2),
            ("y", &full.y, 2),
            ("inj", &full.inj, 4),
        ] {
            let unwritten = bytes
                .chunks_exact(width)
                .filter(|c| c.iter().all(|&b| b == SENTINEL))
                .count();
            assert_eq!(unwritten, 0, "{shape:?} {label}: unwritten elements");
        }
        check_reference(&s, &full);

        let again = run(g, &k, &s, 0, T, true);
        assert!(
            again.low == full.low && again.y == full.y && again.inj == full.inj,
            "{shape:?}: second run differs"
        );
        for t in 0..T {
            let one = run(g, &k, &s, t, 1, true);
            let row = |b: &[u8], w: u32| b[(t * w) as usize..][..w as usize].to_vec();
            assert_eq!(one.low, row(&full.low, s.rank * 2), "{shape:?} t={t}: low");
            assert_eq!(one.y, row(&full.y, s.h * 2), "{shape:?} t={t}: y");
            assert_eq!(one.inj, row(&full.inj, s.hc * 4), "{shape:?} t={t}: inj");
        }
        let head = run(g, &k, &s, 0, T, false);
        assert_eq!(head.low, full.low, "{shape:?}: low without the injection");
        assert_eq!(head.y, full.y, "{shape:?}: y without the injection");
        assert!(
            head.inj.iter().all(|&b| b == SENTINEL),
            "{shape:?}: inj_out written without an injection weight"
        );
    }

    let s = site(g, SHAPES[0]);
    let args = |normed: DevicePtr| HcGemvArgs {
        normed,
        down_w: s.d_down,
        up_w: s.d_up,
        inject_w: s.d_inject,
        low: s.d_normed,
        y_out: s.d_normed,
        inj_out: s.d_normed,
    };
    let a = args(s.d_normed);
    assert!(k.admits(&a, HC_GEMV_MAX_T, 2560, 4, 320));
    assert!(!k.admits(&a, HC_GEMV_MAX_T + 1, 2560, 4, 320), "T cap");
    assert!(!k.admits(&a, 0, 2560, 4, 320), "T = 0");
    assert!(!k.admits(&a, 1, 2560, 5, 320), "hc > 4");
    assert!(!k.admits(&a, 1, 2560, 4, 520), "rank > 512");
    assert!(!k.admits(&a, 1, 2560, 4, 324), "rank % 8");
    assert!(!k.admits(&a, 1, 2561, 4, 320), "hc * H % 8");
    assert!(
        !k.admits(&args(s.d_normed.offset(2)), 1, 2560, 4, 320),
        "misaligned normed"
    );
}
