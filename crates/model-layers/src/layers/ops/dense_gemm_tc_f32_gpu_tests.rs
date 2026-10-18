// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: GPU check of the FP32-output projection (dense_gemm_tc_f32out.cu) through the
//! launcher production will use, against an FP64 host reference, at the GLM-5.3-Flash cuBLASLt
//! site shapes, the M edges of the GEMV tiers and the tile (1, 16, 32, 33), and N and K off
//! every tile multiple.
//!
//! Three checks per shape:
//! - Exact: entries drawn from {-3..=3}. Every product and every partial sum is an integer of
//!   magnitude at most 9K < 2^24, so FP32 holds each one exactly in any summation order and
//!   under any rounding mode: the output must equal the FP64 reference bit for bit. A dropped,
//!   doubled or misplaced term fails this whatever its size.
//! - Bounded: entries uniform in [-1, 1), rounded to BF16, so the products are exact in FP32
//!   (8-bit by 8-bit significands) and the error is the accumulation's alone. Each path chains at
//!   most `2 ceil(K/16) + 16` FP32 roundings per output (the tile: two per m16n8k16 step, K/16
//!   steps; the GEMV: two per step over K/128 steps per warp, then an 8-way warp reduction).
//!   With unit roundoff u = 2^-23 (truncation allowed, since the MMA's internal adds are not
//!   specified as round-to-nearest), the classical recursive-summation bound is
//!   `|got - ref| <= (2 ceil(K/16) + 16) * 2^-23 * sum_k |a_k * b_k|`.
//! - Parity: rounding the FP32 output to BF16 (nearest-even) gives the BF16 sibling's bits
//!   (dense_gemm_tc, dense_gemv_bf16_tc*), which share the mainloop.
//!
//! Each output buffer has a guard tail and starts at a NaN sentinel, so an unwritten output or a
//! write past M*N fails.
//!
//! `#[ignore]`: it needs a GPU and the built GB10 kernels. Run with:
//! ```text
//! METRALE_TARGET_MODEL=glm-5.3-flash cargo test -p metrale-model-layers \
//!   fp32_projection_matches -- --ignored --nocapture
//! ```
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

const MODEL: &str = "glm-5.3-flash";
const SENTINEL: u32 = 0x7FC0_1234;
const GUARD: usize = 64;
/// 2026-10-17: `(N, K)` of the GLM-5.3-Flash sites: router, DSA wq_b, DSA weights_proj.
const GLM_SITES: [(u32, u32); 3] = [(288, 4096), (4096, 1536), (32, 4096)];
/// 2026-10-17: N and K off every tile multiple (GEMV rows 16, K step 64; tile 16x64, K step 16).
const RAGGED: [(u32, u32, u32); 9] = [
    (1, 1, 64),
    (3, 17, 64),
    (16, 100, 192),
    (32, 65, 128),
    (33, 65, 100),
    (40, 1, 40),
    (5, 63, 72),
    (48, 130, 17),
    (1, 288, 100),
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn small_int(&mut self) -> bf16 {
        bf16::from_f32((self.next() % 7) as f32 - 3.0)
    }
    fn unit(&mut self) -> bf16 {
        bf16::from_f32((self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
    }
}

fn upload(g: &dyn GpuBackend, v: &[bf16]) -> DevicePtr {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(&bytes, p).unwrap();
    p
}

fn sentinel_buffer(g: &dyn GpuBackend, elems: usize, elem_bytes: usize) -> DevicePtr {
    let bytes: Vec<u8> = SENTINEL
        .to_le_bytes()
        .iter()
        .take(elem_bytes)
        .copied()
        .cycle()
        .take((elems + GUARD) * elem_bytes)
        .collect();
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(&bytes, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    g.synchronize(g.default_stream()).unwrap();
    let mut out = vec![0u8; bytes];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

/// 2026-10-17: FP64 `(sum, sum of |terms|)` per output, from the BF16 values the GPU reads.
fn reference(a: &[bf16], w: &[bf16], m: usize, n: usize, k: usize) -> Vec<(f64, f64)> {
    let mut r = Vec::with_capacity(m * n);
    for row in a.chunks_exact(k).take(m) {
        for col in w.chunks_exact(k).take(n) {
            let (mut s, mut abs) = (0f64, 0f64);
            for (x, y) in row.iter().zip(col) {
                let t = f64::from(x.to_f32()) * f64::from(y.to_f32());
                s += t;
                abs += t.abs();
            }
            r.push((s, abs));
        }
    }
    r
}

/// 2026-10-17: The FP32 output through the production launcher, guard tail checked.
fn run_f32(
    g: &dyn GpuBackend,
    kernels: &TcF32Kernels,
    a: DevicePtr,
    w: DevicePtr,
    (m, n, k): (u32, u32, u32),
) -> Vec<f32> {
    let elems = (m * n) as usize;
    let c = sentinel_buffer(g, elems, 4);
    dense_proj_f32_out(g, kernels, a, w, c, m, n, k, g.default_stream()).unwrap();
    let raw = download(g, c, (elems + GUARD) * 4);
    g.free(c).unwrap();
    let words: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert!(
        words[elems..].iter().all(|&x| x == SENTINEL),
        "M={m} N={n} K={k}: write past M*N"
    );
    let unwritten = words[..elems].iter().filter(|&&x| x == SENTINEL).count();
    assert_eq!(
        unwritten, 0,
        "M={m} N={n} K={k}: {unwritten} outputs unwritten"
    );
    words[..elems].iter().map(|&x| f32::from_bits(x)).collect()
}

/// 2026-10-17: The BF16 sibling of the planned path, on the same inputs.
fn run_bf16_sibling(g: &dyn GpuBackend, a: DevicePtr, w: DevicePtr, plan: &TcF32Plan) -> Vec<u16> {
    let (m, n, k) = plan.shape();
    let elems = (m * n) as usize;
    let c = sentinel_buffer(g, elems, 2);
    let launch = match plan.path() {
        TcF32Path::Gemv { kind, grid_x } => {
            KernelLaunch::new(g, g.kernel("dense_gemv_bf16_tc", kind.entry()).unwrap())
                .grid([grid_x, 1, 1])
                .block([256, 1, 1])
        }
        TcF32Path::Tile { grid_x, grid_y } => {
            KernelLaunch::new(g, g.kernel("gemm_tc", "dense_gemm_tc").unwrap())
                .grid([grid_x, grid_y, 1])
                .block([128, 1, 1])
        }
    };
    let launch = launch
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k);
    let launch = match plan.path() {
        TcF32Path::Gemv { .. } => launch.arg_u32(n),
        TcF32Path::Tile { .. } => launch,
    };
    launch.launch(g.default_stream()).unwrap();
    let raw = download(g, c, elems * 2);
    g.free(c).unwrap();
    raw.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// 2026-10-17: All three checks at one shape; returns the largest error / bound ratio.
fn check(g: &dyn GpuBackend, kernels: &TcF32Kernels, rng: &mut Rng, m: u32, n: u32, k: u32) -> f64 {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let plan = TcF32Plan::new(m, n, k).unwrap();

    let ai: Vec<bf16> = (0..mu * ku).map(|_| rng.small_int()).collect();
    let wi: Vec<bf16> = (0..nu * ku).map(|_| rng.small_int()).collect();
    let (ad, wd) = (upload(g, &ai), upload(g, &wi));
    let got = run_f32(g, kernels, ad, wd, (m, n, k));
    for (i, (x, (want, _))) in got.iter().zip(reference(&ai, &wi, mu, nu, ku)).enumerate() {
        assert!(
            f64::from(*x) == want,
            "exact M={m} N={n} K={k} [{}, {}]: got {x}, want {want}",
            i / nu,
            i % nu
        );
    }
    g.free(ad).unwrap();
    g.free(wd).unwrap();

    let ar: Vec<bf16> = (0..mu * ku).map(|_| rng.unit()).collect();
    let wr: Vec<bf16> = (0..nu * ku).map(|_| rng.unit()).collect();
    let (ad, wd) = (upload(g, &ar), upload(g, &wr));
    let got = run_f32(g, kernels, ad, wd, (m, n, k));
    let roundings = 2.0 * f64::from(k.div_ceil(16)) + 16.0;
    let mut worst = 0f64;
    for (i, (x, (want, abs))) in got.iter().zip(reference(&ar, &wr, mu, nu, ku)).enumerate() {
        let err = (f64::from(*x) - want).abs();
        let bound = roundings * 2f64.powi(-23) * abs;
        assert!(
            err <= bound,
            "bounded M={m} N={n} K={k} [{}, {}]: |{x} - {want}| = {err:e} > {bound:e}",
            i / nu,
            i % nu
        );
        if bound > 0.0 {
            worst = worst.max(err / bound);
        }
    }
    let sibling = run_bf16_sibling(g, ad, wd, &plan);
    for (i, (x, s)) in got.iter().zip(&sibling).enumerate() {
        assert_eq!(
            bf16::from_f32(*x).to_bits(),
            *s,
            "parity M={m} N={n} K={k} [{}, {}] ({:?})",
            i / nu,
            i % nu,
            plan.path()
        );
    }
    g.free(ad).unwrap();
    g.free(wd).unwrap();
    println!(
        "M={m:4} N={n:5} K={k:5} {:?}: worst err/bound {worst:.4}",
        plan.path()
    );
    worst
}

#[test]
#[ignore]
fn fp32_projection_matches_an_fp64_reference_at_glm_and_edge_shapes() {
    let set = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model == MODEL && t.ptx_arch.starts_with("sm_121"))
        .expect("build the gb10 glm-5.3-flash kernels first");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let kernels = TcF32Kernels::resolve(g).unwrap();
    let mut rng = Rng(0x05EE_DF32);
    let mut shapes: Vec<(u32, u32, u32)> = Vec::new();
    for (n, k) in GLM_SITES {
        for m in [1, 16, 17, 32, 33, 64] {
            shapes.push((m, n, k));
        }
    }
    shapes.extend([(300, 288, 4096), (129, 4096, 1536), (300, 32, 4096)]);
    shapes.extend(RAGGED);
    let worst = shapes
        .into_iter()
        .map(|(m, n, k)| check(g, &kernels, &mut rng, m, n, k))
        .fold(0f64, f64::max);
    println!("worst err/bound over all shapes: {worst:.4}");
}
