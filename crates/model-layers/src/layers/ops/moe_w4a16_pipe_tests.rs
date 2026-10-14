// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The pipelined Hopper twin `moe_w4a16_grouped_gemm_ptrtable_pipe` writes the same
//! bytes as the base `moe_w4a16_grouped_gemm_ptrtable` on random NVFP4 experts, for the routed
//! prefill's two shapes (gate/up: N 640, K 2560 with the sorted-token gather; down: N 2560,
//! K 640 without it), for uneven expert row counts around the 64-row tile (0, 1, 63, 64, 65,
//! 200) and for a grid y larger than the batch needs. Every output row is written.
//!
//! 2026-10-14: So does the persistent twin `moe_w4a16_grouped_gemm_ptrtable_persist`, at its own
//! grid and at a grid far smaller than the work (each CTA then strides over many tiles), and on
//! the model's routing shape (512 experts, top-10) at {1, 7, 20, 29, 93, 300, 4096} tokens. That
//! test also prints the time of a routed projection as the prefill ran it before (host copy of
//! the offsets for the exact tile count, then the per-expert grid) and through the persistent
//! grid.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_w4a16_pipe_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set and the Hopper kernels.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

pub(super) const SENTINEL: u8 = 0xAB;
/// 2026-10-13: Rows routed to each expert: empty, single, around the 64-row tile, several tiles.
const ROWS: [u32; 8] = [0, 1, 63, 64, 65, 200, 7, 128];
/// 2026-10-13: Token rows of A the gather draws from.
const TOKENS: u32 = 97;
/// 2026-10-13: (N, K, gather): gate/up and down of the routed prefill.
const SHAPES: [(u32, u32, bool); 2] = [(640, 2560, true), (2560, 640, false)];

/// 2026-10-13: Deterministic bytes; scale bytes stay in E4M3's finite range.
fn bytes(n: usize, seed: u32, scale: bool) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if scale {
                0x28 + (x % 24) as u8
            } else {
                x as u8
            }
        })
        .collect()
}

/// 2026-10-13: BF16 activations in [-2, 2].
fn bf16s(n: usize, seed: u32) -> Vec<u8> {
    bytes(n, seed, false)
        .into_iter()
        .flat_map(|b| {
            let v = b as f32 / 63.75 - 2.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn u32s(v: impl IntoIterator<Item = u32>) -> Vec<u8> {
    v.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn upload(g: &dyn GpuBackend, data: &[u8]) -> DevicePtr {
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

/// 2026-10-14: Random NVFP4 weights of `experts` experts for one (N, K): the three pointer
/// tables and the per-expert buffers they point at.
pub(super) struct Weights {
    packed: DevicePtr,
    scales: DevicePtr,
    scale2: DevicePtr,
    experts: u32,
    buffers: Vec<DevicePtr>,
}

pub(super) fn weights(g: &dyn GpuBackend, n: u32, k: u32, experts: u32) -> Weights {
    let (n, k) = (n as usize, k as usize);
    let mut buffers = Vec::new();
    let mut table = |seed: u32, len: usize, scale: bool| -> Vec<u8> {
        (0..experts)
            .flat_map(|e| {
                let p = upload(g, &bytes(len, seed + e * 7919, scale));
                buffers.push(p);
                p.0.to_le_bytes()
            })
            .collect()
    };
    let packed = table(11, n * k / 2, false);
    let scales = table(13, n * k / 16, true);
    Weights {
        packed: upload(g, &packed),
        scales: upload(g, &scales),
        scale2: upload(
            g,
            &(0..experts)
                .flat_map(|e| (0.0625 + (e % 64) as f32 / 64.0).to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
        experts,
        buffers,
    }
}

impl Weights {
    pub(super) fn free(self, g: &dyn GpuBackend) {
        for p in [self.packed, self.scales, self.scale2]
            .into_iter()
            .chain(self.buffers)
        {
            g.free(p).unwrap();
        }
    }
}

/// 2026-10-13: One problem's device buffers: A, offsets and gather, over a `Weights`.
pub(super) struct Problem<'w> {
    w: &'w Weights,
    a: DevicePtr,
    offsets: DevicePtr,
    sorted: DevicePtr,
    pub(super) total: u32,
    /// 2026-10-14: The largest expert's 64-row tile count (the exact grid y).
    pub(super) tiles: u32,
}

/// 2026-10-14: `rows[e]` expert-sorted rows per expert of `w`; with `gather`, A has `tokens`
/// rows and each sorted row reads a random one of them.
pub(super) fn problem<'w>(
    g: &dyn GpuBackend,
    w: &'w Weights,
    k: u32,
    gather: bool,
    rows: &[u32],
    tokens: u32,
) -> Problem<'w> {
    assert_eq!(rows.len(), w.experts as usize);
    let total: u32 = rows.iter().sum();
    let a_rows = if gather { tokens } else { total };
    let offsets = std::iter::once(0)
        .chain(rows.iter().scan(0, |acc, r| {
            *acc += r;
            Some(*acc)
        }))
        .collect::<Vec<u32>>();
    let gather_rows = bytes(total as usize * 4, 0x9e37_79b9, false)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) % tokens)
        .collect::<Vec<u32>>();
    Problem {
        w,
        a: upload(g, &bf16s(a_rows.max(1) as usize * k as usize, 3)),
        offsets: upload(g, &u32s(offsets)),
        sorted: if gather {
            upload(g, &u32s(gather_rows))
        } else {
            DevicePtr(0)
        },
        total,
        tiles: rows.iter().map(|r| r.div_ceil(64)).max().unwrap_or(0),
    }
}

impl Problem<'_> {
    pub(super) fn free(self, g: &dyn GpuBackend) {
        for p in [self.a, self.offsets, self.sorted] {
            if !p.is_null() {
                g.free(p).unwrap();
            }
        }
    }
}

/// 2026-10-14: The three launchers' common signature.
type Launcher = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> anyhow::Result<()>;

/// 2026-10-14: A launcher and its kernel: the base or per-expert pipelined grid with grid y
/// `m_tiles`, or the persistent grid sized for `total_rows` rows.
#[derive(Clone, Copy)]
pub(super) enum Arm {
    Base(KernelHandle, u32),
    Pipe(KernelHandle, u32),
    Persist(KernelHandle, u32),
    /// 2026-10-14: The wgmma twin, its grid sized for `total_rows` rows like `Persist`.
    Wgmma(KernelHandle, u32),
}

pub(super) fn launch(
    g: &dyn GpuBackend,
    p: &Problem,
    arm: Arm,
    (n, k): (u32, u32),
    c: DevicePtr,
    s: u64,
) {
    let (launcher, kernel, rows) = match arm {
        Arm::Base(kn, t) => (ops::moe_w4a16_grouped_gemm_ptrtable as Launcher, kn, t),
        Arm::Pipe(kn, t) => (ops::moe_w4a16_grouped_gemm_ptrtable_pipe as Launcher, kn, t),
        Arm::Persist(kn, r) => (
            ops::moe_w4a16_grouped_gemm_ptrtable_persist as Launcher,
            kn,
            r,
        ),
        Arm::Wgmma(kn, r) => (
            ops::moe_w4a16_grouped_gemm_ptrtable_wgmma as Launcher,
            kn,
            r,
        ),
    };
    let w = p.w;
    launcher(
        g, kernel, p.a, w.packed, w.scales, w.scale2, c, p.offsets, p.sorted, w.experts, n, k,
        rows, s,
    )
    .unwrap();
}

/// 2026-10-13: Run `arm` into a sentinel-filled C; returns C's bytes.
pub(super) fn run(g: &dyn GpuBackend, p: &Problem, arm: Arm, nk: (u32, u32), s: u64) -> Vec<u8> {
    let c_bytes = (p.total * nk.0).max(1) as usize * 2;
    let c = sentinel(g, c_bytes);
    launch(g, p, arm, nk, c, s);
    g.synchronize(s).unwrap();
    let out = download(g, c, c_bytes);
    g.free(c).unwrap();
    out
}

/// 2026-10-14: The first output row where `got` differs from `want`.
pub(super) fn first_differing_row(want: &[u8], got: &[u8], n: u32) -> Option<usize> {
    let row = n as usize * 2;
    (0..want.len() / row).find(|&r| got[r * row..][..row] != want[r * row..][..row])
}

pub(super) struct Kernels {
    pub(super) base: KernelHandle,
    pipe: KernelHandle,
    pub(super) persist: KernelHandle,
}

pub(super) fn kernels(g: &dyn GpuBackend) -> Kernels {
    let pipe = |f: &str| {
        g.kernel("moe_w4a16_grouped_pipe_hopper", f)
            .expect("the Hopper pipelined twins are not in this build")
    };
    Kernels {
        base: g
            .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")
            .unwrap(),
        pipe: pipe("moe_w4a16_grouped_gemm_ptrtable_pipe"),
        persist: pipe("moe_w4a16_grouped_gemm_ptrtable_persist"),
    }
}

pub(super) fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

#[test]
#[ignore]
fn pipelined_w4a16_grouped_gemm_is_byte_identical_to_the_base_kernel() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let kn = kernels(g);
    let s = g.create_stream().unwrap();
    for (n, k, gather) in SHAPES {
        assert!(ops::moe_w4a16_pipe_admits(g, kn.pipe, n, k), "N={n} K={k}");
        assert!(
            ops::moe_w4a16_pipe_admits(g, kn.persist, n, k),
            "N={n} K={k}"
        );
        let w = weights(g, n, k, ROWS.len() as u32);
        let p = problem(g, &w, k, gather, &ROWS, TOKENS);
        let want = run(g, &p, Arm::Base(kn.base, p.tiles), (n, k), s);
        let unwritten = want
            .chunks_exact(n as usize * 2)
            .filter(|r| r.iter().all(|&b| b == SENTINEL))
            .count();
        assert_eq!(unwritten, 0, "N={n} K={k}: base left {unwritten} rows");
        assert!(
            want.chunks_exact(2).any(|b| b != [0, 0]),
            "N={n} K={k}: base wrote only zeros"
        );
        // 2026-10-14: `Persist(_, 1)` sizes the grid for one row: a handful of CTAs, each
        // striding over many tiles.
        for arm in [
            Arm::Pipe(kn.pipe, p.tiles),
            Arm::Pipe(kn.pipe, p.tiles + 2),
            Arm::Persist(kn.persist, p.total),
            Arm::Persist(kn.persist, 1),
        ] {
            let got = run(g, &p, arm, (n, k), s);
            let first = first_differing_row(&want, &got, n);
            assert_eq!(
                first, None,
                "N={n} K={k} gather={gather}: first differing row"
            );
        }
        p.free(g);
        w.free(g);
    }
}

/// 2026-10-14: The model's routed experts and top-k.
pub(super) const EXPERTS: u32 = 512;
const TOP_K: u32 = 10;
/// 2026-10-14: Prefill lengths: decode-width batches, short prompts, one full 4096-token chunk.
const ROUTED_TOKENS: [u32; 7] = [1, 7, 20, 29, 93, 300, 4096];

/// 2026-10-14: Rows per expert when each token picks TOP_K distinct pseudo-random experts.
pub(super) fn routed_rows(tokens: u32) -> Vec<u32> {
    let mut rows = vec![0u32; EXPERTS as usize];
    let mut x = 0x2545_f491u32 ^ tokens;
    for _ in 0..tokens {
        let mut picked = Vec::with_capacity(TOP_K as usize);
        while picked.len() < TOP_K as usize {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let e = x % EXPERTS;
            if !picked.contains(&e) {
                picked.push(e);
                rows[e as usize] += 1;
            }
        }
    }
    rows
}

/// 2026-10-14: Mean wall time of `f` in microseconds over `iters` calls, after one warm-up, with
/// the stream drained at both ends.
pub(super) fn time_us(g: &dyn GpuBackend, s: u64, iters: u32, f: &dyn Fn()) -> f64 {
    f();
    g.synchronize(s).unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    g.synchronize(s).unwrap();
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[test]
#[ignore]
fn persistent_w4a16_grouped_gemm_matches_the_base_kernel_on_routed_shapes() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let kn = kernels(g);
    let s = g.create_stream().unwrap();
    for (n, k, gather) in SHAPES {
        let w = weights(g, n, k, EXPERTS);
        for tokens in ROUTED_TOKENS {
            let p = problem(g, &w, k, gather, &routed_rows(tokens), tokens);
            let want = run(g, &p, Arm::Base(kn.base, p.tiles), (n, k), s);
            let got = run(g, &p, Arm::Persist(kn.persist, p.total), (n, k), s);
            let first = first_differing_row(&want, &got, n);
            assert_eq!(first, None, "T={tokens} N={n} K={k}: first differing row");
            let c = sentinel(g, (p.total * n) as usize * 2);
            let iters = if tokens >= 1024 { 5 } else { 50 };
            let exact = time_us(g, s, iters, &|| {
                let mut raw = vec![0u8; (w.experts as usize + 1) * 4];
                g.copy_d2h_on_stream(p.offsets, &mut raw, s).unwrap();
                let tiles = raw
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<u32>>()
                    .windows(2)
                    .map(|w| (w[1] - w[0]).div_ceil(64))
                    .max()
                    .unwrap_or(0)
                    .max(1);
                launch(g, &p, Arm::Pipe(kn.pipe, tiles), (n, k), c, s);
            });
            let persist = time_us(g, s, iters, &|| {
                launch(g, &p, Arm::Persist(kn.persist, p.total), (n, k), c, s)
            });
            println!(
                "T={tokens:5} N={n:4} K={k:4} rows={:6} max tiles={:3}: exact-tiles copy + pipe \
                 {exact:9.1} us, persistent {persist:9.1} us",
                p.total, p.tiles
            );
            g.free(c).unwrap();
            p.free(g);
        }
        w.free(g);
    }
}
