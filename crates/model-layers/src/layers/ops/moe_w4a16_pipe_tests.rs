// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The pipelined Hopper twin `moe_w4a16_grouped_gemm_ptrtable_pipe` writes the same
//! bytes as the base `moe_w4a16_grouped_gemm_ptrtable` on random NVFP4 experts, for the routed
//! prefill's two shapes (gate/up: N 640, K 2560 with the sorted-token gather; down: N 2560,
//! K 640 without it), for uneven expert row counts around the 64-row tile (0, 1, 63, 64, 65,
//! 200) and for a grid y larger than the batch needs. Every output row is written.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_w4a16_pipe_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set and the Hopper kernels.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

const SENTINEL: u8 = 0xAB;
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

/// 2026-10-13: One problem's device buffers: A, the three expert tables, offsets and gather.
struct Problem {
    a: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
    scale2: DevicePtr,
    offsets: DevicePtr,
    sorted: DevicePtr,
    total: u32,
}

fn problem(g: &dyn GpuBackend, n: u32, k: u32, gather: bool) -> Problem {
    let total: u32 = ROWS.iter().sum();
    let a_rows = if gather { TOKENS } else { total };
    let table = |seed: u32, len: usize, scale: bool| -> Vec<u8> {
        (0..ROWS.len() as u32)
            .flat_map(|e| {
                upload(g, &bytes(len, seed + e * 7919, scale))
                    .0
                    .to_le_bytes()
            })
            .collect()
    };
    let (n, k) = (n as usize, k as usize);
    let offsets = std::iter::once(0)
        .chain(ROWS.iter().scan(0, |acc, r| {
            *acc += r;
            Some(*acc)
        }))
        .collect::<Vec<u32>>();
    let gather_rows = bytes(total as usize * 4, 0x9e37_79b9, false)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) % TOKENS)
        .collect::<Vec<u32>>();
    Problem {
        a: upload(g, &bf16s(a_rows as usize * k, 3)),
        packed: upload(g, &table(11, n * k / 2, false)),
        scales: upload(g, &table(13, n * k / 16, true)),
        scale2: upload(
            g,
            &(0..ROWS.len())
                .flat_map(|e| (0.0625 + e as f32 / 64.0).to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
        offsets: upload(g, &u32s(offsets)),
        sorted: if gather {
            upload(g, &u32s(gather_rows))
        } else {
            DevicePtr(0)
        },
        total,
    }
}

/// 2026-10-13: Run `launch` into a sentinel-filled C with grid y `m_tiles`; returns C's bytes.
fn run(
    g: &dyn GpuBackend,
    p: &Problem,
    kernel: metrale_gpu_runtime::gpu::KernelHandle,
    pipe: bool,
    (n, k, m_tiles): (u32, u32, u32),
    s: u64,
) -> Vec<u8> {
    let c_bytes = (p.total * n) as usize * 2;
    let c = sentinel(g, c_bytes);
    let launch = if pipe {
        ops::moe_w4a16_grouped_gemm_ptrtable_pipe
    } else {
        ops::moe_w4a16_grouped_gemm_ptrtable
    };
    launch(
        g,
        kernel,
        p.a,
        p.packed,
        p.scales,
        p.scale2,
        c,
        p.offsets,
        p.sorted,
        ROWS.len() as u32,
        n,
        k,
        m_tiles,
        s,
    )
    .unwrap();
    g.synchronize(s).unwrap();
    download(g, c, c_bytes)
}

#[test]
#[ignore]
fn pipelined_w4a16_grouped_gemm_is_byte_identical_to_the_base_kernel() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let base = g
        .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")
        .unwrap();
    let pipe = g
        .kernel(
            "moe_w4a16_grouped_pipe_hopper",
            "moe_w4a16_grouped_gemm_ptrtable_pipe",
        )
        .expect("the Hopper pipelined twin is not in this build");
    let s = g.create_stream().unwrap();
    let tiles = ROWS.iter().map(|r| r.div_ceil(64)).max().unwrap();
    for (n, k, gather) in SHAPES {
        assert!(ops::moe_w4a16_pipe_admits(g, pipe, n, k), "N={n} K={k}");
        let p = problem(g, n, k, gather);
        let want = run(g, &p, base, false, (n, k, tiles), s);
        let row = n as usize * 2;
        let unwritten = want
            .chunks_exact(row)
            .filter(|r| r.iter().all(|&b| b == SENTINEL))
            .count();
        assert_eq!(unwritten, 0, "N={n} K={k}: base left {unwritten} rows");
        assert!(
            want.chunks_exact(2).any(|b| b != [0, 0]),
            "N={n} K={k}: base wrote only zeros"
        );
        for m_tiles in [tiles, tiles + 2] {
            let got = run(g, &p, pipe, true, (n, k, m_tiles), s);
            let first =
                (0..p.total as usize).find(|&r| got[r * row..][..row] != want[r * row..][..row]);
            assert_eq!(
                first, None,
                "N={n} K={k} gather={gather} grid y={m_tiles}: first differing row"
            );
        }
    }
}
