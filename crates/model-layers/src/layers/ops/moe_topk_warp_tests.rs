// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `moe_topk_softmax_rows_warp` writes `moe_topk_softmax_rows`' bytes: indices and
//! weights, normalized or not, for 512 experts and top 10 (the Flash-Next router), 256/8, fewer
//! experts than MAX_EXPERTS (padding), more (only the first 512 read), top_k above the expert
//! count (the entries past it left unwritten), a row count that leaves a block part-empty, rows
//! of tied logits (the lower-index tie-break) and a row at or below the -1e30 sentinel (the
//! pseudo pick of expert 0).
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_topk_warp_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{moe_topk_softmax_batched, moe_topk_softmax_rows_warp};

const SENTINEL: u8 = 0xAB;
/// 2026-10-14: (experts, top_k): the Flash-Next router, a 256/8 router, 100 experts (padding),
/// 600 (only the first 512 read), and top_k above the expert count.
const SHAPES: [(u32, u32); 5] = [(512, 10), (256, 8), (100, 6), (600, 10), (5, 8)];
const ROWS: u32 = 13;

/// 2026-10-14: Row `r`'s BF16 logits: row 0 all equal, row 1 at the BF16 minimum (below the
/// sentinel), row 2 in 8 coarse levels (many ties), the rest spread over 2001 levels.
fn logits(rows: u32, e: u32, seed: u32) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity((rows * e * 2) as usize);
    for r in 0..rows {
        for _ in 0..e {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v: f32 = match r {
                0 => 0.75,
                1 => f32::from_bits(0xff7f_0000),
                2 => (x % 8) as f32 * 0.5 - 2.0,
                _ => (x % 2001) as f32 / 250.0 - 4.0,
            };
            out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    out
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

/// 2026-10-14: (indices, weights) bytes of one launch over `rows` rows.
fn route(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    warp: bool,
    d_logits: DevicePtr,
    (e, k): (u32, u32),
    rows: u32,
    normalize: bool,
) -> (Vec<u8>, Vec<u8>) {
    let n = (rows * k * 4) as usize;
    let (idx, w) = (sentinel(g, n), sentinel(g, n));
    let launch = if warp {
        moe_topk_softmax_rows_warp
    } else {
        moe_topk_softmax_batched
    };
    launch(g, kernel, d_logits, idx, w, e, k, normalize, rows, 0).unwrap();
    g.synchronize(0).unwrap();
    (download(g, idx, n), download(g, w, n))
}

#[test]
#[ignore]
fn moe_topk_warp_writes_the_block_kernels_bytes() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let block = g.kernel("moe_topk", "moe_topk_softmax_rows").unwrap();
    let warp = g.kernel("moe_topk", "moe_topk_softmax_rows_warp").unwrap();
    for (s, shape) in SHAPES.into_iter().enumerate() {
        let d_logits = upload(g, &logits(ROWS, shape.0, 7 + s as u32));
        for normalize in [true, false] {
            let want = route(g, block, false, d_logits, shape, ROWS, normalize);
            let got = route(g, warp, true, d_logits, shape, ROWS, normalize);
            let label = format!("{shape:?} normalize={normalize}");
            assert_eq!(got.0, want.0, "{label}: indices");
            assert_eq!(got.1, want.1, "{label}: weights");
            let written = shape.1.min(shape.0) as usize * 4;
            for (r, row) in want.0.chunks_exact(shape.1 as usize * 4).enumerate() {
                assert!(
                    row[..written].iter().any(|&b| b != SENTINEL),
                    "{label} row {r}: no index written"
                );
                assert!(
                    row[written..].iter().all(|&b| b == SENTINEL),
                    "{label} row {r}: written past the expert count"
                );
            }
            for rows in [1, 4, 5] {
                let one = route(g, warp, true, d_logits, shape, rows, normalize);
                let n = (rows * shape.1 * 4) as usize;
                assert_eq!(one.0, want.0[..n], "{label} rows={rows}: indices");
                assert_eq!(one.1, want.1[..n], "{label} rows={rows}: weights");
            }
        }
    }
    let d_logits = upload(g, &logits(ROWS, 512, 3));
    let (idx, _) = route(g, warp, true, d_logits, (512, 10), ROWS, true);
    let first = |r: usize| u32::from_le_bytes(idx[r * 40..][..4].try_into().unwrap());
    assert_eq!(first(0), 0, "all-equal row: the lowest index first");
    assert_eq!(
        first(1),
        0,
        "row below the sentinel: the pseudo pick of expert 0"
    );
}
