// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The batched dense-BF16 Q/K/V decode route against the per-row route it
//! replaces, at the gated attention shape of Qwen3.8-Flash-Next (hidden 2560, 24 Q heads
//! of 256 with an interleaved gate, 2 KV heads): `dense_gemv_batchm_split` over block rows
//! of at most `DENSE_GEMV_BATCHM_DECODE_MAX_M` plus one strided `deinterleave_qg` must
//! leave every row byte-identical to `dense_gemv` plus a one-row `deinterleave_qg`, at
//! every decode width, inside and past one block row.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release --lib gemm_quant_gemv_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::*;
use crate::layers::ops::deinterleave_qg;

const H: usize = 2560;
const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const Q_PROJ: usize = 2 * NQ * HD;
const KV: usize = NKV * HD;
/// 2026-10-13: One row of `qkv_buf`: `[Q | gate]`, K, V, in BF16 elements.
const ROW: usize = Q_PROJ + 2 * KV;
const SENTINEL: u8 = 0xAB;

/// 2026-10-13: BF16 values in [-1, 1] from a xorshift stream.
fn bf16s(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .flat_map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = (x % 2001) as f32 / 1000.0 - 1.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
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

#[test]
#[ignore]
fn batched_gated_qkv_rows_equal_the_per_row_route_at_every_width() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let row_k = g.kernel("gemv", "dense_gemv_bf16").unwrap();
    let batch_k = g
        .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        .unwrap();
    let split_k = g.kernel("ssm_preprocess", "deinterleave_qg").unwrap();
    let stream = g.create_stream().unwrap();
    let w = |n: usize, seed| DenseWeight {
        weight: upload(g, &bf16s(n * H, seed)),
    };
    let (wq, wk, wv) = (w(Q_PROJ, 7), w(KV, 11), w(KV, 13));
    let row_bytes = ROW * 2;

    for n in [2usize, 3, 8, 9, 16, 59, 64, 128] {
        let normed = upload(g, &bf16s(n * H, 0x9E37_79B9 ^ n as u32));
        let per_row = sentinel(g, n * row_bytes);
        for i in 0..n {
            let x = normed.offset(i * H * 2);
            let q = per_row.offset(i * row_bytes);
            let k = q.offset(Q_PROJ * 2);
            let v = k.offset(KV * 2);
            dense_gemv(g, row_k, x, &wq, q, Q_PROJ as u32, H as u32, stream).unwrap();
            deinterleave_qg(
                g,
                split_k,
                q,
                1,
                NQ as u32,
                HD as u32,
                Q_PROJ as u32,
                stream,
            )
            .unwrap();
            dense_gemv(g, row_k, x, &wk, k, KV as u32, H as u32, stream).unwrap();
            dense_gemv(g, row_k, x, &wv, v, KV as u32, H as u32, stream).unwrap();
        }
        let batched = sentinel(g, n * row_bytes);
        let y = (n as u32).div_ceil(DENSE_GEMV_BATCHM_DECODE_MAX_M);
        let gemv = |wt: &DenseWeight, out: DevicePtr, n_out: usize| {
            dense_gemv_batchm_split(
                g,
                batch_k,
                normed,
                wt,
                out,
                n as u32,
                y,
                n_out as u32,
                H as u32,
                ROW as u32,
                stream,
            )
            .unwrap()
        };
        gemv(&wq, batched, Q_PROJ);
        gemv(&wk, batched.offset(Q_PROJ * 2), KV);
        gemv(&wv, batched.offset((Q_PROJ + KV) * 2), KV);
        g.synchronize(stream).unwrap();
        let raw_q0 = download(g, batched, Q_PROJ * 2);
        deinterleave_qg(
            g, split_k, batched, n as u32, NQ as u32, HD as u32, ROW as u32, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();

        let (a, b) = (
            download(g, per_row, n * row_bytes),
            download(g, batched, n * row_bytes),
        );
        // 2026-10-13: Controls: every row of every segment was written, and the split moved
        // values (the interleaved Q differs from the split one).
        for i in 0..n {
            let r = &a[i * row_bytes..(i + 1) * row_bytes];
            for (name, seg) in [
                ("q", &r[..Q_PROJ * 2]),
                ("k", &r[Q_PROJ * 2..(Q_PROJ + KV) * 2]),
                ("v", &r[(Q_PROJ + KV) * 2..]),
            ] {
                assert!(
                    seg.chunks(2).any(|c| c != [SENTINEL, SENTINEL]),
                    "control: n={n} row {i} {name} written"
                );
            }
        }
        assert_ne!(raw_q0, b[..Q_PROJ * 2], "control: n={n} the split ran");
        for i in 0..n {
            assert!(
                a[i * row_bytes..(i + 1) * row_bytes] == b[i * row_bytes..(i + 1) * row_bytes],
                "n={n}: row {i} differs from the per-row route"
            );
        }
        g.free(normed).unwrap();
        g.free(per_row).unwrap();
        g.free(batched).unwrap();
    }
}
