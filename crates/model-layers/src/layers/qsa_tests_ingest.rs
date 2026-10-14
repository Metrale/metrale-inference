// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: L-K9a, the prefill ingest that keeps only the incomplete block's raw keys.
//! Chunks that start and end mid-block and cross the ingest slab, with a snapshot and
//! restore mid-block, must leave block keys byte-identical to the same projections pooled
//! from one sequence-long raw-key buffer, and the ring holding the last partial block.
//!
//! A child module: `super` is the `qsa_tests` module, `super::super` is `qsa`.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release qsa_ring -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::super::INGEST_SLAB;
use super::*;

/// 2026-10-13: Qwen3.8-Flash-Next's indexer geometry with a small model width.
const N_HEADS: usize = 2;
const HD: usize = 128;
const RATIO: usize = 4;
const HIDDEN: usize = 256;
/// 2026-10-13: Chunks `[start, end)`: mid-block starts and ends, one slab crossing,
/// one longer than a slab; the restore happens after the second (2,050 = mid-block).
const CHUNKS: [(usize, usize); 4] = [(0, 5), (5, 2050), (2050, 4103), (4103, 4110)];
const TOKENS: usize = 4110;

/// 2026-10-13: `n` BF16 values in [-0.25, 0.25] (bit patterns), xorshift from `seed`.
fn bf16s(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .flat_map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = ((x % 2001) as f32 / 1000.0 - 1.0) / 4.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn download(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

#[test]
#[ignore]
fn qsa_ring_chunked_ingest_with_a_restore_equals_pooling_the_whole_sequence() {
    assert!(
        CHUNKS[2].1 - CHUNKS[2].0 > INGEST_SLAB,
        "a chunk must span two slabs"
    );
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let qkw = (N_HEADS + 1) * HD;
    let (row, hrow) = (HD * 2, HIDDEN * 2);
    let w = upload(g, &bf16s(qkw * HIDDEN, 0x2545_F491));
    let k_norm = upload(g, &bf16s(HD, 0x9E37_79B9));
    let qsa = QsaIndexer::new(
        w,
        upload(g, &bf16s(HD, 0x1234_5678)),
        k_norm,
        N_HEADS,
        HD,
        RATIO,
        256,
        64,
        1.0e7,
        1e-6,
        HIDDEN,
        1,
        HD,
        8192,
        g,
    )
    .unwrap();
    let hidden = upload(g, &bf16s(TOKENS * HIDDEN, 0x0BAD_F00D));

    // 2026-10-13: Production: chunked ingest, the state snapshotted after the second
    // chunk and restored into a fresh one that ingests the rest.
    let mut st = qsa.new_seq_state(g).unwrap();
    let mut blob = Vec::new();
    for (i, &(s, e)) in CHUNKS.iter().enumerate() {
        qsa.prefill_ingest(&mut st, hidden.offset(s * hrow), e - s, s, g, stream)
            .unwrap();
        if i == 1 {
            blob = qsa.snapshot_aux(&st, g, stream).unwrap();
            qsa.release_seq_state(&mut st, g).unwrap();
            st = qsa.new_seq_state(g).unwrap();
            qsa.restore_aux(&mut st, &blob, g, stream).unwrap();
        }
    }
    g.synchronize(stream).unwrap();
    let (blocks, tail) = (TOKENS / RATIO, TOKENS % RATIO);
    assert_eq!((st.ingested, st.pooled), (TOKENS, blocks));
    let split = CHUNKS[1].1;
    assert_eq!(
        blob.len(),
        16 + (split / RATIO + split % RATIO) * row,
        "the blob holds the block keys and the raw tail, not every raw key"
    );

    // 2026-10-13: Reference: the same slab GEMMs, every raw key kept, one pool at the end.
    let scratch = g.alloc(INGEST_SLAB * qkw * 2).unwrap();
    let raw_ref = g.alloc(TOKENS * row).unwrap();
    let blocks_ref = g.alloc(blocks * row).unwrap();
    for &(s, e) in &CHUNKS {
        let mut off = 0;
        while off < e - s {
            let ts = INGEST_SLAB.min(e - s - off);
            let at = s + off;
            ops::cublas_bf16_proj_dense(
                hidden.offset(at * hrow),
                w,
                scratch,
                ts as u32,
                qkw as u32,
                HIDDEN as u32,
                stream,
            )
            .unwrap();
            g.copy_d2d_2d_async(
                scratch.offset(N_HEADS * HD * 2),
                qkw * 2,
                raw_ref.offset(at * row),
                row,
                row,
                ts,
                stream,
            )
            .unwrap();
            off += ts;
        }
    }
    ops::qsa_block_pool(
        g,
        g.kernel("qsa_indexer", "qsa_block_pool").unwrap(),
        raw_ref,
        k_norm,
        blocks_ref,
        0,
        blocks as u32,
        RATIO as u32,
        HD as u32,
        64,
        1.0e7,
        1e-6,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let want = download(g, blocks_ref, blocks * row);
    assert!(want.iter().any(|&b| b != 0), "the reference pooled nothing");
    let got = download(g, st.block_keys, blocks * row);
    for (b, (gb, wb)) in got.chunks(row).zip(want.chunks(row)).enumerate() {
        assert_eq!(gb, wb, "block {b}");
    }
    let raw = download(g, raw_ref, TOKENS * row);
    assert_eq!(
        download(g, st.raw_tail, tail * row),
        raw[blocks * RATIO * row..],
        "the ring holds the incomplete block's keys from row 0"
    );
}

#[test]
#[ignore]
fn qsa_ring_restore_refuses_a_blob_with_an_unpooled_block() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let qkw = (N_HEADS + 1) * HD;
    let qsa = QsaIndexer::new(
        upload(g, &bf16s(qkw * HIDDEN, 7)),
        upload(g, &bf16s(HD, 8)),
        upload(g, &bf16s(HD, 9)),
        N_HEADS,
        HD,
        RATIO,
        256,
        64,
        1.0e7,
        1e-6,
        HIDDEN,
        1,
        HD,
        8192,
        g,
    )
    .unwrap();
    let mut st = qsa.new_seq_state(g).unwrap();
    qsa.prefill_ingest(&mut st, upload(g, &bf16s(9 * HIDDEN, 10)), 9, 0, g, stream)
        .unwrap();
    let blob = qsa.snapshot_aux(&st, g, stream).unwrap();
    assert_eq!(blob.len(), 16 + (2 + 1) * HD * 2);
    // 2026-10-13: The pre-L-K9a layout (every raw key, nothing pooled) and a pooled count
    // one short are both refused, and the state is left untouched.
    let mut old = blob[..16].to_vec();
    old[8..16].copy_from_slice(&0u64.to_le_bytes());
    old.resize(16 + 9 * HD * 2, 0);
    let mut short = blob.clone();
    short[8..16].copy_from_slice(&1u64.to_le_bytes());
    let mut fresh = qsa.new_seq_state(g).unwrap();
    for (label, b) in [("old layout", &old), ("one block short", &short)] {
        let err = qsa.restore_aux(&mut fresh, b, g, stream).unwrap_err();
        assert!(format!("{err:#}").contains("pooled"), "{label}: {err:#}");
        assert_eq!((fresh.ingested, fresh.pooled), (0, 0), "{label}");
    }
    let mut truncated = blob.clone();
    truncated.pop();
    assert!(qsa.restore_aux(&mut fresh, &truncated, g, stream).is_err());
    qsa.restore_aux(&mut fresh, &blob, g, stream).unwrap();
    assert_eq!((fresh.ingested, fresh.pooled), (9, 2));
}
