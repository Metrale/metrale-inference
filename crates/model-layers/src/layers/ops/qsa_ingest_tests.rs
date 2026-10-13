// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `qsa_ingest_step`, the decode ingest that reads its position on the
//! device, against the path it replaces (the key copied to `raw_keys[pos]`, then
//! `qsa_block_pool` over the block it completes), launched directly and replayed from one
//! CUDA graph captured once: raw and block keys must be byte-identical at every position,
//! those that complete a block and those that do not.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release qsa_ingest -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::*;

/// 2026-10-13: Qwen3.8-Flash-Next's indexer geometry: head dim 128, compress ratio 4,
/// rope over a quarter of the 256-wide attention head, theta 1e7.
const HD: usize = 128;
const RATIO: usize = 4;
const ROT: usize = 64;
const THETA: f32 = 1e7;
const EPS: f32 = 1e-6;
/// 2026-10-13: Seven complete blocks and three tokens of an eighth.
const TOKENS: usize = 7 * RATIO + 3;

/// 2026-10-13: `n` BF16 values (bit patterns) of varied sign and magnitude.
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

/// 2026-10-13: Fresh zeroed raw-key and block-key buffers for `TOKENS` tokens.
fn buffers(g: &dyn GpuBackend) -> (DevicePtr, DevicePtr) {
    let raw = g.alloc(TOKENS * HD * 2).unwrap();
    let blocks = g.alloc(TOKENS / RATIO * HD * 2).unwrap();
    g.memset(raw, 0, TOKENS * HD * 2).unwrap();
    g.memset(blocks, 0, TOKENS / RATIO * HD * 2).unwrap();
    (raw, blocks)
}

fn download(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

#[test]
#[ignore]
fn the_device_position_ingest_equals_copy_then_pool_direct_and_replayed() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let pool_k = g.kernel("qsa_indexer", "qsa_block_pool").unwrap();
    let ingest_k = g.kernel("qsa_indexer", "qsa_ingest_step").unwrap();
    let stream = g.create_stream().unwrap();
    let keys = bf16s(TOKENS * HD, 0x2545_F491);
    let norm = g.alloc(HD * 2).unwrap();
    g.copy_h2d(&bf16s(HD, 0x9E37_79B9), norm).unwrap();
    let key = g.alloc(HD * 2).unwrap();
    let pos_dev = g.alloc(4).unwrap();
    let row = |p: usize| &keys[p * HD * 2..(p + 1) * HD * 2];
    let (h, r, t, e) = (HD as u32, RATIO as u32, ROT as u32, (THETA, EPS));

    // 2026-10-13: The replaced path: copy the key, pool the block it completes.
    let (raw_ref, blocks_ref) = buffers(g);
    for p in 0..TOKENS {
        g.copy_h2d(row(p), raw_ref.offset(p * HD * 2)).unwrap();
        if (p + 1) % RATIO == 0 {
            let b = ((p + 1) / RATIO - 1) as u32;
            qsa_block_pool(
                g, pool_k, raw_ref, norm, blocks_ref, b, 1, r, h, t, e.0, e.1, stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
        }
    }

    // 2026-10-13: Direct launches, the position uploaded before each.
    let (raw_new, blocks_new) = buffers(g);
    let step = |raw: DevicePtr, blocks: DevicePtr| {
        qsa_ingest_step(
            g, ingest_k, key, raw, norm, blocks, pos_dev, r, h, t, e.0, e.1, stream,
        )
    };
    for p in 0..TOKENS {
        g.copy_h2d(row(p), key).unwrap();
        g.copy_h2d(&(p as u32).to_le_bytes(), pos_dev).unwrap();
        step(raw_new, blocks_new).unwrap();
        g.synchronize(stream).unwrap();
    }

    // 2026-10-13: One graph captured once (capture runs nothing), replayed per position.
    let (raw_graph, blocks_graph) = buffers(g);
    g.begin_capture(stream).unwrap();
    let captured = step(raw_graph, blocks_graph);
    let graph = g.end_capture(stream).unwrap();
    captured.unwrap();
    for p in 0..TOKENS {
        g.copy_h2d(row(p), key).unwrap();
        g.copy_h2d(&(p as u32).to_le_bytes(), pos_dev).unwrap();
        g.launch_graph(graph, stream).unwrap();
        g.synchronize(stream).unwrap();
    }
    g.destroy_graph(graph).unwrap();

    let want_raw = download(g, raw_ref, TOKENS * HD * 2);
    let want_blocks = download(g, blocks_ref, TOKENS / RATIO * HD * 2);
    assert_eq!(&want_raw[..], &keys[..], "the reference holds every key");
    assert!(
        want_blocks.iter().any(|&b| b != 0),
        "the reference pooled nothing"
    );
    for (label, raw, blocks) in [
        ("direct", raw_new, blocks_new),
        ("replayed", raw_graph, blocks_graph),
    ] {
        assert_eq!(
            download(g, raw, TOKENS * HD * 2),
            want_raw,
            "{label}: raw keys"
        );
        let got = download(g, blocks, TOKENS / RATIO * HD * 2);
        for (b, (gb, wb)) in got
            .chunks(HD * 2)
            .zip(want_blocks.chunks(HD * 2))
            .enumerate()
        {
            assert_eq!(gb, wb, "{label}: block {b}");
        }
    }
}
