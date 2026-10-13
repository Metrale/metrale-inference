// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `dsa_topk_pools` against the host sort the QSA prefill selection used
//! before it (score descending, then index ascending, over each row's `complete` prefix),
//! and the launcher's tile checks.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release pool_topk -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::*;

/// 2026-10-13: The QSA scorers' value for a block at or past a row's `complete`.
const NOT_COMPLETE: f32 = -1e30;

/// 2026-10-13: The host top-k `qsa_select.rs` ran until 2026-10-13, kept as the reference.
fn host_topk(row: &[f32], complete: usize, k: usize) -> Vec<i32> {
    let mut order: Vec<u32> = (0..complete as u32).collect();
    order.sort_by(|&a, &b| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order[..k].iter().map(|&b| b as i32).collect()
}

/// 2026-10-13: Scores as the QSA scorers leave them: non-negative over `[0, complete)`,
/// `NOT_COMPLETE` after. `levels` values only, so most scores tie and the index
/// tie-break decides the order.
fn scores(rows: usize, n: usize, first_pos: usize, ratio: usize, levels: u32) -> Vec<f32> {
    let mut x = 0x9E37_79B9u32;
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        let complete = (first_pos + r + 1) / ratio;
        for b in 0..n {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            out[r * n + b] = if b < complete {
                (x % levels) as f32 * 0.125
            } else {
                NOT_COMPLETE
            };
        }
    }
    out
}

#[test]
fn the_launcher_refuses_a_tile_the_kernel_cannot_hold() {
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let k = gpu.kernel("dsa_indexer", "dsa_topk_pools").unwrap();
    let p = DevicePtr::NULL;
    let launch =
        |tile: u32, select_k: u32| dsa_topk_pools(&gpu, k, p, p, 1, 4096, tile, select_k, p, 0);
    assert!(launch(1000, 512).is_err(), "not a power of two");
    assert!(launch(256, 512).is_err(), "select_k above the tile");
    assert!(
        launch(2 * topk_tile() as u32, 512).is_err(),
        "above the smem ceiling"
    );
}

#[test]
fn the_tile_ceiling_fits_the_shared_memory_ceiling() {
    let t = topk_tile();
    assert!(topk_smem_for_tile(t) <= TOPK_SMEM_CEILING);
    assert!(topk_smem_for_tile(2 * t) > TOPK_SMEM_CEILING);
}

#[test]
#[ignore]
fn the_device_top_k_equals_the_qsa_host_sort() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let kernel = g.kernel("dsa_indexer", "dsa_topk_pools").unwrap();
    let stream = g.default_stream();
    let (ratio, topk) = (4usize, 512usize);
    // 2026-10-13: (first_pos, rows, levels): the first selective row (complete 513, one
    // spare block), a whole 2,048-row slab mid-context, a 36,864-token tail, all-ties.
    for (first_pos, rows, levels) in [
        (2051usize, 64usize, 7u32),
        (6000, 2048, 1000),
        (34_000, 300, 50_000),
        (2051, 1000, 1),
    ] {
        let n = (first_pos + rows) / ratio;
        let sc = scores(rows, n, first_pos, ratio, levels);
        let bytes: Vec<u8> = sc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let s_dev = g.alloc(bytes.len()).unwrap();
        g.copy_h2d(&bytes, s_dev).unwrap();
        let out = g.alloc(rows * topk * 4).unwrap();
        for tile in [1024usize, topk_tile()] {
            let tile = tile.min(n.next_power_of_two());
            dsa_topk_pools(
                g,
                kernel,
                s_dev,
                out,
                rows as u32,
                n as u32,
                tile as u32,
                topk as u32,
                DevicePtr::NULL,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let mut got = vec![0u8; rows * topk * 4];
            g.copy_d2h(out, &mut got).unwrap();
            for r in 0..rows {
                let complete = (first_pos + r + 1) / ratio;
                let want = host_topk(&sc[r * n..(r + 1) * n], complete, topk);
                let row: Vec<i32> = got[r * topk * 4..(r + 1) * topk * 4]
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(row, want, "first_pos {first_pos} row {r} tile {tile}");
            }
        }
    }
}
