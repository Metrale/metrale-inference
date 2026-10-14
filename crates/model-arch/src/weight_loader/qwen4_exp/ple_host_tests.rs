// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests of the host-resident PLE n-gram table: shards land back to back in
//! shard order whatever the thread count, malformed shard sets are refused, a backend
//! unable to map host memory fails naming the lever, and (GPU, `#[ignore]`) the
//! production gather kernel reads the checkpoint's exact rows, the last shard included,
//! through the mapped table with the row id as the slot.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-arch --release ple_host -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::{PLE_HOST_LEVER, load, read_shards};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype};

const ROWS: usize = 7;
const DIM: usize = 12;
const ROW_BYTES: usize = DIM * 2;

/// 2026-10-13: `n` shards of `ROWS` BF16 rows, each in its own file at a distinct,
/// non-zero offset, and the table they form. Every byte is distinct modulo 251 across
/// the whole table, so a misplaced shard or a wrong offset cannot read back equal.
fn shards(n: usize) -> (tempfile::TempDir, Vec<DeferredTensor>, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let shard_bytes = ROWS * ROW_BYTES;
    let table: Vec<u8> = (0..n * shard_bytes).map(|i| (i % 251) as u8).collect();
    let ds = (0..n)
        .map(|i| {
            let path = dir.path().join(format!("shard{i}.safetensors"));
            let header = 5 + 3 * i;
            let mut file = vec![0xEEu8; header];
            file.extend_from_slice(&table[i * shard_bytes..(i + 1) * shard_bytes]);
            file.extend_from_slice(&[0xEE; 9]);
            std::fs::write(&path, &file).unwrap();
            DeferredTensor {
                path,
                offset: header as u64,
                shape: vec![ROWS, DIM],
                dtype: WeightDtype::BF16,
            }
        })
        .collect();
    (dir, ds, table)
}

#[test]
fn shards_land_back_to_back_in_shard_order_at_every_thread_count() {
    let (_dir, ds, table) = shards(5);
    for threads in [1, 2, 3, 5, 16] {
        let mut dst = vec![0u8; table.len()];
        read_shards(&ds, &mut dst, threads).unwrap();
        assert_eq!(dst, table, "{threads} threads");
    }
}

#[test]
fn malformed_shard_sets_are_refused() {
    let (_dir, ds, table) = shards(3);
    let mut dst = vec![0u8; table.len()];
    assert!(read_shards(&[], &mut dst, 4).is_err(), "no shards");
    assert!(read_shards(&ds, &mut dst, 0).is_err(), "zero threads");
    let mut short = vec![0u8; table.len() - 1];
    assert!(
        read_shards(&ds, &mut short, 4).is_err(),
        "buffer one byte short"
    );
    let mut long = vec![0u8; table.len() + ROW_BYTES];
    assert!(
        read_shards(&ds, &mut long, 4).is_err(),
        "buffer one row long"
    );
    let mut uneven = ds.clone();
    uneven[1].shape = vec![ROWS - 1, DIM];
    let mut dst = vec![0u8; table.len() - ROW_BYTES];
    let err = read_shards(&uneven, &mut dst, 4).unwrap_err();
    assert!(format!("{err:#}").contains("differ in size"), "{err:#}");
}

#[test]
fn a_truncated_shard_fails_naming_it() {
    let (_dir, ds, table) = shards(4);
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&ds[2].path)
        .unwrap();
    f.set_len(ds[2].offset + (ROWS * ROW_BYTES) as u64 - 1)
        .unwrap();
    let mut dst = vec![0u8; table.len()];
    let err = read_shards(&ds, &mut dst, 2).unwrap_err();
    assert!(format!("{err:#}").contains("shard 2"), "{err:#}");
}

#[test]
fn a_backend_that_cannot_map_host_memory_fails_naming_the_lever() {
    let (_dir, ds, _) = shards(2);
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let err = load(&ds, ROW_BYTES, &gpu).expect_err("the mock backend maps no host memory");
    assert!(format!("{err:#}").contains(PLE_HOST_LEVER), "{err:#}");
}

#[test]
#[ignore]
fn the_gather_kernel_reads_every_shards_rows_through_the_mapped_table() {
    let (_dir, ds, table) = shards(3);
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let resident = load(&ds, ROW_BYTES, g).unwrap();
    // 2026-10-13: Rows from every shard, the last row of the table and a repeat; the
    // slot is the row id, uploaded as u32 exactly as `PleLayer::gather_host` does.
    let last = (3 * ROWS - 1) as u32;
    let ids: [u32; 7] = [
        last,
        0,
        9,
        9,
        (ROWS - 1) as u32,
        ROWS as u32,
        2 * ROWS as u32,
    ];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    let ids_dev = g.alloc(id_bytes.len()).unwrap();
    g.copy_h2d(&id_bytes, ids_dev).unwrap();
    let out = g.alloc(ids.len() * ROW_BYTES).unwrap();
    let stream = g.default_stream();
    let kernel = g.kernel("embed_from_argmax", "batched_embed").unwrap();
    metrale_model_layers::layers::ops::batched_embed(
        g,
        kernel,
        ids_dev,
        resident.weight,
        out,
        ids.len() as u32,
        DIM as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let mut got = vec![0u8; ids.len() * ROW_BYTES];
    g.copy_d2h(out, &mut got).unwrap();
    for (i, &r) in ids.iter().enumerate() {
        let r = r as usize;
        let want = &table[r * ROW_BYTES..(r + 1) * ROW_BYTES];
        assert_eq!(
            &got[i * ROW_BYTES..(i + 1) * ROW_BYTES],
            want,
            "slot {i} (row {r})"
        );
    }
}
