// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host tests of the paged indexer cache on the mock backend, which records every
//! launch with its arguments: a paged row is written into the V pool block its block table
//! names, and the selection reads the V pool through that table with no validity buffer. The
//! numerics (paged bytes equal flat bytes) are checked on a GPU by
//! `examples/glm5next_dsa_paged_parity.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::LayerState;

use crate::glm5next_dsa::paged::IndexerCache;
use crate::glm5next_dsa::state::Glm5NextDsaState;

use super::super::decode_rows_fixture::*;

fn bytes32(v: u32) -> metrale_gpu_runtime::gpu::mock::MockArg {
    metrale_gpu_runtime::gpu::mock::MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn paged_states(lens: &[usize]) -> Vec<Box<dyn LayerState>> {
    lens.iter()
        .map(|&len| {
            let mut s = Glm5NextDsaState::paged(&cfg()).unwrap();
            s.advance(len).unwrap();
            Box::new(s) as Box<dyn LayerState>
        })
        .collect()
}

/// 2026-10-09: Batched decode over two paged rows, eager. Each row is placed on the device by
/// `dsa_indexer_store` through ITS metadata block-table row into the V pool (keys at the pool
/// base, gates one key region further, no validity byte), and its selection reads the same
/// rows. A path that kept the flat pointers (NULL for a paged state), dropped the table, or
/// used row 0's table for every row fails here.
#[test]
fn a_paged_batched_row_is_stored_and_selected_through_its_own_table() {
    let rig = Rig::new();
    let mut layer = rig.layer();
    layer.indexer_cache = IndexerCache::Paged;
    let meta = rig.meta(2);
    let mut kv = rig.kv();
    let pool = kv.v_pool_ptr(0);
    let stride = kv.v_block_stride_bytes_for_layer(0);
    let gate = pool.offset(16 * cfg().index_head_dim * 2);
    let lens = [5usize, 20];
    let mut boxes = paged_states(&lens);
    let hidden = rig.buf(2 * HIDDEN * 2);
    let from = rig.gpu.launch_count();
    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
        boxes.iter_mut().map(|b| b.as_mut()).collect();
    layer
        .decode_rows(
            hidden,
            &mut refs,
            &lens,
            &mut kv,
            &meta,
            0,
            &rig.ctx(false),
            7,
        )
        .unwrap();
    let l = rig.since(from);

    let store = of(&l, STORE);
    assert_eq!(store.len(), 2, "every paged row is placed on the device");
    let kpool = of(&l, KPOOL);
    assert_eq!(kpool.len(), 2);
    for r in 0..2 {
        let bt = ptr(meta.block_table.offset(r * MB * 4));
        let s = &store[r].args;
        assert_eq!(
            s[2],
            ptr(meta.positions.offset(r * 4)),
            "row {r}'s position"
        );
        assert_eq!(
            s[3..6],
            [ptr(pool), ptr(gate), ptr(DevicePtr::NULL)],
            "row {r}"
        );
        assert_eq!(s[7], bt, "row {r} is placed through its own table");
        assert_eq!(s[8..10], [bytes32(16), bytes32((stride / 2) as u32)]);
        let c = &kpool[r].args;
        assert_eq!(
            c[..3],
            [ptr(pool), ptr(gate), ptr(DevicePtr::NULL)],
            "row {r}"
        );
        assert_eq!(c[12], bt, "row {r} selects through its own table");
        assert_eq!(c[13..15], [bytes32(16), bytes32((stride / 2) as u32)]);
    }
    for e in of(&l, EXPAND) {
        assert_eq!(
            e.args[3],
            ptr(DevicePtr::NULL),
            "a paged cache has no validity bytes"
        );
    }
    let after: Vec<usize> = boxes.iter().map(|b| dsa(b.as_ref()).len()).collect();
    assert_eq!(after, vec![6, 21]);
}

/// 2026-10-09: The eager single-sequence path without metadata places a paged row on the host,
/// at the V pool offset its block table gives (row 20 of a 16-token block size is logical
/// block 1, physical 2, slot 4), and launches no store kernel. A paged state behind the
/// sequence (a prefix-cache hit) is adopted rather than refused.
#[test]
fn a_paged_host_row_lands_in_the_block_its_table_names() {
    let rig = Rig::new();
    let mut layer = rig.layer();
    layer.indexer_cache = IndexerCache::Paged;
    let mut kv = rig.kv();
    let pool = kv.v_pool_ptr(0);
    let stride = kv.v_block_stride_bytes_for_layer(0);
    let d = cfg().index_head_dim;
    let mut st: Box<dyn LayerState> = Box::new(layer.alloc_dsa_state(&rig.gpu).unwrap());
    let mut bt = vec![5u32, 2, 7, 1];
    let hidden = rig.buf(HIDDEN * 2);
    let from = rig.gpu.launch_count();
    layer
        .decode_k(
            hidden,
            1,
            st.as_mut(),
            &mut kv,
            20,
            &mut bt,
            &rig.ctx(false),
            7,
            false,
        )
        .unwrap();
    let l = rig.since(from);
    let knorm = of(&l, KNORM);
    assert_eq!(knorm.len(), 1);
    assert_eq!(knorm[0].args[0], ptr(pool.offset(2 * stride + 4 * d * 2)));
    assert!(of(&l, STORE).is_empty(), "the host path launches no store");
    assert_eq!(of(&l, KPOOL)[0].args[12], ptr(layer.workspace.bt));
    assert_eq!(dsa(st.as_ref()).len(), 21);
}

/// 2026-10-09: A KV cache whose V side cannot hold the rows (a 32-wide FP8 latent: 512 B per
/// block against the 1,024 B sixteen 16-wide key and gate rows need) is refused before any
/// launch, not written past.
#[test]
fn a_v_side_too_small_is_refused_before_any_launch() {
    use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
    let rig = Rig::new();
    let mut layer = rig.layer();
    layer.indexer_cache = IndexerCache::Paged;
    let mut kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 32,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        8,
        &rig.gpu,
    )
    .unwrap();
    let meta = rig.meta(1);
    let mut boxes = paged_states(&[3]);
    let hidden = rig.buf(HIDDEN * 2);
    let from = rig.gpu.launch_count();
    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
        boxes.iter_mut().map(|b| b.as_mut()).collect();
    let err = layer
        .decode_rows(
            hidden,
            &mut refs,
            &[3],
            &mut kv,
            &meta,
            0,
            &rig.ctx(false),
            7,
        )
        .unwrap_err();
    assert!(err.to_string().contains("V side"), "{err}");
    assert_eq!(
        rig.gpu.launch_count(),
        from,
        "refused before the first launch"
    );
}

/// 2026-10-09: A 12-row prefill sub-chunk from position 10 crosses the 16-token block boundary:
/// rows 10..15 land in physical block 5 (slots 10..15) and rows 16..21 in physical block 2
/// (slots 0..5), and the batched selection reads through the uploaded table. A placement that
/// kept the chunk's first block for every row fails at row 16.
#[test]
fn a_prefill_chunk_crossing_a_block_boundary_follows_the_table() {
    let rig = Rig::new();
    let mut layer = rig.layer();
    layer.indexer_cache = IndexerCache::Paged;
    let mut kv = rig.kv();
    let pool = kv.v_pool_ptr(0);
    let stride = kv.v_block_stride_bytes_for_layer(0);
    let d = cfg().index_head_dim;
    let mut st: Box<dyn LayerState> = Box::new(layer.alloc_dsa_state(&rig.gpu).unwrap());
    let mut bt = vec![5u32, 2, 7, 1];
    let hidden = rig.buf(12 * HIDDEN * 2);
    let mut ctx = rig.ctx(false);
    ctx.decode_step = false;
    let from = rig.gpu.launch_count();
    layer
        .decode_k(hidden, 12, st.as_mut(), &mut kv, 10, &mut bt, &ctx, 7, true)
        .unwrap();
    let l = rig.since(from);
    let knorm = of(&l, KNORM);
    assert_eq!(knorm.len(), 12);
    for (i, w) in knorm.iter().enumerate() {
        let pos = 10 + i;
        let block = bt[pos / 16] as usize;
        let want = pool.offset(block * stride + (pos % 16) * d * 2);
        assert_eq!(w.args[0], ptr(want), "row at position {pos}");
    }
    let kpool = of(&l, KPOOL);
    assert_eq!(kpool.len(), 1, "one batched selection for the chunk");
    assert_eq!(kpool[0].args[12], ptr(layer.workspace.bt));
    assert_eq!(dsa(st.as_ref()).len(), 22);
}
