// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Host tests of `Glm5NextDsaLayer::decode_rows` on the mock backend, which records
//! every launch with its arguments: each row reads and writes only its own sequence's state and
//! metadata row, one row launches exactly what the single-sequence `decode_k` launches, and
//! every refusal comes before the first launch. The numerics are checked on a GPU by
//! `examples/glm5next_multi_seq_decode_gate.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockArg;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layer::LayerState;

use crate::glm5next_dsa::state::Glm5NextDsaState;

#[path = "decode_rows_fixture.rs"]
mod fixture;
use fixture::*;

/// 2026-10-08: Three sequences at different lengths. One latent write covers the rows,
/// reading the metadata slots and `kv_a` from row 0; row `r`'s indexer row lands at its own
/// cache's next row; its selection reads its own cache and its own position; the attend covers the three rows
/// through the metadata's per-row tables; each cache grows by one row. A bug that handed
/// every row sequence 0's state or metadata row 0 fails each of these.
#[test]
fn each_row_reads_and_writes_only_its_own_sequence() {
    let rig = Rig::new();
    let layer = rig.layer();
    let meta = rig.meta(3);
    let lens = [5usize, 9, 2];
    let mut boxes = rig.states(&lens);
    let before: Vec<(DevicePtr, DevicePtr, DevicePtr)> = boxes
        .iter()
        .map(|b| {
            let s = dsa(b.as_ref());
            (s.k_normed, s.gate, s.valid)
        })
        .collect();
    let from = rig.gpu.launch_count();
    run_rows(&rig, &layer, &mut boxes, &lens, &meta, 0, false).unwrap();
    let l = rig.since(from);

    let latent = of(&l, LATENT);
    assert_eq!(latent.len(), 1, "one latent write for every row");
    assert_eq!(latent[0].grid[0], 3, "one block per row");
    assert_eq!(latent[0].args[0], ptr(layer.workspace.kv_a));
    assert_eq!(
        latent[0].args[3],
        ptr(meta.slot),
        "row r writes metadata slot r"
    );
    let knorm = of(&l, KNORM);
    assert_eq!(knorm.len(), 3);
    for (r, w) in knorm.iter().enumerate() {
        let row = lens[r] * cfg().index_head_dim * 2;
        assert_eq!(
            w.args[0],
            ptr(before[r].0.offset(row)),
            "row {r}'s next indexer row"
        );
    }
    let kpool = of(&l, KPOOL);
    assert_eq!(kpool.len(), 3);
    for (r, w) in kpool.iter().enumerate() {
        assert_eq!(
            w.args[..3],
            [ptr(before[r].0), ptr(before[r].1), ptr(before[r].2)],
            "row {r} selects over its own cache"
        );
    }
    let expand = of(&l, EXPAND);
    assert_eq!(expand.len(), 3);
    for (r, w) in expand.iter().enumerate() {
        assert_eq!(
            w.args[4],
            ptr(meta.positions.offset(r * 4)),
            "row {r}'s q_pos"
        );
    }
    let attend = of(&l, MOCK_K);
    assert_eq!(attend.len(), 1, "one attend for every row");
    assert_eq!(attend[0].grid[1], 3, "grid y is the row");
    assert_eq!(attend[0].args[4], ptr(meta.block_table));
    assert_eq!(attend[0].args[5], ptr(meta.seq_len));
    assert_eq!(
        attend[0].args[8],
        MockArg::Bytes((MB as u32).to_le_bytes().to_vec()),
        "the block-table row stride is the metadata's"
    );
    let proj = of(&l, BATCHM);
    assert_eq!(proj.len(), 4, "q_a, q_absorb, kv_a and o_absorb, each once");
    for p in &proj {
        assert_eq!(p.args[3], MockArg::Bytes(3u32.to_le_bytes().to_vec()));
    }
    let after: Vec<usize> = boxes.iter().map(|b| dsa(b.as_ref()).len()).collect();
    assert_eq!(after, vec![6, 10, 3]);
}

/// 2026-10-08: At one row, `decode_rows` issues exactly the launches of `decode_k` at `k = 1`
/// on a decode step over the same state and metadata, eager and captured: the batched path at
/// C1 is the single-sequence path. The state is rewound between the two runs so both start
/// from the same row.
#[test]
fn one_row_launches_what_the_single_sequence_decode_launches() {
    for capture in [false, true] {
        let rig = Rig::new();
        let layer = rig.layer();
        let meta = rig.meta(1);
        let mut kv = rig.kv();
        let hidden = rig.buf(HIDDEN * 2);
        let mut boxes = rig.states(&[7]);
        let mut ctx = rig.ctx(capture);
        ctx.attn_metadata = Some(meta);

        let from = rig.gpu.launch_count();
        let mut bt = vec![0u32; MB];
        layer
            .decode_k(
                hidden,
                1,
                boxes[0].as_mut(),
                &mut kv,
                7,
                &mut bt,
                &ctx,
                7,
                false,
            )
            .unwrap();
        let single = rig.since(from);
        boxes[0]
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .unwrap()
            .rewind_to(7)
            .unwrap();

        let from = rig.gpu.launch_count();
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            boxes.iter_mut().map(|b| b.as_mut()).collect();
        layer
            .decode_rows(hidden, &mut refs, &[7], &mut kv, &meta, 0, &ctx, 7)
            .unwrap();
        let rows = rig.since(from);

        assert!(!single.is_empty());
        assert_eq!(rows.len(), single.len(), "capture={capture}: launch count");
        for (i, (a, b)) in rows.iter().zip(&single).enumerate() {
            assert_eq!(
                (a.func, a.grid, a.block, a.shared_mem, &a.args),
                (b.func, b.grid, b.block, b.shared_mem, &b.args),
                "capture={capture}: launch {i} differs from the single-sequence decode"
            );
        }
    }
}

/// 2026-10-08: In a capture, each row places its indexer row and sizes its selection from
/// device memory: `dsa_indexer_store` reads row `r`'s metadata position into row `r`'s cache,
/// and `dsa_write_geom` reads row `r`'s metadata `seq_len`. A replay then serves later
/// positions.
#[test]
fn a_captured_row_places_its_indexer_row_from_its_metadata() {
    let rig = Rig::new();
    let layer = rig.layer();
    let meta = rig.meta(2);
    let lens = [5usize, 9];
    let mut boxes = rig.states(&lens);
    let caches: Vec<DevicePtr> = boxes.iter().map(|b| dsa(b.as_ref()).k_normed).collect();
    let from = rig.gpu.launch_count();
    run_rows(&rig, &layer, &mut boxes, &lens, &meta, 0, true).unwrap();
    let l = rig.since(from);
    let store = of(&l, STORE);
    assert_eq!(store.len(), 2);
    let geom = of(&l, GEOM);
    assert_eq!(geom.len(), 2);
    for r in 0..2 {
        assert_eq!(store[r].args[2], ptr(meta.positions.offset(r * 4)));
        assert_eq!(store[r].args[3], ptr(caches[r]));
        assert_eq!(geom[r].args[0], ptr(meta.seq_len.offset(r * 4)));
    }
}

/// 2026-10-08: A group that starts at metadata row 3 reads row 3's slot, block-table row and
/// `seq_len`, not row 0's; a group that would pass the metadata's rows is refused before any
/// launch.
#[test]
fn a_group_reads_the_metadata_rows_at_its_base() {
    let rig = Rig::new();
    let layer = rig.layer();
    let meta = rig.meta(4);
    let mut boxes = rig.states(&[4]);
    let from = rig.gpu.launch_count();
    run_rows(&rig, &layer, &mut boxes, &[4], &meta, 3, false).unwrap();
    let l = rig.since(from);
    assert_eq!(of(&l, LATENT)[0].args[3], ptr(meta.slot.offset(3 * 8)));
    let attend = of(&l, MOCK_K);
    assert_eq!(attend[0].args[4], ptr(meta.block_table.offset(3 * MB * 4)));
    assert_eq!(attend[0].args[5], ptr(meta.seq_len.offset(3 * 4)));

    let mut two = rig.states(&[4, 4]);
    let from = rig.gpu.launch_count();
    let e = run_rows(&rig, &layer, &mut two, &[4, 4], &meta, 3, false).unwrap_err();
    assert!(e.to_string().contains("metadata rows"), "{e}");
    assert_eq!(rig.gpu.launch_count(), from, "refused before any launch");
}

/// 2026-10-08: A row whose cache is behind its sequence is refused before the first launch,
/// and no row advances; a cache ahead of its sequence (a rejected draft) is rewound first.
#[test]
fn lockstep_is_checked_for_every_row_before_any_launch() {
    let rig = Rig::new();
    let layer = rig.layer();
    let meta = rig.meta(2);
    let mut boxes = rig.states(&[5, 7]);
    let from = rig.gpu.launch_count();
    let e = run_rows(&rig, &layer, &mut boxes, &[5, 9], &meta, 0, false).unwrap_err();
    assert!(e.to_string().contains("MISSING"), "{e}");
    assert_eq!(rig.gpu.launch_count(), from, "refused before any launch");
    assert_eq!(dsa(boxes[0].as_ref()).len(), 5, "no row advanced");

    let mut ahead = rig.states(&[5, 12]);
    run_rows(&rig, &layer, &mut ahead, &[5, 9], &meta, 0, false).unwrap();
    assert_eq!(
        dsa(ahead[1].as_ref()).len(),
        10,
        "rewound to 9, then one row"
    );
}

/// 2026-10-08: Padding rows decode over the workspace's padding buffers: two padding views at
/// position 0 both write row 0 of `pad_k`, so no padding row touches a sequence's cache, and
/// the call allocates nothing (the old padding state was a context-sized cache per padded
/// row per step).
#[test]
fn padding_rows_write_the_workspace_padding_buffers() {
    let rig = Rig::new();
    let layer = rig.layer();
    let meta = rig.meta(2);
    let mut kv = rig.kv();
    let hidden = rig.buf(2 * HIDDEN * 2);
    let mut boxes: Vec<Box<dyn LayerState>> = (0..2)
        .map(|_| Box::new(layer.workspace.pad_state(&cfg())) as Box<dyn LayerState>)
        .collect();
    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
        boxes.iter_mut().map(|b| b.as_mut()).collect();
    let allocs = rig.gpu.live_alloc_count();
    let from = rig.gpu.launch_count();
    layer
        .decode_rows(
            hidden,
            &mut refs,
            &[0, 0],
            &mut kv,
            &meta,
            0,
            &rig.ctx(false),
            7,
        )
        .unwrap();
    assert_eq!(
        rig.gpu.live_alloc_count(),
        allocs,
        "padding rows allocate nothing"
    );
    let knorm = of(&rig.since(from), KNORM);
    assert_eq!(knorm.len(), 2);
    for w in knorm {
        assert_eq!(w.args[0], ptr(layer.workspace.pad_k));
    }
}
