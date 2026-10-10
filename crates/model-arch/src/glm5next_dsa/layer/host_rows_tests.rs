// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host tests of `decode_k`'s host-path staging (`host_rows.rs`) on the mock
//! backend: the slot arithmetic, and that a k-row call uploads its rows in four copies with
//! no stream synchronisation, with the bytes the per-row path wrote.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::row_slots;

use super::super::decode_rows_fixture::*;
use metrale_gpu_runtime::gpu::mock::MockArg;

/// 2026-10-09: Slots follow the block table across a block boundary, and a position past
/// the table is refused.
#[test]
fn row_slots_follow_the_block_table_and_refuse_a_short_one() {
    let slots = row_slots(3, 14, 4, &[7, 2], 16).unwrap();
    assert_eq!(slots, vec![7 * 16 + 14, 7 * 16 + 15, 2 * 16, 2 * 16 + 1]);
    let e = row_slots(3, 30, 3, &[7, 2], 16).unwrap_err().to_string();
    assert!(e.contains("needs logical block 2"), "{e}");
}

fn i64s(b: &[u8], n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| i64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()))
        .collect()
}
fn i32s(b: &[u8], n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect()
}

/// 2026-10-09: A host-path `decode_k` (no metadata) of 1 and of 6 rows uploads four times
/// either way and synchronises nothing; the per-row path copied four times per row, each
/// copy blocking. The staged slots, positions and lengths are the per-row values.
#[test]
fn a_host_path_call_uploads_its_rows_once_without_a_sync() {
    for k in [1usize, 6] {
        let rig = Rig::new();
        let layer = rig.layer();
        let mut kv = rig.kv();
        let hidden = rig.buf(k * HIDDEN * 2);
        let seq_len = 13;
        let mut boxes = rig.states(&[seq_len]);
        let mut ctx = rig.ctx(false);
        ctx.decode_step = false;
        let mut bt = vec![5u32, 1, 3];
        let (h2d, syncs) = (rig.gpu.h2d_count(), rig.gpu.sync_count());
        layer
            .decode_k(
                hidden,
                k,
                boxes[0].as_mut(),
                &mut kv,
                seq_len,
                &mut bt,
                &ctx,
                7,
                false,
            )
            .unwrap();
        assert_eq!(rig.gpu.h2d_count() - h2d, 4, "k={k}: four uploads per call");
        assert_eq!(
            rig.gpu.sync_count(),
            syncs,
            "k={k}: no stream synchronisation"
        );
        let w = &layer.workspace;
        let want = row_slots(3, seq_len, k, &bt, 16).unwrap();
        assert_eq!(i64s(&rig.gpu.read_alloc(w.slot).unwrap(), k), want);
        let pos: Vec<i32> = (0..k).map(|r| (seq_len + r) as i32).collect();
        assert_eq!(i32s(&rig.gpu.read_alloc(w.q_pos).unwrap(), k), pos);
        let lens: Vec<i32> = pos.iter().map(|p| p + 1).collect();
        assert_eq!(i32s(&rig.gpu.read_alloc(w.sl).unwrap(), k), lens);
    }
}

/// 2026-10-09: A 6-row prefill sub-chunk on a paged cache with the batched indexer: one latent
/// write for the six rows (from the staged slots), the key norm once over six staging rows, one
/// rows store at the staged positions through the one staged table (row stride 0), the head
/// weights and the selector query each as one FP32-out batched GEMV, and no per-row indexer
/// launch; the state grows by six rows. Without the lever every row runs its own.
#[test]
fn a_paged_prefill_chunk_stages_its_indexer_rows_once() {
    for batched in [true, false] {
        let rig = Rig::new();
        let layer = rig.layer();
        let mut kv = rig.kv();
        let k = 6;
        let hidden = rig.buf(k * HIDDEN * 2);
        let seq_len = 13;
        let mut st = crate::glm5next_dsa::state::Glm5NextDsaState::paged(&cfg()).unwrap();
        st.advance(seq_len).unwrap();
        let mut ctx = rig.ctx(false);
        ctx.decode_step = false;
        let mut bt = vec![5u32, 1, 3];
        let from = rig.gpu.launch_count();
        layer
            .decode_k_with(
                hidden, k, &mut st, &mut kv, seq_len, &mut bt, &ctx, 7, true, batched,
            )
            .unwrap();
        let l = rig.since(from);
        let w = &layer.workspace;
        let six = MockArg::Bytes((k as u32).to_le_bytes().to_vec());
        if batched {
            let latent = of(&l, LATENT);
            assert_eq!((latent.len(), latent[0].grid[0]), (1, k as u32));
            assert_eq!(latent[0].args[3], ptr(w.slot));
            let knorm = of(&l, KNORM);
            assert_eq!((knorm.len(), knorm[0].grid[0]), (1, k as u32));
            let store = of(&l, STORE_ROWS);
            assert_eq!((store.len(), store[0].grid[1]), (1, k as u32));
            assert_eq!(store[0].args[2], ptr(w.q_pos));
            assert_eq!(store[0].args[6], ptr(w.bt));
            assert_eq!(
                store[0].args[7],
                MockArg::Bytes(0u32.to_le_bytes().to_vec())
            );
            let f32s = of(&l, BATCHM_F32);
            assert_eq!(f32s.len(), 2, "head weights and selector query");
            assert!(f32s.iter().all(|p| p.args[3] == six));
            assert!(of(&l, GEMV_F32).is_empty() && of(&l, STORE).is_empty());
        } else {
            assert_eq!(of(&l, LATENT).len(), k);
            assert_eq!(of(&l, KNORM).len(), k);
            assert!(of(&l, STORE_ROWS).is_empty());
        }
        assert_eq!(st.len(), seq_len + k);
    }
}
