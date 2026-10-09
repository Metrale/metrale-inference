// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host tests of `decode_k`'s host-path staging (`host_rows.rs`) on the mock
//! backend: the slot arithmetic, and that a k-row call uploads its rows in four copies with
//! no stream synchronisation, with the bytes the per-row path wrote.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::row_slots;

use super::super::decode_rows_fixture::*;

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
