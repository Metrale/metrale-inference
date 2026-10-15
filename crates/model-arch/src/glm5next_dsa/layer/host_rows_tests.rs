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

/// 2026-10-15: `METRALE_GLM_DSA_HOST_ROWS_PINNED`: unset and `0` off, `1` on, anything else
/// an error naming the value.
#[test]
fn the_pinned_lever_parses_zero_and_one_only() {
    assert_eq!(super::parse_pinned(None), Ok(false));
    assert_eq!(super::parse_pinned(Some("0")), Ok(false));
    assert_eq!(super::parse_pinned(Some("1")), Ok(true));
    for bad in ["", "on", "2", "true"] {
        let e = super::parse_pinned(Some(bad)).unwrap_err();
        assert!(e.contains(&format!("{bad:?}")), "{e}");
    }
}

/// 2026-10-15: The ring's steps: the first call writes region 0 without waiting; the same
/// bytes read it again; new bytes go to the other region, waiting only once that region has
/// copies behind it; a region read again from another stream is waited on first.
#[test]
fn the_staging_steps_never_rewrite_the_region_last_read() {
    use super::{Step, step};
    let (w, r) = (
        |region, wait| Step::Write { region, wait },
        |region, wait| Step::Reuse { region, wait },
    );
    assert_eq!(step(None, false, [false; 2], [false; 2]), w(0, false));
    assert_eq!(
        step(Some(0), true, [true, false], [true, false]),
        r(0, false)
    );
    assert_eq!(
        step(Some(0), true, [true, false], [false, false]),
        r(0, true)
    );
    assert_eq!(
        step(Some(0), false, [true, false], [true, false]),
        w(1, false)
    );
    assert_eq!(step(Some(1), false, [true, true], [true, true]), w(0, true));
    assert_eq!(
        step(Some(0), false, [true, true], [true, false]),
        w(1, true)
    );
}

/// 2026-10-15: The workspace slot, position and length arrays and the table prefix after a
/// host-path `decode_k` of `k` rows from `seq_len`, staged or not.
fn staged_bytes(k: usize, seq_len: usize, pinned: bool) -> (Vec<Vec<u8>>, usize, usize) {
    let rig = Rig::new();
    let mut layer = rig.layer();
    if pinned {
        let ws = std::sync::Arc::get_mut(&mut layer.workspace).unwrap();
        ws.host_staging = Some(super::HostRowsStaging::alloc(&rig.gpu, 16, ws.bt_cap).unwrap());
    }
    let mut kv = rig.kv();
    let hidden = rig.buf(k * HIDDEN * 2);
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
    let w = &layer.workspace;
    let read = |p, n| rig.gpu.read_alloc(p).unwrap()[..n].to_vec();
    let bt_n = super::bt_entries_needed(seq_len, k, 16).min(bt.len()) * 4;
    let bytes = vec![
        read(w.slot, k * 8),
        read(w.q_pos, k * 4),
        read(w.sl, k * 4),
        read(w.bt, bt_n),
    ];
    (
        bytes,
        rig.gpu.h2d_count() - h2d,
        rig.gpu.sync_count() - syncs,
    )
}

/// 2026-10-15: The byte gate on the mock: staged and pageable calls leave the same device
/// bytes at the first sub-chunk, a full 16-row one and a prefix-hit continuation, in four
/// uploads and no stream synchronisation; negative control: the rows one position on differ.
#[test]
fn staged_rows_are_the_pageable_bytes() {
    for (k, seq_len) in [(1usize, 0usize), (6, 13), (16, 0), (16, 30)] {
        let (off, off_h2d, off_sync) = staged_bytes(k, seq_len, false);
        let (on, on_h2d, on_sync) = staged_bytes(k, seq_len, true);
        assert_eq!(on, off, "k={k} seq_len={seq_len}: staged bytes differ");
        assert_eq!((on_h2d, on_sync), (4, 0), "k={k} seq_len={seq_len}");
        assert_eq!((off_h2d, off_sync), (4, 0), "k={k} seq_len={seq_len}");
        let (shifted, _, _) = staged_bytes(k, seq_len + 1, true);
        assert_ne!(
            shifted[0], off[0],
            "negctl k={k}: slots one row on must differ"
        );
        assert_ne!(
            shifted[1], off[1],
            "negctl k={k}: positions one row on must differ"
        );
    }
}

/// 2026-10-15: The same bytes again read the region they were written to, without
/// rewriting it; new bytes alternate regions; the device receives every call's bytes; more
/// bytes than the staging holds are refused before anything is enqueued.
#[test]
fn the_staging_ring_reuses_equal_bytes_and_alternates_new_ones() {
    let rig = Rig::new();
    let s = super::HostRowsStaging::alloc(&rig.gpu, 2, 1).unwrap();
    let (d0, d1) = (rig.buf(32), rig.buf(32));
    let up = |a: &[u8], b: &[u8]| {
        s.upload(&rig.gpu, &[(a, d0), (b, d1)], 7).unwrap();
        let ring = s.ring.lock().unwrap();
        assert_eq!(rig.gpu.read_alloc(d0).unwrap()[..a.len()], *a);
        assert_eq!(rig.gpu.read_alloc(d1).unwrap()[..b.len()], *b);
        (
            ring.last,
            ring.filled,
            ring.held(ring.last.unwrap()).to_vec(),
        )
    };
    let (a, b, c) = ([1u8; 8], [2u8; 4], [3u8; 8]);
    let first = up(&a, &b);
    assert_eq!(first.0, Some(0));
    assert_eq!(first.2, [&a[..], &b[..]].concat());
    assert_eq!(up(&a, &b), first, "equal bytes reuse region 0");
    let second = up(&c, &b);
    assert_eq!(second.0, Some(1), "new bytes go to region 1");
    assert_eq!(second.1, [12, 12]);
    assert_eq!(up(&a, &c).0, Some(0), "and back to region 0");
    let h2d = rig.gpu.h2d_count();
    let e = s
        .upload(&rig.gpu, &[(&[0u8; 32], d0), (&[0u8; 9], d1)], 7)
        .unwrap_err()
        .to_string();
    assert!(e.contains("41 B of rows exceed the 36 B staging"), "{e}");
    assert_eq!(rig.gpu.h2d_count(), h2d, "nothing enqueued on refusal");
}

/// 2026-10-15: Lever off (the test process leaves `METRALE_GLM_DSA_HOST_ROWS_PINNED` unset), a
/// DSA workspace allocates no page-locked memory and holds no staging, and the readback says
/// 0 B; with staging it states both regions' bytes.
#[test]
fn the_lever_off_allocates_nothing_and_the_readback_says_so() {
    let rig = Rig::new();
    let before = rig.gpu.host_pinned_alloc_count();
    let layer = rig.layer();
    assert!(layer.workspace.host_staging.is_none());
    assert_eq!(rig.gpu.host_pinned_alloc_count(), before);
    let off = super::staging_readback(None, 16, 4);
    assert!(
        off.contains("=0:") && off.contains("0 B page-locked"),
        "{off}"
    );
    let s = super::HostRowsStaging::alloc(&rig.gpu, 2, 1).unwrap();
    assert_eq!(rig.gpu.host_pinned_alloc_count(), before + 2);
    let on = super::staging_readback(Some(&s), 2, 1);
    assert!(on.contains("=1:") && on.contains("2 x 36 B = 72 B"), "{on}");
}
