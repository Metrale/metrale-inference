// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests of `prefill_span_groups` (moved out of `prefill_spans.rs` to keep it under
//! 500 lines).

use super::*;

fn piece(seq: usize, t0: usize, row: usize, rows: usize) -> SpanPiece {
    SpanPiece { seq, t0, row, rows }
}

/// 2026-10-09: Two ladder prompts share a 512-row group; the third starts the next one,
/// since the three do not fit together.
#[test]
fn short_prompts_pack_whole_into_groups() {
    assert_eq!(
        prefill_span_groups(&[198, 198, 197], &[], 512, |_| Some(())),
        vec![
            vec![piece(0, 0, 0, 198), piece(1, 0, 198, 198)],
            vec![piece(2, 0, 396, 197)],
        ]
    );
}

/// 2026-10-09: A sequence longer than the cap is cut where its own prefill cuts it (every
/// `cap` rows from its first row); a piece that does not fit the open group starts the next
/// one instead of being split.
#[test]
fn a_long_sequence_keeps_its_own_sub_chunks() {
    assert_eq!(
        prefill_span_groups(&[10, 3, 1], &[], 4, |_| Some(())),
        vec![
            vec![piece(0, 0, 0, 4)],
            vec![piece(0, 4, 4, 4)],
            vec![piece(0, 8, 8, 2)],
            vec![piece(1, 0, 10, 3), piece(2, 0, 13, 1)],
        ]
    );
}

/// 2026-10-09: For assorted shapes: every row lands in exactly one piece, in pass order;
/// groups are contiguous and at most `cap` rows; each sequence's pieces are exactly the
/// sub-chunks its single-sequence prefill runs (`t0` a multiple of `cap`, `cap` rows but
/// the last).
#[test]
fn pieces_cover_every_row_once_as_the_single_sequence_sub_chunks() {
    for (rows, cap) in [
        (vec![198usize, 198, 198, 198, 197], 512usize),
        (vec![1, 1, 1], 2),
        (vec![513, 7, 1024, 1], 512),
        (vec![5], 1),
        (vec![3, 0, 4], 0),
    ] {
        let groups = prefill_span_groups(&rows, &[], cap, |_| Some(()));
        let cap = cap.max(1);
        let mut next_row = 0usize;
        let mut next_t = vec![0usize; rows.len()];
        for g in &groups {
            assert!(!g.is_empty());
            assert!(
                g.iter().map(|p| p.rows).sum::<usize>() <= cap,
                "{rows:?} cap {cap}"
            );
            for p in g {
                assert_eq!(p.row, next_row, "{rows:?} cap {cap}: rows out of order");
                assert_eq!(
                    p.t0, next_t[p.seq],
                    "{rows:?} cap {cap}: sequence rows skipped"
                );
                assert_eq!(
                    p.t0 % cap,
                    0,
                    "{rows:?} cap {cap}: piece off the sub-chunk grid"
                );
                assert_eq!(p.rows, cap.min(rows[p.seq] - p.t0), "{rows:?} cap {cap}");
                next_row += p.rows;
                next_t[p.seq] += p.rows;
            }
        }
        assert_eq!(next_row, rows.iter().sum::<usize>());
        assert_eq!(next_t, rows);
    }
}

/// 2026-10-11: A sequence with capture points is cut there as well, and its sub-chunk grid
/// restarts at each point, as its own prefill splits the pass there; for assorted shapes
/// every sequence's pieces are exactly the sub-chunks of separate passes ending and starting
/// at its points. Points outside the sequence cut nothing.
#[test]
fn capture_points_cut_a_sequence_into_separate_passes_sub_chunks() {
    assert_eq!(
        prefill_span_groups(&[10, 6], &[vec![6], vec![]], 4, |_| Some(())),
        vec![
            vec![piece(0, 0, 0, 4)],
            vec![piece(0, 4, 4, 2)],
            vec![piece(0, 6, 6, 4)],
            vec![piece(1, 0, 10, 4)],
            vec![piece(1, 4, 14, 2)],
        ]
    );
    for (rows, cuts, cap) in [
        (
            vec![4000usize, 300],
            vec![vec![3968usize, 3984], vec![]],
            2048usize,
        ),
        (vec![50, 50], vec![vec![16], vec![32, 48]], 16),
        (vec![7, 9], vec![vec![0, 7, 9], vec![1]], 4),
        (vec![2048, 17], vec![vec![2048], vec![16]], 2048),
    ] {
        let groups = prefill_span_groups(&rows, &cuts, cap, |_| Some(()));
        for (s, &n) in rows.iter().enumerate() {
            let mut want = Vec::new();
            let mut starts: Vec<usize> = cuts[s]
                .iter()
                .copied()
                .filter(|&c| c > 0 && c < n)
                .collect();
            starts.insert(0, 0);
            starts.push(n);
            for w in starts.windows(2) {
                let mut t = w[0];
                while t < w[1] {
                    let k = cap.min(w[1] - t);
                    want.push((t, k));
                    t += k;
                }
            }
            let got: Vec<(usize, usize)> = groups
                .iter()
                .flatten()
                .filter(|p| p.seq == s)
                .map(|p| (p.t0, p.rows))
                .collect();
            assert_eq!(got, want, "{rows:?} {cuts:?} cap {cap}: sequence {s}");
        }
    }
}

/// 2026-10-09: The serve's routed site shares only from 17 rows (the grouped GEMM): a
/// shorter piece runs alone, and a piece may not join a group whose total the predicate
/// refuses.
#[test]
fn a_piece_the_predicate_refuses_runs_alone() {
    let grouped = |r: usize| (r >= 17).then_some(());
    assert_eq!(
        prefill_span_groups(&[77, 9, 77, 77, 16, 20], &[], 512, grouped),
        vec![
            vec![piece(0, 0, 0, 77)],
            vec![piece(1, 0, 77, 9)],
            vec![piece(2, 0, 86, 77), piece(3, 0, 163, 77)],
            vec![piece(4, 0, 240, 16)],
            vec![piece(5, 0, 256, 20)],
        ]
    );
    let below_300 = |r: usize| (r < 300).then_some(());
    assert_eq!(
        prefill_span_groups(&[200, 150, 50], &[], 512, below_300),
        vec![
            vec![piece(0, 0, 0, 200)],
            vec![piece(1, 0, 200, 150), piece(2, 0, 350, 50)],
        ]
    );
}

/// 2026-10-14: Pieces of two shareable classes never meet in a group, even when the group's
/// total has one piece's class: a 20-row piece (class `false`, under 50 rows) does not join a
/// 50-row piece (class `true`), nor a 60-row piece (class `true`) a group of a 10-row one
/// although their 70 rows read `true`; two pieces of one class whose total changes class
/// (30 + 40) stay apart.
#[test]
fn pieces_of_different_classes_never_share_a_group() {
    let class = |r: usize| Some(r >= 50);
    assert_eq!(
        prefill_span_groups(&[50, 20, 20, 30, 40, 10, 60], &[], 512, class),
        vec![
            vec![piece(0, 0, 0, 50)],
            vec![piece(1, 0, 50, 20), piece(2, 0, 70, 20)],
            vec![piece(3, 0, 90, 30)],
            vec![piece(4, 0, 120, 40)],
            vec![piece(5, 0, 160, 10)],
            vec![piece(6, 0, 170, 60)],
        ]
    );
}
