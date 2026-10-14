// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Which row groups of `forward_spans` take the multi-sequence KDA verify
//! (`Glm5NextKdaLayer::decode_seqs_tokens`, `METRALE_GLM_KDA_SEQS_TOKENS=1`) and with what
//! runs. A pure plan: every group it turns down keeps `decode_seq_rows`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A plan's snapshots are exactly the ones `kda_after_row` would take for the group: the
//!   state after sequence row `t` for each `t` below the snapshot count inside the span.

use super::multi_seq::{GroupSpan, KdaRowKeep};
use crate::glm5next_kda::KdaSeqState;
use crate::glm5next_kda::seqs_tokens::KdaSeqRun;
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-10-14: Each sequence's snapshot list under `keep` (`None`: it keeps none).
pub(in crate::glm5next_layer) fn snapshot_lists(
    keep: Option<&[KdaRowKeep]>,
) -> Vec<Option<&[(DevicePtr, DevicePtr)]>> {
    keep.unwrap_or(&[])
        .iter()
        .map(|k| match k {
            KdaRowKeep::Snapshots(s) => Some(s.as_slice()),
            _ => None,
        })
        .collect()
}

/// 2026-10-14: The runs of a group whose spans step `kda[span.seq]`, keeping the state after
/// sequence row `t` in `snaps[span.seq][t]` (a missing or `None` list keeps none), or `None`
/// when the group should keep `decode_seq_rows`: no span has two rows (a batched decode), a
/// span asks for more than one snapshot (the seqs kernels store the state after a run's first
/// row only), or two spans step one state (a padded batch's dummy slot).
pub fn kda_seqs_runs(
    spans: &[GroupSpan],
    snaps: &[Option<&[(DevicePtr, DevicePtr)]>],
    kda: &[KdaSeqState],
) -> Option<Vec<KdaSeqRun>> {
    if !spans.iter().any(|sp| sp.rows > 1) {
        return None;
    }
    let mut runs: Vec<KdaSeqRun> = Vec::with_capacity(spans.len());
    for sp in spans {
        let state = *kda.get(sp.seq)?;
        if runs
            .iter()
            .any(|r| r.state.recurrent.0 == state.recurrent.0 || r.state.conv.0 == state.conv.0)
        {
            return None;
        }
        let snapshot = match snaps.get(sp.seq).copied().flatten() {
            Some(list) => match list.len().min(sp.t0 + sp.rows).saturating_sub(sp.t0) {
                0 => None,
                1 => Some(list[sp.t0]),
                _ => return None,
            },
            None => None,
        };
        runs.push(KdaSeqRun {
            state,
            rows: sp.rows,
            snapshot,
        });
    }
    Some(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-14: `kda_seqs_runs` over `keep` as `forward_spans` passes it.
    fn plan(
        spans: &[GroupSpan],
        keep: Option<&[KdaRowKeep]>,
        kda: &[KdaSeqState],
    ) -> Option<Vec<KdaSeqRun>> {
        kda_seqs_runs(spans, &snapshot_lists(keep), kda)
    }

    fn st(i: u64) -> KdaSeqState {
        KdaSeqState {
            conv: DevicePtr(0x1000 + i),
            recurrent: DevicePtr(0x2000 + i),
        }
    }

    fn snap(i: u64) -> (DevicePtr, DevicePtr) {
        (DevicePtr(0x3000 + i), DevicePtr(0x4000 + i))
    }

    fn span(seq: usize, row0: usize, t0: usize, rows: usize) -> GroupSpan {
        GroupSpan {
            seq,
            row0,
            t0,
            rows,
        }
    }

    fn shape(runs: &[KdaSeqRun]) -> Vec<(u64, usize, Option<u64>)> {
        runs.iter()
            .map(|r| (r.state.recurrent.0, r.rows, r.snapshot.map(|s| s.0.0)))
            .collect()
    }

    /// 2026-10-14: k = 2 verifies under snapshot rollback: each run keeps its one
    /// intermediate (the state after row 0), on its own state, in span order.
    #[test]
    fn k2_snapshot_verify_keeps_each_runs_first_row() {
        let kda: Vec<KdaSeqState> = (0..3).map(st).collect();
        let keep: Vec<KdaRowKeep> = (0..3)
            .map(|i| KdaRowKeep::Snapshots(vec![snap(i)]))
            .collect();
        let spans = [span(0, 0, 0, 2), span(1, 2, 0, 2), span(2, 4, 0, 2)];
        let runs = plan(&spans, Some(&keep), &kda).unwrap();
        assert_eq!(
            shape(&runs),
            vec![
                (0x2000, 2, Some(0x3000)),
                (0x2001, 2, Some(0x3001)),
                (0x2002, 2, Some(0x3002))
            ]
        );
        assert_eq!(runs[1].snapshot.unwrap().1, DevicePtr(0x4001));
    }

    /// 2026-10-14: A span straddling two groups: in the first group its row 0 takes the
    /// snapshot; in the second, rows 1.. take none (row 1 is the last, never snapshotted).
    #[test]
    fn a_straddling_span_snapshots_only_where_its_row_zero_lands() {
        let kda: Vec<KdaSeqState> = (0..2).map(st).collect();
        let keep: Vec<KdaRowKeep> = (0..2)
            .map(|i| KdaRowKeep::Snapshots(vec![snap(i)]))
            .collect();
        let first = [span(0, 0, 0, 2), span(1, 2, 0, 1)];
        let runs = plan(&first, Some(&keep), &kda).unwrap();
        assert_eq!(
            shape(&runs),
            vec![(0x2000, 2, Some(0x3000)), (0x2001, 1, Some(0x3001))]
        );
        let second = [span(1, 0, 1, 1), span(0, 1, 0, 2)];
        let runs = plan(&second, Some(&keep), &kda).unwrap();
        assert_eq!(
            shape(&runs),
            vec![(0x2001, 1, None), (0x2000, 2, Some(0x3000))]
        );
    }

    /// 2026-10-14: Groups it turns down: a decode (no span of two rows), a span needing two
    /// snapshots (k = 3), a shared state (padding), and a sequence without a state.
    #[test]
    fn decodes_multi_snapshot_spans_and_shared_states_fall_back() {
        let kda: Vec<KdaSeqState> = (0..3).map(st).collect();
        let ones = [span(0, 0, 0, 1), span(1, 1, 0, 1)];
        assert!(plan(&ones, None, &kda).is_none());
        let keep3 = vec![
            KdaRowKeep::Snapshots(vec![snap(0), snap(1)]),
            KdaRowKeep::Nothing,
        ];
        let k3 = [span(0, 0, 0, 3), span(1, 3, 0, 3)];
        assert!(plan(&k3, Some(&keep3), &kda).is_none());
        let shared = [st(0), st(0)];
        let k2 = [span(0, 0, 0, 2), span(1, 2, 0, 2)];
        assert!(plan(&k2, None, &shared).is_none());
        let conv_shared = [
            st(0),
            KdaSeqState {
                conv: st(0).conv,
                recurrent: st(1).recurrent,
            },
        ];
        assert!(plan(&k2, None, &conv_shared).is_none());
        assert!(plan(&[span(5, 0, 0, 2)], None, &kda).is_none());
    }

    /// 2026-10-14: The last row of a k-row span is never snapshotted (k - 1 snapshots), so a
    /// span that starts on it takes none, and a plan without a keep takes none at all.
    #[test]
    fn rows_past_the_snapshot_count_and_no_keep_take_no_snapshot() {
        let kda: Vec<KdaSeqState> = (0..2).map(st).collect();
        let keep = vec![KdaRowKeep::Snapshots(vec![snap(0)]), KdaRowKeep::Nothing];
        let spans = [span(0, 0, 1, 1), span(1, 1, 0, 2)];
        let runs = plan(&spans, Some(&keep), &kda).unwrap();
        assert_eq!(shape(&runs), vec![(0x2000, 1, None), (0x2001, 2, None)]);
        let runs = plan(&[span(0, 0, 0, 2)], None, &kda).unwrap();
        assert_eq!(shape(&runs), vec![(0x2000, 2, None)]);
    }
}
