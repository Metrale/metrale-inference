// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Which row groups of `forward_spans` take the multi-sequence KDA verify
//! (`Glm5NextKdaLayer::decode_seqs_tokens`, `METRALE_GLM_KDA_SEQS_TOKENS=1`) and with what
//! runs. A pure plan: every group it turns down keeps `decode_seq_rows`.
//! 2026-10-16: A span keeps every snapshot its rows owe (up to `KDA_SEQS_SNAP_MAX`), so a
//! verify of three or more rows per sequence takes the seqs kernels too. A plan that does not
//! add up (a snapshot list that is not `k - 1` long, a span outside its sequence, a sequence
//! without a state) is refused, not walked; a group that legitimately walks says why, and the
//! serve logs each reason the first time it happens.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A plan's snapshots are exactly the ones `kda_after_row` would take for the group: the
//!   state after sequence row `t` for each `t` below the snapshot count inside the span.

use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::{Result, bail};

use super::multi_seq::{GroupSpan, KdaRowKeep};
use crate::glm5next_kda::KdaSeqState;
use crate::glm5next_kda::seqs_tokens::{KDA_SEQS_SNAP_MAX, KdaDefer, KdaSeqRun, KdaSnaps};
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-10-16: Why a group keeps the per-row walk with the lever on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqsWalk {
    /// No span has two rows: a batched decode, not a verify. Not a fallback; never logged.
    Decode,
    /// Two spans step one state (a padded batch's dummy slot).
    SharedState,
    /// A span keeps more than `KDA_SEQS_SNAP_MAX` snapshots.
    SnapshotsOverCap,
    /// The seqs kernels are not in this build, or the config does not fit them.
    NotReady,
}

impl SeqsWalk {
    fn reason(self) -> &'static str {
        match self {
            Self::Decode => "a batched decode",
            Self::SharedState => "two spans step one state",
            Self::SnapshotsOverCap => "a span keeps more snapshots than the kernels take",
            Self::NotReady => "the seqs kernels cannot run here",
        }
    }
}

/// 2026-10-16: A group's plan: the seqs kernels' runs, or the per-row walk and why.
#[derive(Debug)]
pub enum SeqsPlan {
    Runs(Vec<KdaSeqRun>),
    Walk(SeqsWalk),
}

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
/// sequence row `t` in `snaps[span.seq][t]` (a missing or `None` list keeps none).
/// 2026-10-16: `ks[s]` is sequence `s`'s rows in the whole pass; a list must hold `ks[s] - 1`
/// slots (the verify's intermediates). Refused (`Err`): a span with no state, outside its
/// sequence's rows, or on a list of another length. Walked ([`SeqsWalk`]): no span of two rows,
/// two spans on one state, a span owing more than `KDA_SEQS_SNAP_MAX` snapshots.
/// 2026-10-16: A span that starts its sequence's rows takes `defer[span.seq]` (SNAPZ; a
/// missing entry takes none): its first launch reads the state there, so only the group
/// holding row 0 may.
pub fn kda_seqs_runs(
    spans: &[GroupSpan],
    ks: &[usize],
    snaps: &[Option<&[(DevicePtr, DevicePtr)]>],
    kda: &[KdaSeqState],
    defer: &[Option<KdaDefer>],
) -> Result<SeqsPlan> {
    let mut owed: Vec<&[(DevicePtr, DevicePtr)]> = Vec::with_capacity(spans.len());
    for sp in spans {
        let (Some(&k), Some(_)) = (ks.get(sp.seq), kda.get(sp.seq)) else {
            bail!(
                "KDA seqs plan: span of sequence {} with {} row counts and {} states",
                sp.seq,
                ks.len(),
                kda.len()
            );
        };
        if sp.rows == 0 || sp.t0 + sp.rows > k {
            bail!(
                "KDA seqs plan: sequence {} rows {}..{} outside its {k}",
                sp.seq,
                sp.t0,
                sp.t0 + sp.rows
            );
        }
        let list = snaps.get(sp.seq).copied().flatten().unwrap_or(&[]);
        if snaps.get(sp.seq).copied().flatten().is_some() && list.len() + 1 != k {
            bail!(
                "KDA seqs plan: sequence {} keeps {} snapshots for a {k}-row verify, not {}; \
                 refused",
                sp.seq,
                list.len(),
                k - 1
            );
        }
        owed.push(&list[sp.t0.min(list.len())..(sp.t0 + sp.rows).min(list.len())]);
    }
    if !spans.iter().any(|sp| sp.rows > 1) {
        return Ok(SeqsPlan::Walk(SeqsWalk::Decode));
    }
    let mut runs: Vec<KdaSeqRun> = Vec::with_capacity(spans.len());
    for (sp, at) in spans.iter().zip(owed) {
        let state = kda[sp.seq];
        if runs
            .iter()
            .any(|r| r.state.recurrent.0 == state.recurrent.0 || r.state.conv.0 == state.conv.0)
        {
            return Ok(SeqsPlan::Walk(SeqsWalk::SharedState));
        }
        if at.len() > KDA_SEQS_SNAP_MAX {
            return Ok(SeqsPlan::Walk(SeqsWalk::SnapshotsOverCap));
        }
        runs.push(KdaSeqRun {
            state,
            rows: sp.rows,
            snapshots: KdaSnaps::new(at)?,
            defer: defer.get(sp.seq).copied().flatten().filter(|_| sp.t0 == 0),
        });
    }
    Ok(SeqsPlan::Runs(runs))
}

/// 2026-10-16: Whether `w` is the first walk of its reason in `seen` (one bit per reason); a
/// decode is never one.
fn first_walk(seen: &AtomicU8, w: SeqsWalk) -> bool {
    if w == SeqsWalk::Decode {
        return false;
    }
    let bit = 1u8 << (w as u8);
    seen.fetch_or(bit, Ordering::Relaxed) & bit == 0
}

/// 2026-10-16: The runs of `plan`, or `None` for the per-row walk, logging the first walk of
/// each reason in the process: the serve log's proof that a lever set on did not go inert.
pub(in crate::glm5next_layer) fn runs_or_logged_walk(plan: SeqsPlan) -> Option<Vec<KdaSeqRun>> {
    static SEEN: AtomicU8 = AtomicU8::new(0);
    match plan {
        SeqsPlan::Runs(runs) => Some(runs),
        SeqsPlan::Walk(w) => {
            if first_walk(&SEEN, w) {
                tracing::warn!(
                    "METRALE_GLM_KDA_SEQS_TOKENS: a verify group keeps the per-row walk \
                     (rank-local, first of this reason): {}",
                    w.reason()
                );
            }
            None
        }
    }
}

#[cfg(test)]
#[path = "kda_seqs_tests.rs"]
mod tests;
