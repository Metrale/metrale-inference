// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The GLM layer's batched speculative verify (`decode_verify_multi`): `ks[i]` rows
//! for each of `n` sequences in one pass, the rows sequence-major, through `forward_spans`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Sequence `i`'s rows are exactly the rows its single-sequence verify (`decode_batched`)
//!   runs: positions `seq_lens[i] + t`, highway slot = the row's index in the whole pass, its
//!   KDA state stepped in row order and its DSA rows written and selected in row order.
//! - A KDA layer keeps what the single-sequence verify keeps for the same pool: the state
//!   after each row `t < k - 1` when the pool has per-row intermediates, else (replay
//!   rollback) one checkpoint taken before the pass and each such row's recurrent inputs, so
//!   `commit_accepted_prefix` rebuilds the accepted state the same way after either verify.

use super::multi_seq::KdaRowKeep;
use super::*;

impl Glm5NextLayer {
    /// 2026-10-09: The batched verify of `ks[i]` rows for each sequence `i` (`states[i]`, length
    /// `seq_lens[i]` before the verify). Errors before any launch when the counts disagree, a
    /// sequence has no row, or a KDA sequence can keep neither snapshots nor a record for its
    /// rows; the replay checkpoints are taken before the first row runs.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        seq_lens: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if ks.is_empty() || ks.contains(&0) {
            bail!(
                "GLM layer {}: a batched verify needs at least one row per sequence, got {ks:?}",
                self.layer_idx
            );
        }
        let keep = match &self.mixer {
            Glm5NextMixer::Kda { layer, .. } => {
                // 2026-10-09: Every sequence's plan is decided before any checkpoint is taken,
                // so a refusal leaves every state as it was.
                let mut replay = Vec::with_capacity(ks.len());
                let mut keep = Vec::with_capacity(ks.len());
                for (s, &k) in states.iter_mut().zip(ks) {
                    let st = self.kda_state(&mut **s)?;
                    let snapshots = st.h_state_intermediates.len() + 1 >= k
                        && st.conv_state_intermediates.len() + 1 >= k;
                    if k == 1 {
                        keep.push(KdaRowKeep::Nothing);
                        replay.push(false);
                    } else if snapshots {
                        keep.push(KdaRowKeep::Snapshots(
                            (0..k - 1)
                                .map(|t| {
                                    (st.h_state_intermediates[t], st.conv_state_intermediates[t])
                                })
                                .collect(),
                        ));
                        replay.push(false);
                    } else if st.replay_ring.is_some() {
                        let (_, _, record) = self.replay_parts(layer, st)?;
                        if record.rows() + 1 < k {
                            bail!(
                                "GLM layer {}: a {k}-row verify records {} rows, but the slot's \
                                 verify record holds {}",
                                self.layer_idx,
                                k - 1,
                                record.rows()
                            );
                        }
                        keep.push(KdaRowKeep::Record {
                            record,
                            rows: k - 1,
                        });
                        replay.push(true);
                    } else {
                        bail!(
                            "GLM layer {}: a {k}-row batched verify needs {} per-row state \
                             snapshots or a replay record (--ssm-rollback-mode replay); the \
                             pool has h={} conv={} and no record",
                            self.layer_idx,
                            k - 1,
                            st.h_state_intermediates.len(),
                            st.conv_state_intermediates.len(),
                        );
                    }
                }
                for (s, (&k, &r)) in states.iter_mut().zip(ks.iter().zip(&replay)) {
                    if r {
                        let st = self.kda_state(&mut **s)?;
                        self.replay_verify_prepare(layer, st, k, ctx, stream)?;
                    }
                }
                Some(keep)
            }
            Glm5NextMixer::Dsa(_) => None,
        };
        self.forward_spans(
            hidden,
            ks,
            seq_lens,
            states,
            kv_cache,
            keep.as_deref(),
            ctx,
            stream,
        )
    }
}
