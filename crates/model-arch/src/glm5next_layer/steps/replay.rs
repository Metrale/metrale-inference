// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM layer's side of a replay-mode verify (`--ssm-rollback-mode replay`): the
//! KDA state, checkpoint and verify record a slot's `SsmLayerState` carries, and the checkpoint
//! taken before the verify rows run.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the checkpoint is taken before `forward_k` updates the state in place, on the
//! forward's stream.

use super::*;
use crate::glm5next_kda::KdaVerifyRecord;

impl Glm5NextLayer {
    /// 2026-10-08: The live KDA state, its checkpoint and the verify record of a replay-mode
    /// slot. Errors when the slot lacks the checkpoint or the record.
    pub(in crate::glm5next_layer) fn replay_parts(
        &self,
        layer: &Glm5NextKdaLayer,
        st: &SsmLayerState,
    ) -> Result<(KdaSeqState, KdaSeqState, KdaVerifyRecord)> {
        let (Some(h_ckpt), Some(conv_ckpt), Some(ring)) = (
            st.h_state_checkpoint,
            st.conv_state_checkpoint,
            st.replay_ring,
        ) else {
            bail!(
                "GLM layer {}: a replay-mode verify needs the slot's checkpoint and verify record \
                 (checkpoint h={:?} conv={:?}, record={:?})",
                self.layer_idx,
                st.h_state_checkpoint,
                st.conv_state_checkpoint,
                st.replay_ring
            );
        };
        Ok((
            KdaSeqState {
                conv: st.conv_state,
                recurrent: st.h_state,
            },
            KdaSeqState {
                conv: conv_ckpt,
                recurrent: h_ckpt,
            },
            KdaVerifyRecord::new(&layer.cfg, ring.base, ring.bytes),
        ))
    }

    /// 2026-10-08: Before a `k`-row replay-mode verify: copy the state to the checkpoint and
    /// return the record the rows will go to. `Ok(None)` when the slot has no record (snapshot
    /// mode); errors when the record holds fewer than `k - 1` rows.
    pub(in crate::glm5next_layer) fn replay_verify_prepare(
        &self,
        layer: &Glm5NextKdaLayer,
        st: &SsmLayerState,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<KdaVerifyRecord>> {
        if st.replay_ring.is_none() {
            return Ok(None);
        }
        let (live, checkpoint, record) = self.replay_parts(layer, st)?;
        if record.rows() + 1 < k {
            bail!(
                "GLM layer {}: a {k}-row verify records {} rows, but the slot's verify record \
                 holds {}; the pool was sized for a narrower verify",
                self.layer_idx,
                k - 1,
                record.rows()
            );
        }
        layer.checkpoint_state(ctx.gpu, &live, &checkpoint, stream)?;
        Ok(Some(record))
    }
}
