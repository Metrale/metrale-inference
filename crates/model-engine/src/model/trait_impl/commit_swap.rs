// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_SSM_COMMIT_SWAP` (default on, `0` off): a speculative verify that keeps
//! only its first row commits by swapping the slot's live state with intermediate 0
//! (`ssm_reserve::commit_swap`) instead of copying intermediate 0 over the live state.
//!
//! Where the swap is visible:
//! - The pool's accessors resolve the role bit, and the sequence's cached pointers are swapped
//!   with it, so every reader that takes an address at launch time (eager kernels, the copy
//!   plans, snapshots, the prefix cache, compaction) reads the bytes the copy would have left.
//! - A CUDA graph bakes addresses: the single-sequence slot graphs (`SlotGraphKey`) carry the
//!   role bit in their key, and only those verifies arm a swap (`arm_commit_swap`).
//! - The multi-sequence graphs key on slot vectors and are not role-aware: a batched step first
//!   moves a swapped slot back home ([`TransformerModel::unswap_for_batch`], one copy, the bytes
//!   the commit used to move) and drops its arm, so a batched verify commits by copy as before.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - The live state after a commit holds the same bytes with the lever on or off; only which
//!   buffer holds them differs.
//! - A swap moves the pool's role bit and every pool-backed SSM layer pointer of the sequence
//!   together, or neither.

use anyhow::{Result, ensure};
use metrale_config::LayerType;
use metrale_model_layers::layer::SsmLayerState;

use super::super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-21: The slot-graph key of a step on `slot`, in the slot's current commit-swap
    /// role: a graph captured with the home buffers never replays against swapped ones.
    pub(super) fn slot_graph_key(
        &self,
        slot: usize,
        seq_slot: metrale_gpu_runtime::gpu::DevicePtr,
        route: metrale_model_layers::layer::MoeLoraRoute,
    ) -> super::SlotGraphKey {
        super::SlotGraphKey::new(slot, self.ssm_pool.is_swapped(slot), seq_slot, route)
    }

    /// 2026-10-21: A single-sequence verify of `slot` ran through a graph keyed by the role
    /// bit: its commit may swap.
    pub(in crate::model) fn arm_commit_swap(&self, slot: usize) {
        self.ssm_pool.swaps.arm(slot);
    }

    /// 2026-10-21: Commit `num_accepted` of `k` rows of `seq`'s armed verify by swap when it
    /// keeps only the first row; `Ok(false)` with nothing changed otherwise (the copy commit
    /// then runs). Consumes the slot's arm either way.
    pub(super) fn try_commit_by_swap(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<bool> {
        let slot = seq.slot_idx;
        let armed = self.ssm_pool.swaps.take_arm(slot);
        if !(armed
            && self.settings.scheduler.ssm_commit_swap
            && num_accepted == 1
            && k >= 2
            && self.dflash_hidden_save.is_none()
            && self.ssm_pool.can_swap(slot)
            && !self.gdn_woa_folded_slots.lock().contains(&slot)
            && !self.gdn_carry_has_committed(slot)
            && self.pointers_match_pool(seq)?)
        {
            return Ok(false);
        }
        self.ssm_pool.swap_roles(slot)?;
        self.for_each_pool_layer(seq, |_, ssm| ssm.swap_live_with_inter0())?;
        Ok(true)
    }

    /// 2026-10-21: Before a multi-sequence step: a swapped slot's live state is copied into its
    /// home buffer and the slot returns to the home layout (the batched graphs bake home
    /// addresses); the arm is dropped. On the default stream, after the secondary stream's
    /// pending commit copies. No-op for a home slot or a sequence without an SSM slot.
    pub(in crate::model) fn unswap_for_batch(&self, seq: &mut SequenceState) -> Result<()> {
        let Some(slot) = seq.ssm_slot_idx() else {
            return Ok(());
        };
        self.ssm_pool.swaps.take_arm(slot);
        if !self.ssm_pool.is_swapped(slot) {
            return Ok(());
        }
        ensure!(
            self.pointers_match_pool(seq)?,
            "SSM commit swap: slot {slot}'s sequence pointers disagree with the pool"
        );
        let (mut h_plan, mut conv_plan) = (Vec::new(), Vec::new());
        self.for_each_pool_layer(seq, |_, ssm| {
            h_plan.push(StateCopy {
                src: ssm.h_state,
                dst: ssm.h_state_intermediates[0],
                bytes: self.ssm_pool.h_stored_bytes,
            });
            conv_plan.push(StateCopy {
                src: ssm.conv_state,
                dst: ssm.conv_state_intermediates[0],
                bytes: self.ssm_pool.conv_bytes,
            });
            Ok(())
        })?;
        let stream = self.gpu.default_stream();
        self.gpu.stream_wait_event(stream, self.secondary_event)?;
        run_ssm_state_copies(
            self.gpu.as_ref(),
            &h_plan,
            &conv_plan,
            !self.settings.scheduler.no_batched_ssm_rollback,
            stream,
        )?;
        self.ssm_pool.swap_roles(slot)?;
        self.for_each_pool_layer(seq, |_, ssm| ssm.swap_live_with_inter0())
    }

    /// 2026-10-21: Whether every SSM layer of `seq` caches the pool's current live and
    /// intermediate-0 addresses: the precondition of moving them together.
    fn pointers_match_pool(&self, seq: &mut SequenceState) -> Result<bool> {
        let slot = seq.slot_idx;
        let mut all = true;
        self.for_each_pool_layer(seq, |i, ssm| {
            all &= ssm.h_state == self.ssm_pool.h_state(i, slot)
                && ssm.conv_state == self.ssm_pool.conv_state(i, slot)
                && ssm.h_state_intermediates.first()
                    == Some(&self.ssm_pool.h_intermediate(i, slot, 0))
                && ssm.conv_state_intermediates.first()
                    == Some(&self.ssm_pool.conv_intermediate(i, slot, 0));
            Ok(())
        })?;
        Ok(all)
    }

    /// 2026-10-21: `f(ssm layer index, state)` over `seq`'s linear-attention layers in order,
    /// the layers `ssm_commit_plan` walks.
    fn for_each_pool_layer(
        &self,
        seq: &mut SequenceState,
        mut f: impl FnMut(usize, &mut SsmLayerState) -> Result<()>,
    ) -> Result<()> {
        let mut ssm_idx = 0usize;
        for (i, st) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = st
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            f(ssm_idx, ssm)?;
            ssm_idx += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "commit_swap_tests.rs"]
mod tests;
