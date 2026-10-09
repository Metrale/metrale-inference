// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Speculative-verify rollback by replay (`--ssm-rollback-mode replay`) at the model
//! level: whether every recurrent layer can take it, and the commit that asks each layer to
//! rebuild its state from the slot's checkpoint and verify record.
//!
//! Owner: model-engine (SSM state).
//! Invariants:
//! - A replay-mode verify runs only when every pool-backed recurrent layer supports replay
//!   (`require_verify_rollback`); the commit then finds a replay commit on each of them.
//! - The replay launches run on the default stream, in order with every forward, because they
//!   use the layers' shared forward workspace as scratch.

use anyhow::{Result, bail};
use metrale_config::LayerType;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-08: The pool-backed recurrent layers, by index.
    fn ssm_pool_layers(
        &self,
    ) -> impl Iterator<Item = (usize, &dyn metrale_model_layers::layer::TransformerLayer)> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                self.config.layer_type(*i) == LayerType::LinearAttention && l.uses_ssm_pool()
            })
            .map(|(i, l)| (i, l.as_ref()))
    }

    /// 2026-10-08: The guard every `decode_verify*` entry runs: under the replay rollback mode,
    /// a verify needs every pool-backed recurrent layer to support replay.
    pub(super) fn require_verify_rollback(&self) -> Result<()> {
        let wired = self.ssm_pool_layers().all(|(_, l)| l.supports_ssm_replay());
        self.ssm_pool.require_verify_rollback_supported(wired)
    }

    /// 2026-10-08: Commit `num_accepted` of the `k` verified rows (anchor included,
    /// `0 < num_accepted < k`) under the replay rollback mode: each recurrent layer restores its
    /// checkpoint and replays its recorded rows. Errors when a layer has no replay commit.
    pub(super) fn commit_replay_prefix(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let layers: Vec<usize> = self.ssm_pool_layers().map(|(i, _)| i).collect();
        for i in layers {
            if !self.layers[i].ssm_replay_commit(
                self.gpu.as_ref(),
                seq.layer_states[i].as_mut(),
                num_accepted,
                k,
                stream,
            )? {
                bail!(
                    "commit_accepted_prefix: layer {i} has no replay commit, but the pool runs \
                     --ssm-rollback-mode replay (require_verify_rollback admits a verify only \
                     when every recurrent layer has one)"
                );
            }
        }
        Ok(())
    }
}
