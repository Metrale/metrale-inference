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
//! - 2026-10-09: When graphs are allowed (the verify graphs' rule) and the sequence has an SSM
//!   slot, a commit is captured once per `(slot, accepted, k)` and replayed after; its
//!   launches and addresses depend only on that key.

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
        let key = seq.ssm_slot_idx().map(|s| (s, num_accepted, k));
        let graphs_on =
            (self.comm.is_none() || self.levers.ep_graphs || self.layers_capture_with_comm())
                && !self
                    .suppress_graphs
                    .load(std::sync::atomic::Ordering::Relaxed)
                && std::env::var_os("METRALE_NO_REPLAY_COMMIT_GRAPHS").is_none();
        let Some(key) = key.filter(|_| graphs_on) else {
            return self.commit_replay_layers(seq, num_accepted, k, stream);
        };
        let mut cache = self.replay_commit_graphs.lock();
        if let Some(&graph) = cache.get(&key) {
            return self.gpu.launch_graph(graph, stream);
        }
        self.gpu.begin_capture(stream)?;
        let ran = self.commit_replay_layers(seq, num_accepted, k, stream);
        let graph = self.gpu.end_capture(stream)?;
        ran?;
        if graph.0 != 0 {
            cache.insert(key, graph);
            self.gpu.launch_graph(graph, stream)?;
        }
        Ok(())
    }

    /// 2026-10-09: The per-layer replay commits of `commit_replay_prefix`, launched on `stream`.
    fn commit_replay_layers(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
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
