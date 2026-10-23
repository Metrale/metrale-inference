// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `Glm5NextLayer`'s capability, replay, graph and aux-state trait impls, split out
//! of `mod.rs` (file-size cap) by exact copy.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: those of the parent module.

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerWeightSetup, LayerWriteOnAccept,
};

use super::{Glm5NextLayer, Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_dsa::paged::IndexerCache;
use crate::glm5next_dsa::state::Glm5NextDsaState;

impl LayerCapabilities for Glm5NextLayer {
    /// 2026-09-25: Was true while `decode_multi_seq` was the trait default, which ran every
    /// sequence through `decode`'s highway slot 0 and the DSA mixer's metadata row 0.
    /// 2026-10-08: False for a text layer: `decode_multi_seq` (`forward_multi`) gives each row
    /// its own slot, state and metadata row. True for the MTP block (no hyper-connection),
    /// which `forward_multi` refuses; the model never puts that block in its layer list.
    fn decode_multi_seq_unsupported(&self) -> bool {
        self.mhc.is_none()
    }

    /// 2026-10-09: True. Every per-step input of the decode is read from device buffers the
    /// model uploads before a replay (the DSA position, slot, `seq_len` and block table from
    /// the metadata, the KDA state from the slot-keyed pool), and `reduce_partial` issues the
    /// collectives on the capturing stream; the batched decode already replays the same
    /// launches with a communicator.
    fn decode_graph_with_comm(&self) -> bool {
        true
    }

    /// 2026-10-10: Below the routed-MoE grouped GEMM's row floor (`grouped_row_floor`): from
    /// there the grouped path's host-side tile sizing is what a capture must not run.
    fn max_captured_rows(&self) -> Option<usize> {
        Some(crate::glm5next_mlp::forward::grouped_row_floor(&self.levers.mlp) - 1)
    }

    /// 2026-10-08: True: a DSA row selects over its own sequence's indexer cache
    /// (`Glm5NextDsaLayer::decode_rows`) and a KDA layer has no index, so the model's mHC +
    /// sparse-index per-sequence rule does not apply to this layer.
    fn decode_multi_seq_selects_index_per_row(&self) -> bool {
        true
    }

    /// 2026-09-25: Was true while this layer had no `decode_verify_multi`.
    /// 2026-10-09: False for a text layer (`steps/verify_multi.rs`); true for the MTP block,
    /// which has no hyper-connection and which `forward_spans` refuses.
    fn decode_verify_multi_unsupported(&self) -> bool {
        self.mhc.is_none()
    }

    /// 2026-10-09: True: the batched verify issues the same collectives on every rank (one
    /// all-reduce per mixer and MLP site per row group) and keeps no rank-local choice, so
    /// the worker ranks can run it from the batch rank 0 announces.
    fn batch_verify_across_ranks(&self) -> bool {
        true
    }

    /// 2026-10-09: `levers.prefill_rows` for a text layer (`steps/prefill_spans.rs`); `None` for the
    /// MTP block.
    fn prefill_spans_rows(&self) -> Option<usize> {
        self.mhc.is_some().then_some(self.levers.prefill_rows)
    }

    /// 2026-10-14: [`Glm5NextLayer::slot_graphs_stale`] (`types.rs`).
    fn graph_stale_on_new_sequence(&self) -> bool {
        self.slot_graphs_stale()
    }

    /// 2026-10-14: True: a borrowed verify graph's ghost rows have no DSA state of their own.
    fn verify_graph_borrow_unsafe(&self) -> bool {
        true
    }

    /// 2026-09-25: True for a KDA layer. Its recurrent and conv state then come from the SSM
    /// pool, and with MTP on, so do the checkpoint and per-token intermediate pointers that a
    /// speculative rollback restores (`meta.rs`).
    fn uses_ssm_pool(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }

    /// 2026-09-25: True for a KDA layer, whose mixer carries recurrent state.
    fn is_ssm_layer(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }

    /// 2026-10-11: The prefill splits its whole pass at an in-pass capture's points
    /// (`steps/prefill_capture.rs`), so the rows after a capture get a separate pass's bits.
    fn supports_replay_tail_split(&self) -> bool {
        true
    }

    /// 2026-10-11: Every launch of the prefill splits there, not only the recurrence, so the
    /// model needs neither row-count-invariant kernels nor a single rank for the capture.
    fn splits_pass_at_capture(&self) -> bool {
        true
    }

    /// 2026-10-08: True for a KDA layer: `decode_batched` checkpoints and records a
    /// replay-mode verify, and `ssm_replay_commit` rebuilds the accepted state.
    fn supports_ssm_replay(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }
}

impl LayerWeightSetup for Glm5NextLayer {}
impl LayerWriteOnAccept for Glm5NextLayer {
    /// 2026-10-08: A KDA layer's replay commit (`Glm5NextKdaLayer::commit_replay`); `Ok(false)`
    /// on a DSA layer, which keeps no recurrent state.
    fn ssm_replay_commit(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut dyn LayerState,
        accepted: usize,
        k_rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let Glm5NextMixer::Kda { layer, ws, .. } = &self.mixer else {
            return Ok(false);
        };
        let st = self.kda_state(state)?;
        let (live, checkpoint, record) = self.replay_parts(layer, st)?;
        layer.commit_replay(
            gpu,
            &live,
            &checkpoint,
            &record,
            accepted,
            k_rows,
            ws,
            stream,
        )?;
        Ok(true)
    }
}

impl LayerGraphHooks for Glm5NextLayer {
    /// 2026-09-25: A graph replay runs only kernels, so the host-side length of the DSA indexer
    /// cache is reconciled here (`Glm5NextDsaState::sync_to`). The inner `Glm5NextDsaLayer`
    /// implements this too, but the model's layer list holds this composite, so this impl is the
    /// one called. KDA keeps no host-side state.
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => self.dsa_state(state)?.sync_to(seq_len, k),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }

    /// 2026-09-25: Checks before a graph replay that the DSA indexer cache can hold `seq_len + k`
    /// rows. As with `sync_replayed_step`, this composite impl is the one the model calls.
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => state
                .as_any()
                .downcast_ref::<Glm5NextDsaState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM layer {}: a DSA mixer was handed state that is not a \
                         Glm5NextDsaState",
                        self.layer_idx
                    )
                })?
                .ensure_room_through(seq_len + k)
                // 2026-09-25: `ensure_room_through` raises the same error for every caller;
                // the context names this route.
                .with_context(|| {
                    format!(
                        "DSA replay pre-check (layer {}, before launch_graph, seq_len \
                         {seq_len} + k {k})",
                        self.layer_idx
                    )
                }),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }
}

impl LayerAuxState for Glm5NextLayer {
    /// 2026-09-25: True for a DSA layer: its indexer cache is the state `snapshot_aux` and
    /// `restore_aux` carry. A KDA layer's state is in the SSM pool (`uses_ssm_pool`) and is not
    /// carried here. 2026-10-09: False for a paged indexer cache, whose rows are in the KV
    /// blocks the prefix cache already shares.
    fn has_aux_state(&self) -> bool {
        matches!(&self.mixer, Glm5NextMixer::Dsa(l) if l.indexer_cache == IndexerCache::Flat)
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        if !self.has_aux_state() {
            return Ok(None);
        }
        let st = state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })?;
        Ok(Some(st.snapshot_blob(gpu, stream)?))
    }

    /// 2026-09-25: Errors on a KDA layer, whose state travels with the SSM snapshot. On a DSA
    /// layer it restores the blob through `Glm5NextDsaState::restore_blob`; `apply_aux_states`
    /// propagates any error.
    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !matches!(self.mixer, Glm5NextMixer::Dsa(_)) {
            bail!(
                "GLM layer {}: restore_aux on a KDA layer — KDA state is pool-backed and \
                 travels with the SSM snapshot, so a blob addressed here is a routing bug",
                self.layer_idx
            );
        }
        self.dsa_state(state)?.restore_blob(blob, gpu, stream)
    }
}

impl metrale_model_layers::circuit_exec::CircuitBindings for Glm5NextLayer {
    /// 2026-10-23: The instance policy with the MoE settings this layer's levers and layout pick.
    fn plan_policy(&self, policy: &metrale_circuit::Policy) -> Result<metrale_circuit::Policy> {
        crate::glm5next_plan::install::live_policy(policy, &self.mlp_kernels, &self.mlp_cfg)
    }

    /// 2026-10-23: A text layer's mHC sites and, on a routed layer, its MoE site; the MTP block
    /// executes nothing from a plan.
    fn install_plans(
        &self,
        gpu: &dyn GpuBackend,
        plans: Option<metrale_model_layers::circuit_exec::family::RankPlans<'_>>,
    ) -> Result<Vec<String>> {
        let Some(mhc) = &self.mhc else {
            return Ok(Vec::new());
        };
        let mut sites = crate::glm5next_plan::mhc::install(
            gpu,
            self.layer_idx,
            plans,
            &self.levers,
            mhc,
            self.hidden,
            &self.mhc_plans,
        )?;
        if let Glm5NextMlpSite::Moe(w) = &self.mlp {
            sites.extend(self.moe_plans.install(
                gpu,
                self.layer_idx,
                plans,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                &self.mlp_ws,
            )?);
        }
        Ok(sites)
    }

    fn plan_sites(&self) -> Vec<String> {
        let mut s = self.mhc_plans.sites();
        s.extend(self.moe_plans.sites());
        s
    }
}
