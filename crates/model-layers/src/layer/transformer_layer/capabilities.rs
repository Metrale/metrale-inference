// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerCapabilities`, the side-effect-free questions the model asks a layer when it
//! picks a route: graph eligibility, batched decode and verify support, rollback, SSM
//! pool use and the MLA prefill mode.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerCapabilities {
    /// 2026-09-25: True when this layer's prefill attends only over the tokens it is
    /// handed, so skipping a cached prefix would hide it from attention. The Qwen3
    /// attention layer returns true when it runs MLA; the prefix lookup then matches
    /// nothing unless `METRALE_MLA_PREFIX_SKIP=1` (`mla_prefill_needs_full_recompute`).
    fn uses_local_mla_prefill(&self) -> bool {
        false
    }

    /// 2026-09-25: Whether this layer's online FP8-KV calibration has frozen its scale;
    /// `None` when the layer runs no online calibration (the default). The model lifts its
    /// CUDA-graph suppression once no layer reports `Some(false)`
    /// (`graphs_ready_after_fp8_kv_cal`).
    fn fp8_calibration_frozen(&self) -> Option<bool> {
        None
    }

    /// 2026-09-25: True when this layer's decode cannot run inside a CUDA graph, such as
    /// the QSA indexer's host top-k. The model ORs it across layers into
    /// `decode_graph_veto` when it is built.
    fn decode_graph_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: True when this layer cannot serve a batched multi-sequence decode step.
    /// The batched decode ORs it across layers into `hc_perseq` (`decode_a2.rs`) and then
    /// runs each sequence through `decode`; the single-GPU fused decode+prefill ORs it into
    /// `hc_qsa_perseq` (`decode_b.rs`) and then runs the batched decode and the prefill
    /// separately.
    fn decode_multi_seq_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: True when this layer keeps per-sequence state that lowering the
    /// sequence's KV cursor does not rewind, such as a monotonic cache count or an n-gram
    /// history. The model ORs it across layers, and the scheduler's `rollback_to_boundary`
    /// then declines the rollback (`LayerStateNotRewindable`).
    fn decode_rollback_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: True when this layer cannot serve a batched multi-sequence verify
    /// (`decode_verify_multi`); `can_batch_verify_dispatch` then refuses the batch. It is
    /// separate from [`Self::decode_multi_seq_unsupported`] because the answers can
    /// differ.
    fn decode_verify_multi_unsupported(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    /// 2026-09-25: True when a captured decode graph goes stale once a new sequence takes
    /// this slot. Decode graphs are keyed by `slot_idx`, which is safe only while every
    /// per-sequence address a capture bakes lives in the slot-addressed SSM pool. A layer
    /// that allocates its own per-sequence buffers (GLM-5.3's DSA indexer cache) returns
    /// true, and `free_sequence_dispatch` then drops that slot's graphs, except for a slot
    /// index past the SSM pool's `max_slots`.
    fn graph_stale_on_new_sequence(&self) -> bool {
        false
    }

    /// 2026-09-25: True for an SSM layer. The model's split prefill then runs
    /// `prefill_phase1`, `prefill_gdn_full` and `prefill_phase3` instead of `prefill`.
    fn is_ssm_layer(&self) -> bool {
        false
    }

    /// 2026-09-25: Whether this layer's recurrent state lives in the shared SSM pool. When
    /// true (the default), sequence setup hands a linear-attention layer an `SsmLayerState`
    /// with pool addresses and never calls its `alloc_state`. A linear-attention layer with
    /// its own state type returns false; the Kimi K3 layer (`kimi_k3/bound.rs`) does, and
    /// downcasts its state to `K3CpuFallbackState`.
    fn uses_ssm_pool(&self) -> bool {
        true
    }

    /// 2026-10-01: True when this layer's `prefill` honours a replay-tail capture
    /// (`ForwardContext::midchunk_capture` with `replay_tail`) on every build: the rows
    /// from `cap_local` on get the bits a prefill pass starting at `cap_local` gives them
    /// (an SSM layer splits its conv and recurrence there and copies its state into the
    /// plan's slots; an attention layer runs those rows as their own paged call). The
    /// model plans a replay-tail capture only when every layer says so
    /// (`TransformerModel::inpass_cut_capture_supported`); otherwise it keeps the
    /// two-pass tail split.
    fn supports_replay_tail_split(&self) -> bool {
        false
    }

    /// 2026-10-08: True when this layer, a recurrent layer on the SSM pool, can roll a
    /// speculative verify back under `--ssm-rollback-mode replay`: its K-row verify copies
    /// the state to the slot's checkpoint and records each row's recurrent inputs in the
    /// slot's `SsmLayerState::replay_ring`, and `LayerWriteOnAccept::ssm_replay_commit`
    /// restores and replays. The model admits a replay-mode verify only when every
    /// pool-backed recurrent layer answers true.
    fn supports_ssm_replay(&self) -> bool {
        false
    }
}
