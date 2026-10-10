// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that a layer declining the batched multi-sequence path is routed around it.
//!
//! The default `decode_multi_seq` loop passes one `ForwardContext` to every sequence in the
//! batch, so a layer that indexes per-sequence state by a fixed row would compute with another
//! sequence's state. A layer returning true from `decode_multi_seq_unsupported()` is routed per
//! sequence by `decode_a2`'s `hc_perseq` and `decode_b`'s `hc_qsa_perseq`; one returning true
//! from `decode_verify_multi_unsupported()` makes `can_batch_verify_dispatch` refuse the
//! batched verify.
//!
//! The veto is the first disjunct, outside the `hc_mult > 0` conjunction. Inside it, the veto
//! would also need `seq_len >= index_topk + index_compress_ratio - 1` and would miss every
//! shorter sequence. The tests pin that placement, not only the presence of the term.
//!
//! Owner: model-engine decode.
//! Invariants: none beyond the types.

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use metrale_cache::kv_cache::PagedKvCache;
    use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
    use metrale_model_layers::layer::{ForwardContext, LayerState, TransformerLayer};
    use metrale_model_layers::layer::{
        LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
        LayerWriteOnAccept,
    };

    /// 2026-09-25: The two required `TransformerLayer` methods, stubbed. These tests only
    /// call the capability predicates, never a forward.
    macro_rules! stub_forward {
        () => {
            #[allow(clippy::too_many_arguments)]
            fn decode(
                &self,
                _hidden: DevicePtr,
                _residual: DevicePtr,
                _state: &mut dyn LayerState,
                _kv_cache: &mut PagedKvCache,
                _seq_len: usize,
                _block_table: &mut Vec<u32>,
                _disk_block_ids: &mut Vec<u32>,
                _disk_last_offloaded_per_layer: &mut Vec<u32>,
                _ctx: &ForwardContext,
                _stream: u64,
            ) -> Result<()> {
                unreachable!("capability-predicate test never runs a forward")
            }
            fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
                unreachable!("capability-predicate test never allocates state")
            }
        };
    }

    fn src(rel: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"))
    }

    /// 2026-09-25: Slice `s` from the first `from` to the next `to` after it, or to the end.
    fn block<'a>(s: &'a str, from: &str, to: &str) -> &'a str {
        let start = s
            .find(from)
            .unwrap_or_else(|| panic!("missing anchor {from:?}"));
        let rest = &s[start..];
        let end = rest.find(to).unwrap_or(rest.len());
        &rest[..end]
    }

    /// 2026-09-25: Both trait defaults are false, so a layer that does not override them keeps
    /// the batched route.
    #[test]
    fn defaults_are_false_so_no_existing_layer_changes_route() {
        struct Plain;
        impl TransformerLayer for Plain {
            stub_forward!();
        }
        impl LayerCapabilities for Plain {}
        impl LayerWeightSetup for Plain {}
        impl LayerWriteOnAccept for Plain {}
        impl LayerGraphHooks for Plain {}
        impl LayerAuxState for Plain {}
        impl LayerSplitPrefill for Plain {}
        impl metrale_model_layers::circuit_exec::CircuitBindings for Plain {}
        assert!(
            !Plain.decode_multi_seq_unsupported(),
            "default must be false — a new predicate may not re-route existing models"
        );
        assert!(
            !Plain.decode_verify_multi_unsupported(),
            "default must be false"
        );
        assert!(
            !Plain.decode_graph_with_comm(),
            "default must be false — a layer that does not opt in keeps its decode eager under TP/EP"
        );
        assert!(
            !Plain.decode_multi_seq_selects_index_per_row(),
            "default must be false — a layer that does not opt in keeps the QSA per-sequence rule"
        );
    }

    /// 2026-10-08: The QSA per-sequence term is lifted only by the all-layers vote, at both
    /// dispatch sites, and the vote is `all` over a non-empty stack. Dropping the term at a
    /// site, or voting with `any`, fails this test.
    #[test]
    fn the_qsa_term_is_lifted_only_when_every_layer_selects_per_row() {
        let a2 = src("src/model/trait_impl/decode_a2.rs");
        let b = block(&a2, "let qsa_active", "let ms_layer_veto");
        assert!(
            b.contains("self.config.index_topk > 0 && !self.layers_select_index_per_row()"),
            "decode_a2's qsa_active must consult the vote"
        );
        let fused = src("src/model/trait_impl/decode_b.rs");
        let b = block(&fused, "let hc_qsa_perseq", "if self.comm.is_some()");
        assert!(
            b.contains("&& self.config.index_topk > 0\n                && !self.layers_select_index_per_row()"),
            "decode_b's QSA conjunct must consult the vote"
        );
        let hooks = src("src/model/trait_impl/decode_a2/batch_hooks.rs");
        let vote = block(&hooks, "fn layers_select_index_per_row", "\n    }\n");
        assert!(vote.contains("!self.layers.is_empty()"));
        assert!(vote.contains(".all(|l| l.decode_multi_seq_selects_index_per_row())"));
    }

    /// 2026-10-09: The single-sequence decode captures with a communicator only under a lever
    /// or when every layer opts in, and the opt-in can be turned off.
    #[test]
    fn single_sequence_decode_captures_with_comm_only_when_every_layer_opts_in() {
        let a = src("src/model/trait_impl/decode_a.rs");
        assert!(a.contains(
            "let use_graphs = (self.comm.is_none() || ep_graphs || gdn_graphs || layer_comm_graphs)"
        ));
        let hooks = src("src/model/trait_impl/decode_a2/batch_hooks.rs");
        let vote = block(&hooks, "fn layers_capture_with_comm", "\n    }\n");
        assert!(vote.contains(".all(|l| l.decode_graph_with_comm())"));
        assert!(vote.contains("!self.layers.is_empty()"));
        assert!(vote.contains("\"METRALE_COMM_DECODE_GRAPHS\").as_deref() == Ok(\"0\")"));
    }

    /// 2026-10-08: A replayed batched-decode graph checks every row's room before the launch
    /// and reconciles every row's host bookkeeping after it, as `decode_a.rs` does for one
    /// sequence; a replay writes GLM-5.3's DSA indexer rows with no host code in the loop.
    #[test]
    fn a_batched_replay_checks_room_before_and_syncs_after_the_launch() {
        let a2 = src("src/model/trait_impl/decode_a2.rs");
        let room = a2
            .find("self.batch_replay_check_room(seqs)?")
            .expect("room check");
        let launch = a2.find("self.gpu.launch_graph(").expect("replay launch");
        let sync = a2.find("self.batch_replay_sync(seqs)?").expect("reconcile");
        assert!(
            room < launch && launch < sync,
            "room, launch, sync — in that order"
        );
        let hooks = src("src/model/trait_impl/decode_a2/batch_hooks.rs");
        assert!(hooks.contains("layer.check_replay_room(&*seq.layer_states[i], seq.seq_len, 1)?"));
        assert!(
            hooks.contains("layer.sync_replayed_step(seq.layer_states[i].as_mut(), seq_len, 1)?")
        );
    }

    /// 2026-10-08: Both padding-row builders take the layer's padding state, not `alloc_state`,
    /// which for GLM-5.3's DSA layer allocates a context-sized cache that nothing frees.
    #[test]
    fn padding_rows_take_the_layers_padding_state() {
        for rel in [
            "src/model/trait_impl/decode_a2/pad_states.rs",
            "src/model/trait_impl/decode_b/build_states.rs",
        ] {
            let s = src(rel);
            assert!(
                s.contains("layer.alloc_pad_state(self.gpu.as_ref())?"),
                "{rel}"
            );
            assert!(!s.contains("layer.alloc_state("), "{rel}");
        }
    }

    #[test]
    fn a_declining_layer_is_honoured_on_both_axes_independently() {
        struct DeclinesDecode;
        impl TransformerLayer for DeclinesDecode {
            stub_forward!();
        }
        impl LayerCapabilities for DeclinesDecode {
            fn decode_multi_seq_unsupported(&self) -> bool {
                true
            }
        }
        impl LayerWeightSetup for DeclinesDecode {}
        impl LayerWriteOnAccept for DeclinesDecode {}
        impl LayerGraphHooks for DeclinesDecode {}
        impl LayerAuxState for DeclinesDecode {}
        impl LayerSplitPrefill for DeclinesDecode {}
        impl metrale_model_layers::circuit_exec::CircuitBindings for DeclinesDecode {}
        struct DeclinesVerify;
        impl TransformerLayer for DeclinesVerify {
            stub_forward!();
        }
        impl LayerCapabilities for DeclinesVerify {
            fn decode_verify_multi_unsupported(&self) -> bool {
                true
            }
        }
        impl LayerWeightSetup for DeclinesVerify {}
        impl LayerWriteOnAccept for DeclinesVerify {}
        impl LayerGraphHooks for DeclinesVerify {}
        impl LayerAuxState for DeclinesVerify {}
        impl LayerSplitPrefill for DeclinesVerify {}
        impl metrale_model_layers::circuit_exec::CircuitBindings for DeclinesVerify {}
        assert!(DeclinesDecode.decode_multi_seq_unsupported());
        assert!(
            !DeclinesDecode.decode_verify_multi_unsupported(),
            "decode and verify answers must be independent"
        );
        assert!(DeclinesVerify.decode_verify_multi_unsupported());
        assert!(!DeclinesVerify.decode_multi_seq_unsupported());
    }

    /// 2026-09-25: `decode_a2`, the batched decode dispatcher with and without EP. Deleting the
    /// veto term, or moving it inside the `hc_mult > 0` conjunction, fails this test.
    #[test]
    fn decode_a2_routes_a_declining_layer_per_sequence_at_every_length() {
        let s = src("src/model/trait_impl/decode_a2.rs");
        let b = block(&s, "let ms_layer_veto", "if self.comm.is_some()");
        assert!(
            b.contains("decode_multi_seq_unsupported()"),
            "decode_a2 must consult the layer predicate"
        );
        // 2026-09-25: The veto is the first disjunct of hc_perseq, so it is outside `hc_mult > 0`.
        assert!(
            b.contains("let hc_perseq = ms_layer_veto\n            || ("),
            "the veto must be hoisted OUT of the hc_mult/qsa_active conjunction; \
             folded inside, it would only fire at seq_len >= index_topk - 1"
        );
    }

    /// 2026-09-25: `decode_b`, the fused decode + prefill path, which runs only without a comm.
    /// A veto checked only in `decode_a2` would leave this path open.
    #[test]
    fn decode_b_routes_a_declining_layer_per_sequence_at_every_length() {
        let s = src("src/model/trait_impl/decode_b.rs");
        let b = block(&s, "let ms_layer_veto", "if self.comm.is_some()");
        assert!(
            b.contains("decode_multi_seq_unsupported()"),
            "decode_b must consult the layer predicate — it is the single-GPU path"
        );
        assert!(
            b.contains("let hc_qsa_perseq = ms_layer_veto\n            || ("),
            "the veto must be hoisted OUT of the hc_mult/index_topk conjunction"
        );
    }

    /// 2026-09-25: `can_batch_verify_dispatch` refuses the batched verify sweep for a declining
    /// layer as a routing decision, not a mid-request `bail!`.
    #[test]
    fn can_batch_verify_dispatch_consults_the_verify_predicate() {
        let s = src("src/model/trait_impl/verify_e.rs");
        let b = block(&s, "fn can_batch_verify_dispatch", "\n    pub(super) fn ");
        assert!(
            b.contains("decode_verify_multi_unsupported()"),
            "can_batch_verify_dispatch must consult the layer predicate"
        );
        assert!(
            b.contains("&& !self"),
            "the term must be a NEGATED conjunct of the existing self-gate"
        );
    }

    /// 2026-09-30: Under `--forward circuit` no sequence of a multi-sequence MTP step verifies or
    /// drafts in legacy code: the batched verify is admitted only when the executor compiles
    /// batched verifies and runs the program compiled for the batch's row table, and the batched
    /// propose runs the head's n-row draft program or declines, so each sequence drafts through
    /// the circuit's single-row program.
    #[test]
    fn a_circuit_forward_takes_no_batched_verify_or_propose() {
        let s = src("src/model/trait_impl/verify_e.rs");
        let gate = block(&s, "fn can_batch_verify_dispatch", "\n    pub(super) fn ");
        assert!(
            gate.contains(".is_none_or(|e| {\n                    e.verify_batch"),
            "can_batch_verify_dispatch must admit a circuit forward only with batched verifies"
        );
        let call = block(
            &s,
            "fn decode_verify_batched_dispatch",
            "let ctx = ForwardContext",
        );
        assert!(
            call.contains("if let Some((program, gdn)) = circuit_program {"),
            "decode_verify_batched_dispatch must run the circuit's program under a circuit forward"
        );
        let p = src("../model-layers/src/layers/mtp_head/draft_proposer.rs");
        let propose = block(&p, "fn propose_batch(", "let mut mtp_states");
        assert!(
            propose.contains("&& !(r.serves(last_tokens.len() as u64) && out_conf.is_some())"),
            "propose_batch must decline a width the circuit has no draft program for"
        );
        let q = src("../model-layers/src/layers/mtp_head/forward_batch/position/propose.rs");
        assert!(
            q.contains("Some(runner) => self.forward_batch_position_circuit("),
            "the batched propose must run the circuit's draft program when one is installed"
        );
    }

    /// 2026-09-25: The veto is consumed at dispatch, not as a serve-time `max_batch_size` clamp
    /// in `serve_load.rs`, which would lower concurrency for every model.
    #[test]
    fn stage0_did_not_reintroduce_a_serve_time_clamp() {
        // 2026-09-26: `load_model`'s steps live in `serve_load.rs` and its child modules;
        // `serve_load/scheduler_setup.rs` sizes `max_batch_size`.
        for rel in [
            "serve_load.rs",
            "serve_load/adapters.rs",
            "serve_load/carried.rs",
            "serve_load/load_phases.rs",
            "serve_load/model_setup.rs",
            "serve_load/scheduler_setup.rs",
        ] {
            let s = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../server/src/main_modules")
                    .join(rel),
            )
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
            assert!(
                !s.contains("decode_multi_seq_unsupported"),
                "the concurrency capability must be consumed at the DISPATCH site, \
                 never as a serve-time max_batch_size clamp ({rel})"
            );
        }
    }
}
