// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: How [`ModelLevers`] is read: the resolution table (`from_settings`) and the
//! `defaults` constructor.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - 2026-10-17: `from_settings` reads only the resolved settings and its arguments; it never
//!   touches the process environment. `TransformerModel::new` calls it once per model, and every
//!   other reader takes the model's copy from `ForwardContext::levers`.
//! - `from_settings(&defaults)` in the tests equals [`ModelLevers::defaults`]
//!   (`the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off`).

use super::ModelLevers;
use crate::layers::ops::{OpsLevers, gemv_sw};
use metrale_settings::InferenceServerSettings;

impl ModelLevers {
    /// 2026-10-17: The levers of one model, read from the settings resolved once at startup.
    /// `default_split_silu` is the compiled target's `[defaults] decode_split_silu`
    /// (`target_defaults::declared()`), passed in so a test can choose it. `weight_pre_rotated` is
    /// `TQ_PLUS_WEIGHT_ROTATION`, a foreign name with no settings row (`tq_plus_weight_rotation`).
    /// `max_decode_seqs` is 1; `TransformerModel::new` overwrites it.
    pub fn from_settings(
        s: &InferenceServerSettings,
        default_split_silu: bool,
        weight_pre_rotated: bool,
    ) -> Self {
        let (k, sch, sp, t) = (&s.kernels, &s.scheduler, &s.speculative, &s.telemetry);
        ModelLevers {
            max_decode_seqs: 1,
            shadow_topk: crate::speculative::shadow_topk_given(sp.mtp_shadow_topk),
            kv_poison: s.cache.kv_poison,
            drafter: crate::drafter_context::from_settings(sp),
            ops: OpsLevers::from_settings(s),
            gdn_regresident: !k.no_gdn_regresident,
            gdn_batched_fla: sch.gdn_batched_fla,
            gdn_wy17: k.gdn_wy17,
            gdn_wyn: k.gdn_wyn,
            ffn_small_m: k.ffn_smallm,
            gemv_sw: gemv_sw::gemv_sw_from(k.no_gemv_sw.as_deref()),
            decode_ffn_via_gemm: k.decode_ffn_via_gemm,
            holo_moe_down_fp4: k.holo_moe_down_fp4,
            holo_moe_gateup_fp4: k.holo_moe_gateup_fp4,
            moe_union_stats: t.moe_union_stats,
            fp32_routing: k.fp32_routing,
            fp32_gate: k.fp32_gate,
            frankenstein_decode_via_prefill: sch.frankenstein_decode_via_prefill,
            k2_diag: t.k2_diag,
            dflash_debug_dump_full: t.dflash_debug_dump_full,
            mtp_debug_norms: t.mtp_debug_norms,
            mtp_chain_postnorm: sp.mtp_chain_postnorm,
            mtp_target_postnorm: sp.mtp_target_postnorm,
            mtp_kv_exact: sp.mtp_kv_exact,
            mtp_accepted_rows: sp.mtp_accepted_rows,
            mtp_adopt_capture: sp.mtp_adopt_capture,
            mtp_zero_pos0: sp.mtp_zero_pos0,
            one_row_prefill: sch.one_row_prefill,
            one_row_tail_decode: sch.one_row_tail_decode,
            seq_alloc_async: sch.seq_alloc_async,
            glm_keep_slot_graphs: sch.glm_keep_slot_graphs,
            glm_mtp_state_pool: sp.glm_mtp_state_pool,
            moe_fp8_grouped_decode_target: k.fp8_moe_grouped_decode,
            fp8_attn_m32: k.fp8_attn_m32,
            // 2026-09-25: The target's declaration (`kernels/<hw>/HARDWARE.toml` `[defaults]
            // decode_split_silu`; undeclared is on, `crates/closure/src/hardware_defaults.rs`)
            // arrives as `default_split_silu`. The presence of `METRALE_NO_DECODE_SPLIT_SILU`, any
            // value, turns it off (`resolve_toggle`'s `legacy_off`).
            decode_split_silu: crate::layers::ops::target_defaults::resolve_toggle(
                default_split_silu,
                None,
                k.no_decode_split_silu,
            )
            .value,
            bf16_tc_prefill: sch.bf16_tc_prefill,
            fp8_m64_prefill: sch.fp8_m64_prefill,
            int8_prefill: sch.int8_prefill,
            int8_faith5: k.int8_faith5,
            ffn_nvfp4_mmq: !k.no_ffn_nvfp4_mmq,
            ffn_nvfp4_mmq_down: !k.no_ffn_nvfp4_mmq_down,
            ffn_mmq: k.ffn_mmq,
            ffn_mmq_down_q4k: k.ffn_mmq_down_q4k,
            fp4_prefill: sch.fp4_prefill,
            prefill_v2: !sch.disable_prefill_v2,
            moe_grouped_cutlass: k.holo_moe_grouped_cutlass.as_deref() == Some("1"),
            moe_grouped_down: k.holo_moe_grouped_down,
            moe_prefill_exact_tiles: match sch.moe_prefill_exact_tiles.as_deref() {
                Some("0") => Some(false),
                Some("1") => Some(true),
                _ => None,
            },
            moe_prefill_max_load_factor: sch
                .moe_prefill_max_load_factor
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&factor| factor > 0),
            moe_prefill_zero: sch.moe_prefill_zero,
            moe_prefill_fp8_down: sch.moe_prefill_fp8_down,
            ssm_w4a4: !k.no_ssm_w4a4,
            ssd: !k.no_ssd,
            ssm_persistent: !k.no_ssm_persistent,
            moe_zero_intermediates: !k.moe_no_zero_intermediates,
            moe_max_m_tiles_estimate: k.moe_max_m_tiles_estimate,
            moe_w4a4: k.moe_w4a4,
            shared_w4a4: !k.no_shared_w4a4,
            shared_w4a4_down: k.shared_w4a4_down,
            dflash_contig_attn: sp.dflash_contig_attn,
            lora_eager: s.model.lora_eager,
            lora_rotate: s.model.lora_rotate,
            k4_diag: t.k4_diag,
            gemma4_diag: t.diag_gemma4,
            mla_perseq_fallback: k.mla_perseq_fallback,
            hc_perseq_decode: k.hc_perseq_decode,
            decode_batch_log: t.decode_batch_log,
            ms_profile: t.ms_profile,
            conc_hsd: s.runtime.conc_hsd,
            ssm_save_dump: t.ssm_save_dump,
            ep_graphs: k.ep_graphs,
            gdn_decode_graph: k.gdn_decode_graph,
            bf16_tc_proj: k.bf16_tc_proj,
            weight_pre_rotated,
            ssm_ms_profile: t.ssm_ms_profile,
            ssm_detail_profile: t.ssm_detail_profile,
            ssm_gemv_batch4: sch.ssm_gemv_batch4,
            gdn_fused_conv: k.gdn_fused_conv,
            moe_legacy_pertoken_decode: k.moe_legacy_pertoken_decode,
        }
    }

    /// 2026-10-14: The boot readback of the new-sequence levers, as `TransformerModel::new` logs
    /// it on every rank: `seq_alloc_async`, `glm_keep_slot_graphs` and (2026-10-14)
    /// `glm_mtp_state_pool`, each `on` or `off`.
    pub fn new_sequence_readback(&self) -> String {
        let on = |b: bool| if b { "on" } else { "off" };
        format!(
            "METRALE_SEQ_ALLOC_ASYNC={} METRALE_GLM_KEEP_SLOT_GRAPHS={} METRALE_GLM_MTP_STATE_POOL={}",
            on(self.seq_alloc_async),
            on(self.glm_keep_slot_graphs),
            on(self.glm_mtp_state_pool)
        )
    }

    /// 2026-10-17: `TQ_PLUS_WEIGHT_ROTATION` (`1` or `true`, any case): the checkpoint's weights
    /// are pre-rotated. A foreign name, so no `metrale-settings` row can hold it; this is its one
    /// read, which `TransformerModel::new` passes to [`Self::from_settings`].
    pub fn tq_plus_weight_rotation() -> bool {
        std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    }

    /// 2026-09-25: What `from_settings` returns at the declared defaults, a target that declares
    /// `decode_split_silu` on, `DrafterContext::BOTH` and no shadow
    /// top-k: every opt-in off, every opt-out on. Tests build a context with this instead of
    /// mutating the environment.
    pub fn defaults() -> Self {
        Self {
            max_decode_seqs: 1,
            shadow_topk: 0,
            kv_poison: false,
            drafter: crate::drafter_context::DrafterContext::BOTH,
            ops: OpsLevers::declared(),
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            ffn_small_m: true,
            gemv_sw: true,
            // 2026-09-25: On unless `METRALE_SSM_GEMV_BATCH4=0`. Every opt-out lever must be listed
            // here or `the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off` fails.
            ssm_gemv_batch4: true,
            // 2026-09-25: The dense-FFN opt-outs: on, and turned off by the presence of their
            // variable at any value, `=0` included.
            decode_split_silu: true,
            ffn_nvfp4_mmq: true,
            ffn_nvfp4_mmq_down: true,
            prefill_v2: true,
            // 2026-09-25: The Nemotron prefill opt-outs, presence-gated the same way.
            ssm_w4a4: true,
            ssd: true,
            ssm_persistent: true,
            moe_zero_intermediates: true,
            shared_w4a4: true,
            ..Self::default()
        }
    }
}
