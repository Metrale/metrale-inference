// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `SchedLevers::defaults`, the lever values tests use without
//! reading the environment. Moved out of `levers.rs` unchanged to keep it
//! under the file-size cap; the environment resolution (`from_env`) and the
//! struct are in `levers.rs`.
//!
//! Owner: scheduler.
//! Invariants: the values match `SchedLevers::from_env` with no `METRALE_*`
//! set, except the two levers `defaults` names.

use super::*;

impl SchedLevers {
    /// 2026-09-25: Lever values for tests, without reading the environment.
    /// They match `from_env` with no `METRALE_*` set except
    /// `dflash_masked_verify` and `dflash_seam_serial`, which are off here
    /// and on in `from_env`.
    pub fn defaults() -> Self {
        Self {
            fast_greedy_grammar: true,
            fast_masked: true,
            fast_greedy_chat: true,
            force_temp_zero: false,
            tool_response_stop: true,
            tool_eos_escape: true,
            mtp_minp: true,
            mtp_verify_sample: true,
            dflash_eagle_fix: true,
            dflash_step_timing: false,
            vision_timing: false,
            dflash_masked_verify: false,
            dflash_seam_serial: false,
            dflash_adaptive: false,
            dflash_serial_append: false,
            dflash_unified_ctx: true,
            dflash_spec_think: false,
            mtp_spec_think_env: None,
            dflash_gate_pin_c2: true,
            dflash_batch_verify: true,
            dflash_adaptive_min: 2.0,
            dflash_adaptive_reprobe: 256,
            dflash_resume_guard: 0,
            shadow_topk: 0,
            disable_watchdogs: false,
            eos_suppressed_by_thinking: false,
            forced_token_fastpath: true,
            decode_timing: false,
            mtp_timing: false,
            mtp_gate_force: false,
            adadec_diagnostic: false,
            holo_always_mixed: false,
            prefill_codispatch: false,
            prefill_varlen: false,
            codispatch_window_ms: 100,
            codispatch_settle_ms: 10,
            vision_codispatch: false,
            beam_codispatch: true,
            bisect_q12_disable: false,
            bisect_no_mix: false,
            mixed_slice_tokens: 0,
            grammar_budget_close: true,
            think_ended_gpu_argmax: true,
            parallel_sample: true,
            mtp_batch_bootstrap: true,
            mtp_boot_argmax: true,
            mtp_batch_verify: true,
            mtp_batch_propose: true,
            dcut_enabled: true,
            dcut_width_cap: 8,
            dcut_ratio: 0.75,
            mtp_accept_fold_at_16: false,
            mtp_accept_debug: false,
            // 2026-09-25: `mtp_max_seqs()` with the variable unset.
            mtp_max_seqs: 32,
            spec_entry_pin_tokens: 8,
            ssm_tail_ckpt: false,
            ssm_tail_midchunk: true,
            loop_watchdog: AtomicBool::new(false),
        }
    }
}
