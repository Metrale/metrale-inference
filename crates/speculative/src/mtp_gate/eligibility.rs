// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Whether a sequence may take the speculative path this step,
//! and the per-model and per-lane spec-in-think levers that decision reads.
//! Moved out of `mtp_gate.rs` unchanged to keep it under the file-size cap;
//! `mtp_gate` re-exports the three functions, so their paths are unchanged.
//! The gate state machine is in `mtp_gate.rs`.
//!
//! Owner: speculative.
//! Invariants: none beyond the types; each function is a pure function of its
//! arguments.

/// 2026-09-29: A146: the per-model default for MTP-lane speculation inside
/// `<think>`, keyed on `ModelConfig::model_type` like the other
/// glm5_next-specific serve behaviour (`seq_state_reserve.rs`). On only for
/// architectures that passed the spec-in-think quality and safety gates:
/// GLM-5.3 (`glm5_next` and its text-only `glm5_next_text`); measured on
/// Atlas 2026-09-26: K=3 byte-identical 6/6 vs spec-off, TEB 156/176
/// identical to spec-off per scenario. Every other model keeps the pre-split
/// behaviour (serial inside thinking unless opted in) until it passes the
/// same gates.
pub fn mtp_spec_think_default(model_type: &str) -> bool {
    matches!(model_type, "glm5_next" | "glm5_next_text")
}

/// 2026-09-29: A146: the per-lane spec-in-think lever
/// [`spec_dispatch_eligible`] must receive. `dflash_lane` is
/// `dflash_verify_raw_argmax` (`args.dflash`, serve_load.rs), true for EVERY
/// DFlash verify mode. The MTP lane uses `SchedLevers::mtp_spec_think(model
/// default)`; the DFlash lane stays opt-in (`SchedLevers::dflash_spec_think`)
/// on every model.
pub fn spec_think_for_lane(
    dflash_lane: bool,
    mtp_spec_think: bool,
    dflash_spec_think: bool,
) -> bool {
    if dflash_lane {
        dflash_spec_think
    } else {
        mtp_spec_think
    }
}

/// 2026-09-25: Whether one sequence may take the speculative path this step.
/// Never with `suppress_tool_call` or `disable_mtp`, and never inside
/// `<think>` unless `spec_think`. 2026-09-29: A146: `spec_think` is the
/// ACTIVE lane's lever, resolved by [`spec_think_for_lane`]: MTP speculates
/// inside `<think>` by default on models whose [`mtp_spec_think_default`] is
/// on (`METRALE_MTP_SPEC_THINK=0` or `METRALE_DFLASH_SPEC_THINK=0`
/// disables) and on others only when opted in (`=1`); DFlash stays serial
/// inside `<think>` unless `METRALE_DFLASH_SPEC_THINK=1`. Otherwise it needs `resume_guard` emitted tokens: counted
/// after `</think>`, except inside `<think>` (reachable only with
/// `spec_think`), where the whole output counts.
pub fn spec_dispatch_eligible(
    inside_thinking: bool,
    post_think_emitted: u32,
    output_len: u32,
    suppress_tool_call: bool,
    disable_mtp: bool,
    spec_think: bool,
    resume_guard: u32,
    dflash_raw_argmax: bool,
) -> bool {
    if suppress_tool_call || disable_mtp {
        return false;
    }
    // 2026-09-25: Batch-K verify can pick a different low-margin token than
    // serial decode at T=0 (see `parse_entry_pin_tokens`).
    //
    // 2026-09-29: A146: this was once opt-in for BOTH lanes because batch-K
    // verify committed low-margin tokens spec-off decode would not. The
    // spec-in-think parity chain (verify window and `emit_token` commit
    // thinking state exactly like spec-off decode, plus A143/A144/A144b)
    // closed that for MTP on GLM-5.3, so MTP is default-on there only
    // (`mtp_spec_think_default`); other models and DFlash stay opt-in until
    // they pass their own GPU and TEB qualification.
    if inside_thinking && !spec_think {
        return false;
    }
    if dflash_raw_argmax && !spec_think {
        return post_think_emitted >= resume_guard;
    }
    if inside_thinking {
        output_len >= resume_guard
    } else {
        post_think_emitted >= resume_guard
    }
}
