// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25 (moved 2026-10-10 from `mtp_gate`, whose decision moved onto `spec_ctl`): which
//! sequences may take a speculative step at all, and the spec-entry pin that forces the verify
//! step while a sequence is opening its answer. These rules decide eligibility, never depth.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

/// 2026-09-25: The spec-entry verify pin, in tokens after `</think>`, from
/// the value of `METRALE_SPEC_ENTRY_PIN`: 8 when unset or unparseable, and
/// `0` disables it. While any active sequence is within it, the scheduler
/// runs the MTP verify step even when the gate chose plain decode.
///
/// The serial (one-row) and verify (batch-K) forwards can pick different
/// low-margin tokens at T=0, and the controller picks between them by measured
/// cost, so without the pin the first tokens of an answer would depend
/// on the speculation controller's measurements. Measured 2026-07-07/08: every observed flip
/// between the two forwards fell within 7 tokens of speculation entry; 8
/// adds one token of margin.
///
/// `METRALE_DFLASH_RESUME_GUARD` is checked earlier, in
/// `spec_dispatch_eligible`: outside `<think>`, a sequence with fewer tokens
/// after `</think>` than the guard does not reach the gate at all.
pub(crate) fn parse_entry_pin_tokens(env: Option<&str>) -> u32 {
    env.and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// 2026-09-25: The pin width from `METRALE_SPEC_ENTRY_PIN`. The scheduler
/// stores it in `SchedLevers::spec_entry_pin_tokens`.
pub fn entry_pin_tokens_from_env() -> u32 {
    parse_entry_pin_tokens(std::env::var("METRALE_SPEC_ENTRY_PIN").ok().as_deref())
}

/// 2026-09-25: Whether the spec-entry pin overrides a Serial gate decision
/// for this step. `min_post_think_emitted` is the minimum over the active
/// batch, so one entering sequence pins the whole (already spec-eligible)
/// batch. `pin_tokens` is the run's [`entry_pin_tokens_from_env`] value.
pub fn entry_pin_forces_verify(min_post_think_emitted: u32, pin_tokens: u32) -> bool {
    min_post_think_emitted < pin_tokens
}

/// 2026-09-25: Whether one sequence may take the speculative path this step.
/// Never with `suppress_tool_call` or `disable_mtp`, and never inside
/// `<think>` unless `spec_think` (`METRALE_DFLASH_SPEC_THINK=1`), for MTP and
/// DFlash alike. Otherwise it needs `resume_guard` emitted tokens: counted
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Within the pin, an answer opening is forced onto the verify step; at the
    /// pin and beyond it is not.
    #[test]
    fn entry_pin_overrides_serial_mode_for_answer_openings() {
        assert!(entry_pin_forces_verify(0, 8));
        assert!(entry_pin_forces_verify(7, 8));
        assert!(!entry_pin_forces_verify(8, 8));
        assert!(!entry_pin_forces_verify(u32::MAX, 8));
    }

    #[test]
    fn entry_pin_env_parse() {
        assert_eq!(parse_entry_pin_tokens(None), 8);
        assert_eq!(parse_entry_pin_tokens(Some("0")), 0);
        assert_eq!(parse_entry_pin_tokens(Some("12")), 12);
        assert_eq!(parse_entry_pin_tokens(Some("garbage")), 8);
        assert_eq!(parse_entry_pin_tokens(Some("-3")), 8);
    }

    #[test]
    fn standard_mtp_stays_serial_in_think() {
        assert!(!spec_dispatch_eligible(
            true, 0, 0, false, false, false, 0, false
        ));
        assert!(!spec_dispatch_eligible(
            true, 0, 50, false, false, false, 0, false
        ));
        assert!(spec_dispatch_eligible(
            false, 0, 50, false, false, false, 0, false
        ));
    }

    #[test]
    fn standard_mtp_spec_think_opts_in() {
        assert!(spec_dispatch_eligible(
            true, 0, 50, false, false, true, 0, false
        ));
    }

    #[test]
    fn dflash_raw_argmax_stays_serial_in_think() {
        assert!(!spec_dispatch_eligible(
            true, 0, 50, false, false, false, 0, true
        ));
        assert!(spec_dispatch_eligible(
            false, 0, 50, false, false, false, 0, true
        ));
    }

    #[test]
    fn dflash_spec_think_opts_in() {
        assert!(spec_dispatch_eligible(
            true, 0, 0, false, false, true, 0, true
        ));
    }
}
