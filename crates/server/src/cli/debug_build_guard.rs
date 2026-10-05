// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Refuses a gate-recording run started from a debug build.
//!
//! `met benchmark run --pull-request-gate`, `certify` and `calibrate` each
//! write, or drive something that writes, a gate record whose number is a
//! wall time, a rate or a ratio built from one. A debug build is unoptimized
//! by an amount that varies by code path — measured live on dgx1
//! (2026-10-04): a 32k-token cold prefill that certifies at ~17s on a
//! release build took over 600s on a debug one, ~35x, on the same code and
//! box. That number is not a measurement of anything, never mind one
//! comparable to a release-built baseline, so there is no override: a gate
//! number from a debug build is never valid, under any flag.
//!
//! Owner: server CLI.
//! Invariants:
//! - Pure: takes the debug-ness as an argument rather than reading
//!   `cfg!(debug_assertions)` itself, so the refusal path is tested without a
//!   debug build of this crate existing. Every caller passes
//!   `cfg!(debug_assertions)` literally.
//! - No I/O, nothing spawned: a caller decides what to do with the `Err`.

/// 2026-10-04: `Err` names the fix. Call with `cfg!(debug_assertions)`.
pub fn refuse_debug_build(is_debug_build: bool) -> Result<(), String> {
    if is_debug_build {
        Err(
            "this binary is a debug build — rebuild with --release: a gate number from a \
             debug build is never valid"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-04: Path C — the refusal path: a debug build is refused, and
    /// the message names the fix. No override exists to test around.
    #[test]
    fn a_debug_build_is_refused_and_the_message_names_the_fix() {
        let err = refuse_debug_build(true).unwrap_err();
        assert!(err.contains("rebuild with --release"), "{err}");
        assert!(err.contains("debug build"), "{err}");
    }
}
