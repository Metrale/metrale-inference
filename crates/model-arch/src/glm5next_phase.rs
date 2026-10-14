// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The forward phase this thread is running for GLM-5.3, so a split
//! `--dense-quantization decode=w4a16,prefill=fp8` (or `--dsa-quantization decode=w4a16,
//! prefill=w8a8`) serves the prompt pass on the FP8 copies and everything else (decode, verify,
//! the MTP draft steps) on the W4A16 weights. The prompt pass is `Glm5NextLayer::prefill` and
//! `prefill_spans`; each holds a [`PrefillPhase`] guard for its whole body.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Outside a guard the phase is decode; dropping a guard restores the phase it found, so a
//!   nested prompt pass (a layer's prefill inside a model-level one) leaves the outer one set.
//! - Per thread: a guard on one thread never moves another thread's launches.

use std::cell::Cell;

thread_local! {
    static PREFILL: Cell<bool> = const { Cell::new(false) };
}

/// 2026-10-14: While alive, this thread's launches are the prompt pass's.
#[must_use = "the phase holds only while the guard lives"]
pub struct PrefillPhase {
    outer: bool,
}

/// 2026-10-14: Enter the prompt pass until the guard drops.
pub fn enter_prefill() -> PrefillPhase {
    PrefillPhase {
        outer: PREFILL.with(|p| p.replace(true)),
    }
}

impl Drop for PrefillPhase {
    fn drop(&mut self) {
        PREFILL.with(|p| p.set(self.outer));
    }
}

/// 2026-10-14: Whether this thread is inside a prompt pass.
pub fn in_prefill() -> bool {
    PREFILL.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-14: The phase is decode outside a guard, prefill inside, the outer value after a
    /// nested guard drops, and never another thread's.
    #[test]
    fn the_guard_scopes_the_phase_per_thread() {
        assert!(!in_prefill());
        {
            let _outer = enter_prefill();
            assert!(in_prefill());
            {
                let _inner = enter_prefill();
                assert!(in_prefill());
            }
            assert!(in_prefill(), "the inner guard restored the outer pass");
            assert!(
                !std::thread::spawn(in_prefill).join().unwrap(),
                "another thread stays in decode"
            );
        }
        assert!(!in_prefill());
    }
}
