// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `--prefill-tail-mode`: how a prefill pass that captures its SSM state in-pass at
//! the prefix cache's tail split point runs the rows after that point. `split` (the default)
//! runs the pass's rows as separate segments ending and starting at the capture points, so
//! the rows after the point carry the bits of a separate pass starting there, the pass a warm
//! request that restores the captured state runs. `merge` runs the pass's rows in its ordinary
//! sub-chunks and splits only the KDA conv and recurrence at the capture points (to take the
//! captured state there): every weight launch covers the tail rows with the rows before them,
//! so a short prompt does not read every layer's weights a second time for its last partial
//! block. Under `merge` a warm request's replayed tail rows may differ from the cold pass's
//! in low bits (disclosure PREFILL-TAILMERGE-1); a cold request is deterministic either way.
//!
//! Owner: gpu-runtime (process-wide serve switches).
//! Invariants:
//! - The first publication or read wins (`OnceLock`); the serve publishes before the model is
//!   built, and anything that reads first fixes it at `split`.

use std::sync::OnceLock;

/// 2026-10-14: The tail modes of an in-pass SSM capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrefillTailMode {
    /// 2026-10-14: The rows after a capture point run as a separate segment.
    #[default]
    Split,
    /// 2026-10-14: One run of sub-chunks; only the KDA conv and recurrence split at a point.
    Merge,
}

impl PrefillTailMode {
    /// 2026-10-14: Every mode, in the order the flag lists them.
    pub const ALL: [Self; 2] = [Self::Split, Self::Merge];

    /// 2026-10-14: The flag value, recipe value and record value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Merge => "merge",
        }
    }

    /// 2026-10-14: What the mode does to the tail rows, for the boot log.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Split => "tail rows run as their own segment (warm replay bit-exact)",
            Self::Merge => {
                "tail rows share the pass's sub-chunks, KDA split at the capture point \
                 (PREFILL-TAILMERGE-1: a warm replay's tail rows may differ in low bits)"
            }
        }
    }
}

static PREFILL_TAIL_MODE: OnceLock<PrefillTailMode> = OnceLock::new();

/// 2026-10-14: Publish `--prefill-tail-mode`. Returns the mode in force; a caller that gets a
/// different one should warn.
pub fn set_prefill_tail_mode_from_cli(m: PrefillTailMode) -> PrefillTailMode {
    let _ = PREFILL_TAIL_MODE.set(m);
    *PREFILL_TAIL_MODE.get().expect("just set")
}

/// 2026-10-14: The mode in force: `split` unless the serve published another.
pub fn prefill_tail_mode() -> PrefillTailMode {
    *PREFILL_TAIL_MODE.get_or_init(PrefillTailMode::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-14: The names are the flag's values, distinct and in `ALL` order, and `split`
    /// is the default.
    #[test]
    fn names_order_and_the_split_default() {
        let names: Vec<&str> = PrefillTailMode::ALL.iter().map(|m| m.name()).collect();
        assert_eq!(names, ["split", "merge"]);
        assert_eq!(PrefillTailMode::default(), PrefillTailMode::Split);
        assert!(
            PrefillTailMode::Merge
                .describe()
                .contains("PREFILL-TAILMERGE-1")
        );
    }

    /// 2026-10-14: The first publication is final: a later one gets the first back, so the
    /// serve can warn that its value did not take effect. The cell is process-global with no
    /// reset, so no other test in this crate may touch it.
    #[test]
    fn the_first_publication_wins() {
        assert_eq!(
            set_prefill_tail_mode_from_cli(PrefillTailMode::Merge),
            PrefillTailMode::Merge
        );
        assert_eq!(
            set_prefill_tail_mode_from_cli(PrefillTailMode::Split),
            PrefillTailMode::Merge
        );
        assert_eq!(prefill_tail_mode(), PrefillTailMode::Merge);
    }
}
