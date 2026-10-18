// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Resolution of the MTP drafter context from the environment: whether
//! the drafter KV is batch-prefilled over the prompt (`prefill`) and whether it is
//! carried across the turns of a session (`carry`).
//!
//! Carry depends on prefill. In the model's `trait_impl/speculative.rs` a carried
//! drafter KV is adopted (`try_carry_drafter`) only inside
//! `if !self.mtp_prefill_hidden.is_null()`, and the model allocates that capture
//! only when `drafter.prefill` is set (`layers::mtp_drafter_prefill_enabled`), so
//! the two are resolved together.
//! A configured carry is armed only in single-sequence MTP
//! (`mtp_carry::carry_armed`).
//!
//! | variable (kill switch: a settings switch; prefill-only: exactly `1`) | result |
//! |---|---|
//! | neither | `DrafterContext::BOTH` |
//! | `METRALE_NO_MTP_DRAFTER_CONTEXT` | `DrafterContext::OFF` |
//! | `METRALE_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE` | prefill on, carry off; warned at startup |
//!
//! Any other prefill-only value, `0` included, changes nothing; the kill switch reads `1/true/on/yes`
//! as on and `0/false/off/no` as off, and the settings resolve refuses anything else.
//!
//! Owner: model-layers (MTP drafter).
//! Invariants:
//! - [`resolve`] never returns `carry` without `prefill`.
//! - The kill switch wins over the prefill-only variable.

/// 2026-09-25: Kill switch for both halves (2026-10-17: the `speculative.no_mtp_drafter_context`
/// switch).
pub const DISABLE_ENV: &str = "METRALE_NO_MTP_DRAFTER_CONTEXT";

/// 2026-09-25: Prefill without carry; only the exact value `1` counts, and
/// [`DISABLE_ENV`] wins over it.
pub const PREFILL_ONLY_ENV: &str = "METRALE_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE";

/// 2026-09-25: Which halves of the drafter context are configured. [`resolve`]
/// never returns `carry` without `prefill`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterContext {
    /// 2026-09-25: Capture prompt hiddens and batch-prefill the drafter KV.
    pub prefill: bool,
    /// 2026-09-25: Carry the drafter KV across turns of a session; it runs only
    /// when `mtp_carry::carry_armed` also holds.
    pub carry: bool,
}

impl Default for DrafterContext {
    /// 2026-09-25: [`BOTH`](DrafterContext::BOTH). Written out because a derived
    /// `Default` would be [`OFF`](DrafterContext::OFF).
    fn default() -> Self {
        Self::BOTH
    }
}

impl DrafterContext {
    /// 2026-09-25: Both halves on: what [`resolve`] returns when neither variable is `1`.
    pub const BOTH: Self = Self {
        prefill: true,
        carry: true,
    };
    /// 2026-09-25: Both halves off: what [`resolve`] returns for the kill switch.
    pub const OFF: Self = Self {
        prefill: false,
        carry: false,
    };
}

/// 2026-09-25: Resolve the two levers' values with no environment access, so the rule is
/// unit-tested (SBIO). `disable` wins over `prefill_only`. 2026-10-17: `disable` is the
/// `speculative.no_mtp_drafter_context` switch; `prefill_only` keeps its exact-`1` text rule.
pub fn resolve(disable: bool, prefill_only: Option<&str>) -> DrafterContext {
    if disable {
        return DrafterContext::OFF;
    }
    if prefill_only == Some("1") {
        // 2026-09-25: The only branch that returns prefill without carry.
        return DrafterContext {
            prefill: true,
            carry: false,
        };
    }
    DrafterContext::BOTH
}

/// 2026-10-17: The policy from the resolved settings. `ModelLevers::from_settings` calls it, and
/// the result is `ModelLevers::drafter`. The outcome is logged once per process (`REPORTED`).
pub fn from_settings(
    s: &metrale_settings::groups::speculative::SpeculativeSettings,
) -> DrafterContext {
    let cfg = resolve(
        s.no_mtp_drafter_context,
        s.mtp_drafter_context_prefill_only_unsafe.as_deref(),
    );
    REPORTED.call_once(|| report(cfg));
    cfg
}

/// 2026-09-25: Makes `report` run once per process, on the first [`from_settings`].
static REPORTED: std::sync::Once = std::sync::Once::new();

fn report(cfg: DrafterContext) {
    // 2026-09-25: Report the armed carry (`mtp_carry::carry_armed`, the predicate
    // the runtime gates on), not the configured one: the MTP dispatch cap can
    // force the carry off under this config.
    let carry_armed = crate::mtp_carry::carry_armed(cfg);
    tracing::info!(
        "MTP drafter context: prefill={} carry={} ({}). Disable both with {}=1.",
        on_off(cfg.prefill),
        on_off(carry_armed),
        if cfg == DrafterContext::BOTH {
            "default"
        } else {
            "overridden by environment"
        },
        DISABLE_ENV,
    );
    if cfg.carry && !carry_armed {
        tracing::warn!(
            "MTP cross-turn carry is CONFIGURED ON but INERT: the MTP \
             dispatch cap is {} (>1), and the carry slot is single-sequence \
             by design, so it is force-disabled. Pass --mtp-max-seqs 1 \
             to arm it; leave it unset to keep multi-sequence MTP.",
            crate::speculative::mtp_max_seqs(),
        );
    }
    if cfg.prefill && !cfg.carry {
        tracing::warn!(
            "{PREFILL_ONLY_ENV}=1: drafter prefill is ON with cross-turn carry \
             OFF. This is a MEASUREMENT ARM, not a deployment — a warm turn \
             rebuilds the drafter for a measured 1136 ms against ~211 ms/turn \
             of decode saving (net -927 ms/turn, spent on TTFT).",
        );
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "ON" } else { "OFF" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lever_equivalence::{Old, Outcome, check, settings_from};

    #[test]
    fn bare_environment_enables_both() {
        assert_eq!(resolve(false, None), DrafterContext::BOTH);
    }

    #[test]
    fn kill_switch_disables_both() {
        assert_eq!(resolve(true, None), DrafterContext::OFF);
    }

    #[test]
    fn prefill_only_arm_disables_carry_only() {
        assert_eq!(
            resolve(false, Some("1")),
            DrafterContext {
                prefill: true,
                carry: false
            }
        );
    }

    #[test]
    fn kill_switch_beats_the_research_arm() {
        assert_eq!(resolve(true, Some("1")), DrafterContext::OFF);
    }

    #[test]
    fn only_exactly_one_arms_the_prefill_only_arm() {
        for v in ["0", "", "true", "yes", "2", "1 "] {
            assert_eq!(
                resolve(false, Some(v)),
                DrafterContext::BOTH,
                "{PREFILL_ONLY_ENV}={v:?} must not change anything",
            );
        }
    }

    /// 2026-10-17: The settings path against the old readers (`== "1"` on both): the kill switch
    /// is now a switch (`true` disables, garbage is refused); the prefill-only text keeps its
    /// exact-`1` rule.
    #[test]
    fn settings_resolution_matches_the_old_exact_one_readers() {
        let old = |disable: Option<&str>, prefill_only: Option<&str>| {
            if disable == Some("1") {
                DrafterContext::OFF
            } else if prefill_only == Some("1") {
                DrafterContext {
                    prefill: true,
                    carry: false,
                }
            } else {
                DrafterContext::BOTH
            }
        };
        let changes = check(
            DISABLE_ENV,
            Old::OptIn,
            |v| old(v, None),
            |s| from_settings(&s.speculative),
        );
        assert_eq!(
            changes,
            [
                (DISABLE_ENV, Some("true"), Outcome::Changed),
                (DISABLE_ENV, Some("garbage"), Outcome::Rejected),
            ]
        );
        let changes = check(
            PREFILL_ONLY_ENV,
            Old::Verbatim,
            |v| old(None, v),
            |s| from_settings(&s.speculative),
        );
        assert!(changes.is_empty(), "{changes:?}");
        let s = settings_from(&[(DISABLE_ENV, "1"), (PREFILL_ONLY_ENV, "1")]).unwrap();
        assert_eq!(from_settings(&s.speculative), DrafterContext::OFF);
    }
}
