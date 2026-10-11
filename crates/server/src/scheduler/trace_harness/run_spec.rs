// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The speculation settings a scenario's [`RunOptions`] give the scheduler: the
//! controller's policy and the levers that arm it.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::runner::RunOptions;
use crate::scheduler::config::SpecPolicy;
use crate::scheduler::levers::SchedLevers;
use metrale_speculative::spec_ctl::decide::Objective;

impl RunOptions {
    /// 2026-10-10: Throughput when a scenario arms the controller: the harness has no recipe to
    /// name an objective, and its scenarios check tokens, which no objective changes.
    pub(super) fn spec_policy(&self) -> SpecPolicy {
        SpecPolicy {
            objective: (!self.mtp_gate_force).then_some(Objective::Throughput),
            dflash_depth_pinned: false,
        }
    }

    /// 2026-10-10: `--mtp-gate force` and the spec-entry pin as the scenario sets them.
    pub(super) fn apply_spec_levers(&self, levers: &mut SchedLevers) {
        levers.mtp_gate_force = self.mtp_gate_force;
        levers.spec_entry_pin_tokens = self.spec_entry_pin_tokens;
    }
}
