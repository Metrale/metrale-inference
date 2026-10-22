// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The plan arms a blueprint lists but cannot carry. The arms it carries are the
//! circuit's own [`metrale_circuit::FusionPlan`]s (SSOT).
//!
//! Owner: blueprint.
//! Invariants: none beyond the types.

use alloc::string::String;

use metrale_circuit::Mode;
use serde::{Deserialize, Serialize};

/// 2026-10-22: An arm the instance lists that no rule set covers, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanGap {
    /// 2026-10-22: Mode.
    pub mode: Mode,
    /// 2026-10-22: Rows.
    pub rows: u64,
    /// 2026-10-22: The fuser's refusal.
    pub reason: String,
}
