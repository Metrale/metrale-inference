// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: One box's memory under a tensor-parallel split, as a typed plan: the util budget
//! its device class allows, every resident term by name, and the KV room left for sequences.
//! A util above the class's ceiling is refused when the [`Util`] is built, and terms over the
//! budget are refused when the [`BoxPlan`] is built, so a plan that exists is one the box can
//! boot.
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - Pure (SBIO): the device size, the class's driver terms and every resident term arrive as
//!   arguments; the caller reads DEVICES.toml, HARDWARE.toml and the checkpoint.
//! - The ceiling is the class's `[memory] util_ceiling` ([`DriverTerms`]); nothing here holds
//!   a util of its own.
//! - Every term is an explicit input; an unknown term is a refusal at the caller, never a 0
//!   defaulted here. The driver's term is derived from the budget ([`DriverTerms::bytes`]).

use super::budget::{DriverTerms, util_budget};

/// 2026-10-18: Why a box plan was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BoxPlanError {
    /// 2026-10-18: A util outside `(0, ceiling]`.
    #[error("util {util} is outside (0, {ceiling}], the class's [memory] util_ceiling")]
    Util {
        /// 2026-10-18: The util asked for.
        util: f64,
        /// 2026-10-18: The class's ceiling.
        ceiling: f64,
    },
    /// 2026-10-18: The resident terms exceed the util budget.
    #[error("resident {resident} B exceeds the util budget {budget} B")]
    Over {
        /// 2026-10-18: Sum of the terms, bytes.
        resident: u64,
        /// 2026-10-18: The util budget, bytes.
        budget: u64,
    },
}

/// 2026-10-18: A `--gpu-memory-utilization` the device class allows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Util(f64);

impl Util {
    /// 2026-10-18: `util`, refused above `driver.util_ceiling` or at or below 0.
    pub fn new(util: f64, driver: &DriverTerms) -> Result<Self, BoxPlanError> {
        if util > 0.0 && util <= driver.util_ceiling {
            Ok(Self(util))
        } else {
            Err(BoxPlanError::Util {
                util,
                ceiling: driver.util_ceiling,
            })
        }
    }

    /// 2026-10-18: The fraction.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// 2026-10-18: A resident term of one box, in the order a plan lists them; the
/// discriminant is its index in [`BoxPlan::terms`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoxTerm {
    /// 2026-10-18: The rank's share of the main model's weights.
    Weights,
    /// 2026-10-18: The rank's share of a draft model's weights.
    Drafter,
    /// 2026-10-18: The CUDA context and the driver's bookkeeping.
    Driver,
    /// 2026-10-18: Collective buffers.
    Comm,
    /// 2026-10-18: Kernel scratch, graph pools and activations, as a cap.
    Workspace,
    /// 2026-10-18: Host-fed rows in flight (pinned staging).
    Staging,
}

/// 2026-10-18: What a box plan is built from; every term in bytes.
#[derive(Debug, Clone, Copy)]
pub struct BoxInputs {
    /// 2026-10-18: The device's visible memory.
    pub device_bytes: u64,
    /// 2026-10-18: The util the box serves at.
    pub util: Util,
    /// 2026-10-18: The class's driver terms.
    pub driver: DriverTerms,
    /// 2026-10-18: The rank's main weights.
    pub weights: u64,
    /// 2026-10-18: The rank's draft weights.
    pub drafter: u64,
    /// 2026-10-18: Collective buffers.
    pub comm: u64,
    /// 2026-10-18: The workspace cap.
    pub workspace: u64,
    /// 2026-10-18: Staging buffers.
    pub staging: u64,
}

/// 2026-10-18: One box's budget, its resident terms and the room left for KV and per-sequence
/// state.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxPlan {
    /// 2026-10-18: The util it was planned at.
    pub util: Util,
    /// 2026-10-18: `device_bytes x util`.
    pub budget_bytes: u64,
    /// 2026-10-18: Every resident term, in [`BoxTerm`] order.
    pub terms: [(BoxTerm, u64); 6],
    /// 2026-10-18: `budget - resident`.
    pub kv_room: u64,
}

impl BoxPlan {
    /// 2026-10-18: The plan of `inp`, refused when its terms exceed the budget.
    pub fn new(inp: &BoxInputs) -> Result<Self, BoxPlanError> {
        let budget = util_budget(inp.device_bytes, inp.util.get());
        let terms = [
            (BoxTerm::Weights, inp.weights),
            (BoxTerm::Drafter, inp.drafter),
            (BoxTerm::Driver, inp.driver.bytes(budget, 0)),
            (BoxTerm::Comm, inp.comm),
            (BoxTerm::Workspace, inp.workspace),
            (BoxTerm::Staging, inp.staging),
        ];
        let resident = terms
            .iter()
            .try_fold(0u64, |s, (_, b)| s.checked_add(*b))
            .unwrap_or(u64::MAX);
        let kv_room = budget
            .checked_sub(resident)
            .ok_or(BoxPlanError::Over { resident, budget })?;
        Ok(Self {
            util: inp.util,
            budget_bytes: budget,
            terms,
            kv_room,
        })
    }

    /// 2026-10-18: The bytes of `term`.
    pub fn term(&self, term: BoxTerm) -> u64 {
        self.terms[term as usize].1
    }

    /// 2026-10-18: The most sequences of `per_seq` bytes each (KV over the context plus the
    /// per-sequence state) the room holds; `None` for a zero `per_seq`.
    pub fn sequences(&self, per_seq: u64) -> Option<u64> {
        self.kv_room.checked_div(per_seq)
    }
}

#[cfg(test)]
#[path = "box_plan_tests.rs"]
mod box_plan_tests;
