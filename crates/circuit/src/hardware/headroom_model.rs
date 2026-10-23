// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The values of a headroom ([`super::headroom`]): what `met circuit headroom`
//! renders as Markdown and SVG and writes as `headroom/v1` JSON (`schemas/headroom.v1.json`).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: an `Option` measured field is `None` exactly when nothing was captured for it;
//! no value here is a default standing in for a measurement.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use serde::Serialize;

use super::HwError;
pub use super::headroom_ci::{SiteCi, TotalsCi};
pub use super::headroom_shares::PlanShare;
pub use super::headroom_sites::Outside;
use super::headroom_svg::Ceilings;
pub use super::headroom_trace::ExecutedBy;
use super::headroom_trace::{Stated, Trace};

/// 2026-10-22: Which term sets a site's floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// 2026-10-22: Bytes over bandwidth.
    Memory,
    /// 2026-10-22: FLOPs over the tensor peak of each node's MMA class.
    Compute,
    /// 2026-10-22: The roofline floor is below the launches' own floor.
    Launch,
}

impl Bound {
    /// 2026-10-22: The spelling in reports.
    pub fn name(self) -> &'static str {
        match self {
            Bound::Memory => "memory",
            Bound::Compute => "compute",
            Bound::Launch => "launch",
        }
    }
}

/// 2026-10-22: The energy constants of a class (pJ per DRAM byte, per FLOP by peak class, static
/// watts). No class states them yet; the report's energy columns stay blank until one does.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnergyRoofline {
    /// 2026-10-22: Dynamic DRAM energy, pJ/byte.
    pub pj_per_byte: f64,
    /// 2026-10-22: Dynamic MMA energy, pJ/FLOP (one figure for the site's FLOPs).
    pub pj_per_flop: f64,
    /// 2026-10-22: Where the constants come from.
    pub basis: String,
}

/// 2026-10-22: Measured time of a site, per step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measured {
    /// 2026-10-22: Launches captured per step.
    pub calls_per_step: f64,
    /// 2026-10-22: Σ launch durations per step, microseconds.
    pub total_us: f64,
    /// 2026-10-22: Σ non-overlapped time per step, microseconds.
    pub exclusive_us: f64,
    /// 2026-10-22: The captured kernels.
    pub kernels: Vec<String>,
    /// 2026-10-22: Kernels that are another point of the planned kernel's family.
    pub drift: Vec<String>,
}

/// 2026-10-22: The step after one change.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Upside {
    /// 2026-10-22: Step time saved, microseconds.
    pub saved_us: f64,
    /// 2026-10-22: The step, microseconds.
    pub step_us: f64,
    /// 2026-10-22: tok/s at that step.
    pub tok_s: f64,
}

/// 2026-10-22: Energy of a site per step; every field `None` without captured energy.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct EnergyCols {
    /// 2026-10-22: Captured µJ per launch.
    pub uj_per_launch: Option<f64>,
    /// 2026-10-22: Dynamic energy floor per step, µJ.
    pub floor_uj: Option<f64>,
    /// 2026-10-22: Floor over captured, percent.
    pub pct_of_floor: Option<f64>,
    /// 2026-10-22: Captured minus floor per step, µJ.
    pub upside_uj: Option<f64>,
}

/// 2026-10-22: One launch site's row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SiteRow {
    /// 2026-10-22: `mode n=rows`.
    pub run: String,
    /// 2026-10-22: `block [locals]`.
    pub site: String,
    /// 2026-10-22: The planned kernels of one group.
    pub kernels: String,
    /// 2026-10-22: Rules.
    pub rules: Vec<String>,
    /// 2026-10-22: Groups (layers) per step.
    pub groups: u64,
    /// 2026-10-22: Planned launches per step.
    pub launches: u64,
    /// 2026-10-22: Bytes per step at the floor.
    pub bytes: f64,
    /// 2026-10-22: FLOPs per step.
    pub flops: f64,
    /// 2026-10-22: Fused edges dropped from `bytes`.
    pub fused: bool,
    /// 2026-10-22: The floor, microseconds per step.
    pub floor_us: f64,
    /// 2026-10-22: Its bound.
    pub bound: Bound,
    /// 2026-10-22: Σ per-node estimate with every edge counted, microseconds.
    pub node_sum_us: f64,
    /// 2026-10-22: Captured time, or `None` (unmeasured: its projection is `floor_us`).
    pub measured: Option<Measured>,
    /// 2026-10-22: Floor over captured, percent.
    pub pct_of_floor: Option<f64>,
    /// 2026-10-22: Captured minus floor, microseconds.
    pub headroom_us: Option<f64>,
    /// 2026-10-22: The step if this site alone reached its floor.
    pub upside_full: Option<Upside>,
    /// 2026-10-22: The step if it reached `target_pct` of its floor.
    pub upside_target: Option<Upside>,
    /// 2026-10-22: Energy.
    pub energy: EnergyCols,
    /// 2026-10-22: Intervals of the measured fields; `None` without per-step series.
    pub ci: Option<SiteCi>,
    /// 2026-10-22: Its upside interval overlaps a neighbour's in the ranking: the order between
    /// them is not resolved.
    pub tied: bool,
}

/// 2026-10-22: A captured kernel outside the plan.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutsideRow {
    /// 2026-10-22: Kernel.
    pub kernel: String,
    /// 2026-10-22: The site the exporter named, if any, and its shape label.
    pub label: Option<String>,
    /// 2026-10-22: Launches per step.
    pub calls_per_step: f64,
    /// 2026-10-22: Σ durations per step, microseconds.
    pub total_us: f64,
    /// 2026-10-22: Σ exclusive per step, microseconds.
    pub exclusive_us: f64,
    /// 2026-10-22: Why it is outside.
    pub why: Outside,
}

/// 2026-10-22: One point of the cumulative "top N sites to floor" curve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CurvePoint {
    /// 2026-10-22: N.
    pub n: usize,
    /// 2026-10-22: The N-th site added.
    pub site: String,
    /// 2026-10-22: Every one of the first N at 100% of its floor.
    pub full: Upside,
    /// 2026-10-22: Every one of the first N at `target_pct`.
    pub target: Upside,
}

/// 2026-10-22: Step totals, microseconds per step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Totals {
    /// 2026-10-22: Σ site floors: the step ceiling.
    pub ceiling_us: f64,
    /// 2026-10-22: tok/s at the ceiling.
    pub ceiling_tok_s: f64,
    /// 2026-10-22: Σ floors of the sites the capture measured.
    pub measured_floor_us: Option<f64>,
    /// 2026-10-22: Σ projections of the unmeasured sites (their floors).
    pub unmeasured_projection_us: f64,
    /// 2026-10-22: The captured step.
    pub step_us: Option<f64>,
    /// 2026-10-22: tok/s at the captured step.
    pub step_tok_s: Option<f64>,
    /// 2026-10-22: Σ sites' durations (overlaps counted twice).
    pub sites_total_us: Option<f64>,
    /// 2026-10-22: Σ sites' exclusive time.
    pub sites_exclusive_us: Option<f64>,
    /// 2026-10-22: Σ outside-the-plan kernels' exclusive time.
    pub outside_exclusive_us: Option<f64>,
    /// 2026-10-22: Time two kernels ran at once (Σ durations − Σ exclusive).
    pub overlap_us: Option<f64>,
    /// 2026-10-22: Step − Σ exclusive: no kernel running.
    pub residual_us: Option<f64>,
    /// 2026-10-22: Intervals of the step, tok/s and the residual; `None` without series.
    pub ci: Option<TotalsCi>,
}

/// 2026-10-22: The headroom of one step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Headroom {
    /// 2026-10-22: `headroom/v1`.
    pub schema: String,
    /// 2026-10-22: Recipe or checkpoint label.
    pub model: String,
    /// 2026-10-22: Device id.
    pub device: String,
    /// 2026-10-22: Roofline basis.
    pub roofline: String,
    /// 2026-10-22: The roofline's bandwidth and compute ceilings.
    pub ceilings: Ceilings,
    /// 2026-10-22: The runs of the step (`mode n=rows`).
    pub runs: Vec<String>,
    /// 2026-10-22: KV context per sequence, and its basis.
    pub kv_tokens: Stated,
    /// 2026-10-22: Tokens per step, and its basis.
    pub tokens_per_step: Stated,
    /// 2026-10-22: Launch floor, if stated.
    pub launch_floor_us: Option<Stated>,
    /// 2026-10-22: Dims replaced on the instance, and why.
    pub dims: BTreeMap<String, u64>,
    /// 2026-10-22: Why those dims.
    pub dims_basis: Option<String>,
    /// 2026-10-22: The fraction of the floor the target upside assumes, percent.
    pub target_pct: f64,
    /// 2026-10-22: The capture's fingerprint; `None`: no capture, every site unmeasured.
    pub trace: Option<String>,
    /// 2026-10-22: Energy basis; `None`: no energy roofline, energy floors blank.
    pub energy: Option<String>,
    /// 2026-10-22: Launch sites, by descending upside, then floor.
    pub sites: Vec<SiteRow>,
    /// 2026-10-22: Captured kernels outside the plan, by descending exclusive time.
    pub outside: Vec<OutsideRow>,
    /// 2026-10-22: The cumulative curve.
    pub curve: Vec<CurvePoint>,
    /// 2026-10-22: Totals.
    pub totals: Totals,
    /// 2026-10-23: Per run, the plan-describable and plan-executed shares of the step
    /// ([`super::headroom_shares`]); empty without a capture.
    pub shares: Vec<PlanShare>,
    /// 2026-10-22: What the intervals cover, or why there are none.
    pub uncertainty: String,
}

/// 2026-10-22: Why a headroom could not be computed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HeadroomError {
    /// 2026-10-22: Planning.
    #[error(transparent)]
    Hw(#[from] HwError),
    /// 2026-10-22: The capture is unusable.
    #[error("{0}")]
    Trace(String),
    /// 2026-10-22: The capture's kernels do not match the plan.
    #[error("the trace does not match the plan:\n  {}", .0.join("\n  "))]
    Mismatch(Vec<String>),
}

/// 2026-10-22: What the step is computed from, besides the plans.
#[derive(Debug, Clone)]
pub struct Inputs<'a> {
    /// 2026-10-22: Model label.
    pub model: String,
    /// 2026-10-22: KV context; `None`: the class's `context_tokens` assumption.
    pub kv_tokens: Option<Stated>,
    /// 2026-10-22: Tokens per step.
    pub tokens_per_step: Stated,
    /// 2026-10-22: Launch floor.
    pub launch_floor_us: Option<Stated>,
    /// 2026-10-22: Dims the circuit was planned with in place of the instance's, and why.
    pub dims: (BTreeMap<String, u64>, Option<String>),
    /// 2026-10-22: Target fraction of the floor, percent (0, 100].
    pub target_pct: f64,
    /// 2026-10-22: The capture.
    pub trace: Option<&'a Trace>,
    /// 2026-10-22: Energy constants.
    pub energy: Option<EnergyRoofline>,
}
