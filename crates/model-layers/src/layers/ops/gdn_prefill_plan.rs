// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The levers the chunked GDN prefill (`ssm_gdn_a3::gdn_prefill_fla`) and the
//! layer that binds its scalar spine (`qwen3_ssm::init_kernels::fused_spine_kernel`) run under,
//! resolved once from the startup settings ([`GdnPrefillPlan::from_settings`]) and carried on
//! `OpsLevers::gdn_prefill`.
//!
//! Owner: model-layers ops (GDN).
//! Invariants:
//! - The layer binds and the prefill launches the same [`GdnPrefillPlan::spine`], so the
//!   prefill's block size and shared memory match the entry that was bound.
//! - `METRALE_GDN_PIPE` and `METRALE_GDN_VTILE` pick the spine only when set (by the
//!   environment, a recipe or the command line); at their declared defaults the pipe spine
//!   runs, as with both variables unset before.
//! - `METRALE_NO_GDN_PREFILL_TC_REMNANTS` keeps its reader's text predicate: exactly `1`.

use super::GdnScalarSpine;
use metrale_settings::InferenceServerSettings;
use metrale_settings::provenance::Source;

/// 2026-10-17: The chunked GDN prefill levers. Each field names its variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnPrefillPlan {
    /// 2026-09-28: The scalar state spine: `_pipe` unless `METRALE_GDN_PIPE` is switched
    /// off (`_vfused`) or `METRALE_GDN_VTILE` is set on while `METRALE_GDN_PIPE` is not
    /// (`_vtile`).
    pub spine: GdnScalarSpine,
    /// 2026-09-25: On unless `METRALE_GDN_VTILE` is switched off: the bound scalar spine
    /// runs, else `_tc_vblock` or `_ksplit`.
    pub fused: bool,
    /// 2026-09-25: `METRALE_GDN_TC_VBLOCK`: the DV-block-split spine runs where the fused one
    /// does not and its handle resolved.
    pub tc_vblock: bool,
    /// 2026-09-25: `METRALE_GDN_TMA`: the TMA spine is requested.
    pub tma: bool,
    /// 2026-09-28: On unless `METRALE_NO_GDN_FWD_O_MMA8` is present at any value: kernel 3
    /// runs its 8-warp twin where it may (`ssm_gdn_hopper_prefill::gdn_fwd_o_mma8`).
    pub fwd_o_mma8: bool,
    /// 2026-09-25: `METRALE_NO_GDN_PREFILL_TC_REMNANTS` is exactly `1`: kernels 1 and 3 stay
    /// on their parents while the tensor-core spine runs
    /// (`ssm_gdn_hopper_prefill::gdn_hopper_remnants`).
    pub remnants_killed: bool,
}

/// 2026-10-17: The spine of the two switches and whether each was set: a set `PIPE` on wins,
/// then a set `VTILE` on, then `PIPE` off; else the pipe spine.
fn spine_of(pipe: bool, pipe_set: bool, vtile: bool, vtile_set: bool) -> GdnScalarSpine {
    if pipe && pipe_set {
        GdnScalarSpine::Pipe
    } else if vtile && vtile_set {
        GdnScalarSpine::Vtile
    } else if !pipe {
        GdnScalarSpine::Vfused
    } else {
        GdnScalarSpine::Pipe
    }
}

impl GdnPrefillPlan {
    /// 2026-10-17: The plan of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let set = |env: &str| {
            s.provenance
                .source_of(env)
                .is_some_and(|src| src != Source::Default)
        };
        let k = &s.kernels;
        Self {
            spine: spine_of(
                k.gdn_pipe,
                set("METRALE_GDN_PIPE"),
                k.gdn_vtile,
                set("METRALE_GDN_VTILE"),
            ),
            fused: k.gdn_vtile,
            tc_vblock: s.cache.gdn_tc_vblock,
            tma: k.gdn_tma,
            fwd_o_mma8: !k.no_gdn_fwd_o_mma8,
            remnants_killed: s.scheduler.no_gdn_prefill_tc_remnants.as_deref() == Some("1"),
        }
    }

    /// 2026-10-17: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            spine: GdnScalarSpine::Pipe,
            fused: true,
            tc_vblock: false,
            tma: false,
            fwd_o_mma8: true,
            remnants_killed: false,
        }
    }
}

#[cfg(test)]
#[path = "gdn_prefill_plan_tests.rs"]
mod tests;
