// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The GLM DSA mixer's launch levers, resolved once from the startup settings:
//! [`DsaLevers`] (copied, part of `GlmLevers` as `levers.dsa`) and [`DsaOwnedLevers`] (the two
//! values that own heap data, held once per load behind an `Arc` on every DSA layer). Nothing
//! here reads the process environment.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `METRALE_GLM_DSA_DECODE_HB` outside `1..=HB_MAX_SPLITS` is an error at resolve, `0`
//!   included.
//! - The SP-select table and payload are parsed by one resolver ([`sp_resolve`]); the
//!   rank-agreement value ([`DsaLevers::sp_agreement`]) is derived from what it returns.
//! - `METRALE_GLM_DSA_SP_SELECT` and `METRALE_GLM_DSA_DECODE_HB_CHECK` keep their readers' text
//!   predicates.

use anyhow::{Result, anyhow};
use metrale_settings::InferenceServerSettings;

use crate::glm5next_default_on::under_parent;

use super::attend::HB_MAX_SPLITS;
use super::sp_select::MinPoolsTable;
use super::sp_select::lever::{self, Payload};

/// 2026-10-17: The prefill attend levers, applied in this order by
/// `Glm5NextDsaDecodeKernel::for_prefill_levers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsaPrefillLevers {
    /// 2026-10-12: `METRALE_GLM_DSA_HB_PREFILL_ONE_SPLIT`: a prefill sub-chunk's head-batched
    /// attend runs at one split in one launch over all its rows; decode keeps
    /// `METRALE_GLM_DSA_DECODE_HB`'s split count.
    pub one_split: bool,
    /// 2026-10-12: `METRALE_GLM_DSA_HB_PREFILL_P_HI`: the prefill attend on
    /// `glm5next_dsa_mla_decode_hb_fp8_phi` (softmax weights as one bf16; accuracy-gated).
    pub p_hi: bool,
    /// 2026-10-14: `METRALE_GLM_DSA_HB_PREFILL_PIPE`: the P-hi prefill attend on
    /// `glm5next_dsa_mla_prefill_pipe_phi` (byte-identical per row).
    pub pipe: bool,
    /// 2026-10-14: `METRALE_GLM_DSA_HB_PREFILL_TC`: the P-hi prefill attend at one split on
    /// `glm5next_dsa_mla_prefill_tc_phi` (byte-identical per row; exclusive with `pipe`).
    pub tc: bool,
    /// 2026-10-15: `METRALE_GLM_DSA_TC_NANOR`: the tc prefill attend on
    /// `glm5next_dsa_mla_prefill_tc_phi_nanor` (byte-identical; refused without `tc`).
    pub tc_nanor: bool,
}

impl DsaPrefillLevers {
    /// 2026-10-14: The readback of the prefill attend levers, logged once per process after
    /// the first prefill sub-chunk's attend applied them without error, so every rank's serve
    /// log shows which entry its prefill ran (each set lever is refused unless it was swapped
    /// in).
    pub(super) fn readback(&self) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "DSA prefill attend levers: one_split={} p_hi={} pipe={} tc={} tc_nanor={}",
                u8::from(self.one_split),
                u8::from(self.p_hi),
                u8::from(self.pipe),
                u8::from(self.tc),
                u8::from(self.tc_nanor)
            )
        });
    }
}

/// 2026-10-17: The prefill selection levers. Each takes a pass only where its kernel resolved
/// and the pass's shape is eligible (`select::*_for`); the selection scratch carries the copy it
/// was sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsaSelectLevers {
    /// 2026-10-10: `METRALE_GLM_DSA_SCORES_TILED`: the scores in query x pool tiles, bit for
    /// bit `dsa_index_scores`'s.
    pub scores_tiled: bool,
    /// 2026-10-12: `METRALE_GLM_DSA_SCORES_MMA`: the scores on BF16 tensor cores (not
    /// `dsa_index_scores`'s bits), ahead of the tiled path.
    pub scores_mma: bool,
    /// 2026-10-14: `METRALE_GLM_DSA_SCORES_PIPE`: where the tensor-core scores run, run them
    /// streamed from pools prepared once per compression.
    pub scores_pipe: bool,
    /// 2026-10-17: `METRALE_GLM_DSA_SELECT_FUSED`: where the streamed scores run, select in the
    /// scoring kernel.
    pub select_fused: bool,
    /// 2026-10-12: `METRALE_GLM_DSA_TOPK_RADIX`: each row's top pools by radix select (the same
    /// pools in the same order).
    pub topk_radix: bool,
    /// 2026-10-21: `METRALE_GLM_DSA_DECODE_TOPK_RADIX` (default on): a ceiling launch's top
    /// pools by radix select over the live geometry (the same pools in the same order).
    pub decode_topk_radix: bool,
    /// 2026-10-21: `METRALE_GLM_DSA_SCORES_WARP` (default on): `dsa_index_scores`'s launches
    /// run as `dsa_index_scores_warp` (the same bytes).
    pub scores_warp: bool,
    /// 2026-10-21: `METRALE_GLM_DSA_SCORES_HEADS` (default on): a ceiling launch's scores over
    /// the persisted pool keys run as `dsa_index_scores_heads_paged` (the same bytes).
    pub scores_heads: bool,
    /// 2026-10-22: `METRALE_GLM_DSA_SCORES_HEADS_WINDOW` (default on): where `scores_heads`
    /// runs a verify's rows, it runs as `dsa_index_scores_heads_window` (the same bytes, a
    /// straight-line pool step).
    pub scores_heads_window: bool,
    /// 2026-10-21: `METRALE_GLM_DSA_TOPK_MC` (default on): where the ceiling radix top-k runs,
    /// it runs over several CTAs per row (`dsa_topk_mc_*`; the same pools in the same order).
    pub topk_mc: bool,
}

/// 2026-10-17: The copied DSA levers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsaLevers {
    /// 2026-10-09: `METRALE_GLM_DSA_DECODE_HB`: `None` for the per-head decode kernel, else the
    /// head-batched kernel at this many splits.
    pub decode_hb: Option<usize>,
    pub prefill: DsaPrefillLevers,
    pub select: DsaSelectLevers,
    /// 2026-10-09: `METRALE_GLM_DSA_DECODE_HB_SERVE_REF` (diagnostic): after every head-batched
    /// launch the per-head kernel overwrites the output (`attend::check`).
    pub serve_ref: bool,
    /// 2026-10-15: The rank-agreement value of `METRALE_GLM_DSA_SP_SELECT` and its payload
    /// sub-lever (`sp_select::lever::agreement_of`, the payload in the low bit): 0 when off.
    pub sp_agreement: u64,
}

/// 2026-10-17: A head-batched split count in `1..=HB_MAX_SPLITS`, or unset.
fn decode_hb(v: Option<i64>) -> Result<Option<usize>> {
    v.map(|n| {
        usize::try_from(n)
            .ok()
            .filter(|n| (1..=HB_MAX_SPLITS).contains(n))
            .ok_or_else(|| {
                anyhow!(
                    "METRALE_GLM_DSA_DECODE_HB={n}: expected a split count in 1..={HB_MAX_SPLITS}"
                )
            })
    })
    .transpose()
}

/// 2026-10-15: The SP-select table (`None` off) and the payload a split pass gathers; a
/// malformed table, a malformed payload or the selected-pools payload without a table is an
/// error.
pub fn sp_resolve(s: &InferenceServerSettings) -> Result<(Option<MinPoolsTable>, Payload)> {
    let k = &s.kernels;
    let table = lever::parse(k.glm_dsa_sp_select.as_deref()).map_err(anyhow::Error::msg)?;
    let payload =
        lever::payload_of(k.glm_dsa_sp_select_sel, table.is_some()).map_err(anyhow::Error::msg)?;
    Ok((table, payload))
}

impl DsaLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing the values in the module
    /// invariants.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let k = &s.kernels;
        let (table, payload) = sp_resolve(s)?;
        let hb = decode_hb(k.glm_dsa_decode_hb)?;
        let (one_split, p_hi, pipe) = (
            k.glm_dsa_hb_prefill_one_split,
            k.glm_dsa_hb_prefill_p_hi,
            k.glm_dsa_hb_prefill_pipe,
        );
        // 2026-10-22: TC and its NaN-test twin are on by default; left at their defaults they run
        // only where their parents do (inert, reported once, elsewhere). Set to `1` without them,
        // `for_prefill_tc` / `for_prefill_tc_nanor` still refuse them.
        let tc = under_parent(
            s,
            "METRALE_GLM_DSA_HB_PREFILL_TC",
            k.glm_dsa_hb_prefill_tc,
            hb.is_some() && p_hi && one_split && !pipe,
            "METRALE_GLM_DSA_DECODE_HB with METRALE_GLM_DSA_HB_PREFILL_P_HI=1 and \
             METRALE_GLM_DSA_HB_PREFILL_ONE_SPLIT=1, without METRALE_GLM_DSA_HB_PREFILL_PIPE=1",
        );
        let tc_nanor = under_parent(
            s,
            "METRALE_GLM_DSA_TC_NANOR",
            k.glm_dsa_tc_nanor,
            tc,
            "METRALE_GLM_DSA_HB_PREFILL_TC",
        );
        Ok(Self {
            decode_hb: hb,
            prefill: DsaPrefillLevers {
                one_split,
                p_hi,
                pipe,
                tc,
                tc_nanor,
            },
            select: DsaSelectLevers {
                scores_tiled: k.glm_dsa_scores_tiled,
                scores_mma: k.glm_dsa_scores_mma,
                scores_pipe: k.glm_dsa_scores_pipe,
                select_fused: k.glm_dsa_select_fused,
                topk_radix: k.glm_dsa_topk_radix,
                decode_topk_radix: k.glm_dsa_decode_topk_radix,
                scores_warp: k.glm_dsa_scores_warp,
                scores_heads: k.glm_dsa_scores_heads,
                scores_heads_window: k.glm_dsa_scores_heads_window,
                topk_mc: k.glm_dsa_topk_mc,
            },
            serve_ref: s.telemetry.glm_dsa_decode_hb_serve_ref,
            sp_agreement: lever::agreement_of(table.as_ref())
                ^ u64::from(payload == Payload::Selected),
        })
    }

    /// 2026-10-17: The log line the serve-reference diagnostic used to print at its first read.
    pub(crate) fn announce(&self) {
        if self.serve_ref {
            tracing::warn!(
                "METRALE_GLM_DSA_DECODE_HB_SERVE_REF=1: every head-batched DSA decode is \
                 overwritten by the per-head kernel's output"
            );
        }
    }
}

/// 2026-10-17: The DSA levers that own heap data, resolved once per load and shared by the
/// stack's DSA layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsaOwnedLevers {
    /// 2026-10-15: `METRALE_GLM_DSA_SP_SELECT`: the sequence-parallel selection's split table,
    /// `None` off.
    pub sp_table: Option<MinPoolsTable>,
    /// 2026-10-15: `METRALE_GLM_DSA_SP_SELECT_SEL`: what a split pass gathers.
    pub sp_payload: Payload,
    /// 2026-10-09: `METRALE_GLM_DSA_DECODE_HB_CHECK` (diagnostic): the directory the
    /// head-batched decode check dumps to (`attend::check`), `None` off.
    pub hb_check: Option<String>,
}

impl DsaOwnedLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing what [`sp_resolve`] refuses.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let (sp_table, sp_payload) = sp_resolve(s)?;
        Ok(Self {
            sp_table,
            sp_payload,
            hb_check: s.telemetry.glm_dsa_decode_hb_check.clone(),
        })
    }

    /// 2026-10-17: The readback lines the SP-select levers used to print at their first read.
    pub fn announce(&self) {
        lever::readback(self.sp_table.as_ref(), self.sp_payload);
    }
}

#[cfg(test)]
impl DsaLevers {
    /// 2026-10-17: The levers of the declared defaults.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
impl DsaOwnedLevers {
    /// 2026-10-17: The owned levers of the declared defaults.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
