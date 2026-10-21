// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The GLM-5.3 MTP drafter's levers, resolved once from the startup settings
//! ([`MtpLevers::from_settings`]). The drafter head keeps the copy (`Glm5NextMtpHead::levers`);
//! its loader, the per-sequence reserve and the ranks' agreement check resolve the same settings.
//! Nothing here reads the process environment.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - `METRALE_GLM_MTP_SKIP` and `METRALE_GLM_MTP_PREFILL_FULL` keep their readers' text
//!   predicates: `head` / `block` and `1` exactly; any other value is the default.
//! - A negative `METRALE_GLM_MTP_CONTEXT_CAP` is refused, as the old reader refused it.

use anyhow::{Result, anyhow};
use metrale_settings::InferenceServerSettings;

/// 2026-10-17: `METRALE_GLM_MTP_SKIP`, the drafter's timing arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpSkip {
    /// 2026-10-17: Unset or any other value: the whole drafter runs.
    None,
    /// 2026-09-25: `head`: stop after the block, before `shared_head.norm`, the `lm_head` gemv,
    /// the argmax and the D2H; the draft is token 0.
    Head,
    /// 2026-09-25: `block`: skip the MTP block itself and run only the head.
    Block,
}

/// 2026-10-17: The drafter levers. Each field names its variable; the module that runs the lever
/// documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpLevers {
    /// 2026-09-25: `METRALE_GLM_MTP_SKIP`.
    pub skip: MtpSkip,
    /// 2026-10-12: `METRALE_GLM_MTP_PREFILL_FULL=1` (`rows`): context rows through the full
    /// block, a diagnostic arm.
    pub prefill_full: bool,
    /// 2026-10-13: `METRALE_GLM_MTP_CONTEXT_CAP` (`context_cap`): 0 = no cap.
    pub context_cap: usize,
    /// 2026-10-11: `METRALE_GLM_MTP_GRAPH` (`graph`).
    pub graph: bool,
    /// 2026-10-12: `METRALE_GLM_MTP_EAGER_PREFILL` (on unless switched off; `proposer`).
    pub eager_prefill: bool,
    /// 2026-10-11: `METRALE_GLM_MTP_PROPOSE_BATCH` (`batch`).
    pub propose_batch: bool,
    /// 2026-10-11: `METRALE_GLM_MTP_PROPOSE_BATCH_CHECK` (`batch_check`).
    pub propose_batch_check: bool,
    /// 2026-10-11: `METRALE_GLM_MTP_EH_TP` (`init::eh_shard`).
    pub eh_tp: bool,
    /// 2026-10-11: `METRALE_GLM_MTP_XCHG_DEVICE` (`init::xchg_kernels`).
    pub xchg_device: bool,
    /// 2026-09-25: `METRALE_GLM_MTP_HEAD_FP8` (on unless switched off): the FP8 draft head copy.
    pub head_fp8: bool,
    /// 2026-10-21: `METRALE_GLM_GRAPH_EARLY_LAUNCH` (`graph`): the drafter graph launched early.
    pub graph_early_launch: bool,
    /// 2026-10-21: `METRALE_GLM_MTP_ROW_NOSYNC` (`rows`): the one-row context write uploads its
    /// KV slot without a host wait.
    pub row_nosync: bool,
}

impl MtpLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing a negative context cap.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let sp = &s.speculative;
        let cap = sp.glm_mtp_context_cap;
        Ok(Self {
            skip: match sp.glm_mtp_skip.as_deref() {
                Some("head") => MtpSkip::Head,
                Some("block") => MtpSkip::Block,
                _ => MtpSkip::None,
            },
            prefill_full: sp.glm_mtp_prefill_full.as_deref() == Some("1"),
            context_cap: usize::try_from(cap).map_err(|_| {
                anyhow!(
                    "METRALE_GLM_MTP_CONTEXT_CAP={cap}: expected a row count (0 = no cap, unset = \
                     the declared default)"
                )
            })?,
            graph: sp.glm_mtp_graph,
            eager_prefill: sp.glm_mtp_eager_prefill,
            propose_batch: sp.glm_mtp_propose_batch,
            propose_batch_check: sp.glm_mtp_propose_batch_check,
            eh_tp: sp.glm_mtp_eh_tp,
            xchg_device: sp.glm_mtp_xchg_device,
            head_fp8: sp.glm_mtp_head_fp8,
            graph_early_launch: sp.glm_graph_early_launch,
            row_nosync: sp.glm_mtp_row_nosync,
        })
    }

    /// 2026-10-17: Whether `METRALE_GLM_MTP_SKIP=head`.
    pub fn skip_head(&self) -> bool {
        self.skip == MtpSkip::Head
    }

    /// 2026-10-17: Whether `METRALE_GLM_MTP_SKIP=block`.
    pub fn skip_block(&self) -> bool {
        self.skip == MtpSkip::Block
    }

    /// 2026-10-11: Rows the MTP block's DSA and MLP workspaces are built for: the widest batch
    /// under the batched propose, one row otherwise.
    pub fn block_rows(&self) -> usize {
        if self.propose_batch {
            super::batch::PROPOSE_BATCH_MAX
        } else {
            1
        }
    }

    /// 2026-10-11: Rows the MTP block's DSA workspace is built for: [`Self::block_rows`] and a
    /// context-row pass (`ctx_rows`).
    pub fn dsa_rows(&self) -> usize {
        self.block_rows().max(super::ctx_rows::CTX_ROWS_CHUNK)
    }

    /// 2026-10-12: Whether the drafter prefill waits for the first propose: the full-block arm
    /// always waits; otherwise only a switched-off eager prefill defers it.
    pub fn prefill_waits(&self) -> bool {
        self.prefill_full || !self.eager_prefill
    }
}

#[cfg(test)]
impl MtpLevers {
    /// 2026-10-17: The levers of the declared defaults.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
