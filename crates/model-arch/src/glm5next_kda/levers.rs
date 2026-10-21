// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The GLM KDA mixer's launch levers, resolved once from the startup settings
//! ([`KdaLevers::from_settings`]) as part of `GlmLevers` (`levers.kda`). The KDA layer and the
//! layer steps read the fields; nothing here reads the process environment.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - A chunked-scan floor below its chunk width (other than `0`, off) is an error at resolve,
//!   not at the first prefill.
//! - `METRALE_GLM_KDA_SEQ_ROWS` keeps its reader's text predicate: `1` all, `decode` the
//!   batched decode only, anything else off.

use anyhow::{Result, anyhow};
use metrale_settings::InferenceServerSettings;

use super::chunk16::CHUNK16;
use super::chunk64::CHUNK64;

/// 2026-10-09: Which batched KDA steps run through the rows kernels (`rows.rs`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KdaSeqRows {
    /// 2026-10-09: Row by row (the default).
    Off,
    /// 2026-10-09: The batched decode only (one row per sequence).
    DecodeOnly,
    /// 2026-10-09: The batched decode and verify.
    All,
}

/// 2026-10-17: The KDA levers. Each field names its variable; the module that runs the lever
/// documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdaLevers {
    /// 2026-10-10: `METRALE_GLM_KDA_CHUNK_SCAN` (`chunk16.rs`): `None` off, else the narrowest
    /// prefill sub-chunk the 16-token scan takes.
    pub chunk_scan: Option<usize>,
    /// 2026-10-12: `METRALE_GLM_KDA_CHUNK64` (`chunk64.rs`): `None` off, else the narrowest
    /// prefill sub-chunk the 64-token scan takes.
    pub chunk64: Option<usize>,
    /// 2026-10-14: `METRALE_GLM_KDA_CHUNK64_CONV` (`chunk64_conv.rs`).
    pub chunk64_conv: bool,
    /// 2026-10-13: `METRALE_GLM_KDA_CONV_PACK` (`conv_pack.rs`).
    pub conv_pack: bool,
    /// 2026-10-21: `METRALE_GLM_KDA_CONV_SNAP_TO` (on unless `0`; `seq_tokens.rs`).
    pub conv_snap_to: bool,
    /// 2026-10-14: `METRALE_GLM_KDA_GATE_IN_PREPARE` (`gate_at.rs`).
    pub gate_in_prepare: bool,
    /// 2026-10-10: `METRALE_GLM_KDA_GATES_FUSED` (`gates_fused.rs`).
    pub gates_fused: bool,
    /// 2026-10-21: `METRALE_GLM_KDA_GATES_ROWS` (on unless `0`; `gates_fused.rs`).
    pub gates_rows: bool,
    /// 2026-09-25: `METRALE_GLM_KDA_NO_SMEM`: the 2R+2W recurrent kernel instead of the 1R+1W one.
    pub no_smem: bool,
    /// 2026-10-10: `METRALE_GLM_KDA_ROW_REG`: each single-row recurrent step (`stateful_row`)
    /// on `kda_recurrent_decode_bf16_seq_reg` at one token instead of
    /// `kda_recurrent_decode_bf16_smem` (`seq_tokens.rs`): the same expressions in the same order.
    pub row_reg: bool,
    /// 2026-10-09: `METRALE_GLM_KDA_ROWS_REG`: the batched decode's recurrence on the
    /// register-resident rows kernel (`rows.rs`).
    pub rows_reg: bool,
    /// 2026-10-09: `METRALE_GLM_KDA_SEQ_ROWS` (`rows.rs`).
    pub seq_rows: KdaSeqRows,
    /// 2026-10-09: `METRALE_GLM_KDA_SEQ_TOKENS`: the rows of one sequence that need no per-row
    /// snapshot step in one conv and one recurrent launch (`seq_tokens.rs`); same bits.
    pub seq_tokens: bool,
    /// 2026-10-11: `METRALE_GLM_KDA_VERIFY_TOKENS`: a 2..=4-row verify steps its rows through the
    /// per-sequence token kernels, storing each intermediate state as it passes
    /// (`stateful_tokens_snap`); same bits.
    pub verify_tokens: bool,
}

/// 2026-10-17: A chunked-scan floor: unset or `0` off, else at least `min` rows.
fn floor(env: &str, v: Option<i64>, min: usize) -> Result<Option<usize>> {
    match v {
        None | Some(0) => Ok(None),
        Some(n) => usize::try_from(n)
            .ok()
            .filter(|r| *r >= min)
            .map(Some)
            .ok_or_else(|| anyhow!("{env}={n}: expected 0 or a row count of at least {min}")),
    }
}

impl KdaLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing a floor below its chunk width.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let k = &s.kernels;
        let seq_rows = match k.glm_kda_seq_rows.as_deref() {
            Some("1") => KdaSeqRows::All,
            Some("decode") => KdaSeqRows::DecodeOnly,
            _ => KdaSeqRows::Off,
        };
        Ok(Self {
            chunk_scan: floor("METRALE_GLM_KDA_CHUNK_SCAN", k.glm_kda_chunk_scan, CHUNK16)?,
            chunk64: floor("METRALE_GLM_KDA_CHUNK64", k.glm_kda_chunk64, CHUNK64)?,
            chunk64_conv: k.glm_kda_chunk64_conv,
            conv_pack: k.glm_kda_conv_pack,
            conv_snap_to: k.glm_kda_conv_snap_to,
            gate_in_prepare: k.glm_kda_gate_in_prepare,
            gates_fused: k.glm_kda_gates_fused,
            gates_rows: k.glm_kda_gates_rows,
            no_smem: k.glm_kda_no_smem,
            row_reg: k.glm_kda_row_reg,
            rows_reg: k.glm_kda_rows_reg,
            seq_rows,
            seq_tokens: k.glm_kda_seq_tokens,
            verify_tokens: k.glm_kda_verify_tokens,
        })
    }
}

#[cfg(test)]
impl KdaLevers {
    /// 2026-10-17: The levers of the declared defaults.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
