// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The GLM dense-projection tier levers (the W4A16 tier's launch shapes, the
//! dequantize + cuBLASLt prefill paths, the persistent BF16 copies, batch invariance and the MTP
//! drafter's tiers), resolved once from the startup settings ([`DenseLevers::from_settings`]) as
//! part of `GlmLevers` (`levers.dense`). Nothing here reads the process environment.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A value a reader refused is refused at resolve; a lever set without the tier it runs on is
//!   refused by [`DenseLevers::check_tiers`], which every GLM load calls once the tiers are known.
//! - `METRALE_GLM_W4A16_SEG` and `METRALE_GLM_DEQUANT_PERSIST` keep their readers' text
//!   predicates (`dequant_persist::parse` is the one parser of the second).

use anyhow::{Result, anyhow, bail};
use metrale_settings::InferenceServerSettings;

use crate::glm5next_dequant_persist::{self as dequant_persist, Persist};
use crate::glm5next_mlp::forward::count;
use crate::glm5next_w4a16_seg::{GEMV_ROWS_MAX, SegMode};

/// 2026-10-17: The dense tier levers. Each field names its variable; the module that runs the
/// lever documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseLevers {
    /// 2026-10-09: `METRALE_GLM_W4A16_SEG` (`glm5next_w4a16_seg`).
    pub seg: SegMode,
    /// 2026-10-10: `METRALE_GLM_W4A16_GEMV_ROWS`: 0 off, else the widest launch on the GEMV
    /// tiers, 2..=`GEMV_ROWS_MAX`.
    pub gemv_rows: u32,
    /// 2026-10-10: `METRALE_GLM_W4A16_GEMV_SEG`: one-row launches on `w4a16_gemv_seg`.
    pub gemv_seg: bool,
    /// 2026-10-12: `METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS` (`glm5next_w4a16_tc`): `None` off.
    pub tc_min_rows: Option<usize>,
    /// 2026-10-10: `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS` (`glm5next_w4a16_wide`): `None` off.
    pub dequant_min_rows: Option<usize>,
    /// 2026-10-12: `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS` (`glm5next_fp8_wide`): `None` off.
    pub fp8_dequant_min_rows: Option<usize>,
    /// 2026-10-15: `METRALE_GLM_DEQUANT_PERSIST` (`glm5next_dequant_persist`).
    pub persist: Persist,
    /// 2026-10-11: `METRALE_GLM_BATCH_INVARIANT` (`glm5next_batch_inv`).
    pub batch_invariant: bool,
    /// 2026-10-09: `METRALE_GLM_BATCHM_WIDE` (on unless switched off): wide BF16 projections may
    /// take the batched GEMV (`glm5next_layer::wide_gemv`).
    pub batchm_wide: bool,
    /// 2026-10-11: `METRALE_GLM_MTP_DRAFTER_TIERS`: the MTP drafter's dense weights register on
    /// the dense tiers (`glm5_next_load::register_mtp_dense_tiers`).
    pub mtp_drafter_tiers: bool,
}

/// 2026-10-17: The W4A16 segment mode: unset or `0` off, `fuse`, `split`; another value is an
/// error.
fn seg(v: Option<&str>) -> Result<SegMode> {
    match v {
        None | Some("0") => Ok(SegMode::Off),
        Some("fuse") => Ok(SegMode::Fuse),
        Some("split") => Ok(SegMode::Split),
        Some(other) => Err(anyhow!(
            "METRALE_GLM_W4A16_SEG={other:?}: expected 0, fuse or split"
        )),
    }
}

/// 2026-10-17: A row floor of `floor` rows or more, or off.
fn min_rows(env: &str, v: Option<i64>, floor: usize) -> Result<Option<usize>> {
    count(
        env,
        v,
        |r| r >= floor,
        &format!("a row count of at least {floor}"),
    )
}

impl DenseLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing the values in the module
    /// invariants (the tier check is [`Self::check_tiers`]).
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let k = &s.kernels;
        let gemv_rows = count(
            "METRALE_GLM_W4A16_GEMV_ROWS",
            k.glm_w4a16_gemv_rows,
            |r| (2..=GEMV_ROWS_MAX as usize).contains(&r),
            &format!("2..={GEMV_ROWS_MAX}"),
        )?;
        let tc_min_rows = min_rows(
            "METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS",
            k.glm_w4a16_tc_gemm_min_rows,
            crate::glm5next_w4a16_tc::MIN_ROWS_FLOOR,
        )?;
        let dequant_min_rows = min_rows(
            "METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS",
            k.glm_w4a16_dequant_gemm_min_rows,
            crate::glm5next_w4a16_wide::MIN_ROWS_FLOOR,
        )?;
        let fp8_dequant_min_rows = min_rows(
            "METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS",
            k.glm_fp8_dequant_gemm_min_rows,
            crate::glm5next_fp8_wide::MIN_ROWS_FLOOR,
        )?;
        let persist = dequant_persist::parse(
            k.glm_dequant_persist.as_deref(),
            dequant_min_rows.is_some(),
            tc_min_rows.is_some(),
            fp8_dequant_min_rows.is_some(),
        )
        .map_err(anyhow::Error::msg)?;
        Ok(Self {
            seg: seg(k.glm_w4a16_seg.as_deref())?,
            gemv_rows: gemv_rows.map_or(0, |r| r as u32),
            gemv_seg: k.glm_w4a16_gemv_seg,
            tc_min_rows,
            dequant_min_rows,
            fp8_dequant_min_rows,
            persist,
            batch_invariant: k.glm_batch_invariant,
            batchm_wide: k.glm_batchm_wide,
            mtp_drafter_tiers: s.speculative.glm_mtp_drafter_tiers,
        })
    }

    /// 2026-10-17: Refuse a lever set without the tier it runs on: the W4A16 levers without
    /// `--dense-quantization w4a16` (`w4a16_dense`), the FP8 dequant GEMM without the FP8 registry
    /// (`fp8_registry`). The first refused lever names the error.
    pub fn check_tiers(&self, w4a16_dense: bool, fp8_registry: bool) -> Result<()> {
        const W4A16: &str = "needs --dense-quantization w4a16";
        if !w4a16_dense {
            match self.seg {
                SegMode::Off => {}
                SegMode::Fuse => bail!("METRALE_GLM_W4A16_SEG=fuse {W4A16}"),
                SegMode::Split => bail!("METRALE_GLM_W4A16_SEG=split {W4A16}"),
            }
            if self.gemv_rows > 0 {
                bail!("METRALE_GLM_W4A16_GEMV_ROWS={} {W4A16}", self.gemv_rows);
            }
            if self.gemv_seg {
                bail!("METRALE_GLM_W4A16_GEMV_SEG=1 {W4A16}");
            }
            if let Some(r) = self.tc_min_rows {
                bail!("METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS={r} {W4A16}");
            }
            if let Some(r) = self.dequant_min_rows {
                bail!("METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS={r} {W4A16}");
            }
        }
        if let (false, Some(r)) = (fp8_registry, self.fp8_dequant_min_rows) {
            bail!(
                "METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS={r} needs --dense-quantization fp8 or \
                 w4a16, or --dsa-quantization w8a8"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
impl DenseLevers {
    /// 2026-10-17: The levers of the declared defaults.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
#[path = "glm5next_dense_levers_tests.rs"]
mod tests;
