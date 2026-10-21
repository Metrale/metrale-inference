// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The GLM MLP's routed-expert and dense launch levers, resolved once from the
//! startup settings ([`MlpLevers::from_settings`]) as part of `GlmLevers` (`levers.mlp`). The
//! MLP kernels carry the copy (`Glm5NextMlpKernels::levers`) and the forward reads its fields;
//! nothing here reads the process environment.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A value a reader refused at its first read is refused here, at startup; so are a negative
//!   width and an unknown grouped-GEMM tile, which the old readers ran as the default.
//! - `METRALE_GLM_MOE_GEMM_TILE` and `METRALE_GLM_MOE_W4A4_RASTER` keep their readers' text
//!   predicates (`select_gemm_tile`, `parse_raster`).

use anyhow::{Result, anyhow};
use metrale_settings::InferenceServerSettings;

use super::super::forward_prefill_gemm::{
    DEFAULT_GEMM_MIN_ROWS, DEFAULT_GEMM_TILE, GEMM_TILES, GemmTile, select_gemm_tile,
};
use super::MOE_ROW_BATCH_MAX_ROWS;
use super::shared_fork::SHARED_FORK_MAX_ROWS;
use super::w4a4_dense_grouped::MIN_ROWS_FLOOR;
use super::w4a4_grouped_nc::MAX_NC;
use super::w4a4_grouped_raster::{RasterGroups, parse_raster};

/// 2026-10-17: The MLP levers. Each field names its variable; the module that runs the lever
/// documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlpLevers {
    /// 2026-09-25: `METRALE_NO_GLM_MOE_ROW_BATCH`: the row-batched routed path off; each row
    /// then runs its own expert launches.
    pub row_batch_off: bool,
    /// 2026-09-25: `METRALE_GLM_MOE_ROW_BATCH_MAX`: the widest sub-group of the row-batched
    /// routed path, `MOE_ROW_BATCH_MAX_ROWS` when unset, clamped to
    /// `1..=MOE_ROW_BATCH_MAX_ROWS`.
    pub row_batch_max: usize,
    /// 2026-09-25: `METRALE_GLM_MOE_HOST_DISPATCH`: the host-dispatch expert loop (ids read back
    /// to the host, one GEMV per local expert); every device-dispatched routed path off.
    pub host_dispatch: bool,
    /// 2026-09-25: `METRALE_GLM_MOE_GEMM_TILE` (`forward_prefill_gemm`): the routed-expert
    /// prefill's grouped-GEMM tile, `DEFAULT_GEMM_TILE` when unset.
    pub(crate) gemm_tile: GemmTile,
    /// 2026-09-25: `METRALE_GLM_MOE_PREFILL_GEMM` (on unless switched off): the routed-expert
    /// prefill may take the grouped GEMM.
    pub prefill_gemm: bool,
    /// 2026-09-25: `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS`: the narrowest row group the grouped
    /// GEMM takes, `DEFAULT_GEMM_MIN_ROWS` when unset, at least 1.
    pub prefill_gemm_min_rows: usize,
    /// 2026-09-25: `METRALE_GLM_MOE_PREFILL_GEMM_EXACT_TILES` (on unless switched off): the
    /// grouped GEMM's grid height from the real expert histogram, one stream-synchronising read
    /// of `expert_offsets` per call, rather than the worst case.
    pub exact_tiles: bool,
    /// 2026-10-12: `METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS` (`w4a4_dense_grouped`): `None` off,
    /// else the narrowest dense launch that takes the one-group GEMM.
    pub dense_grouped_min_rows: Option<usize>,
    /// 2026-10-11: `METRALE_GLM_SHARED_FORK` (`shared_fork`): 0 off, else the widest launch
    /// whose shared expert forks.
    pub shared_fork: usize,
    /// 2026-10-10: `METRALE_GLM_MOE_W4A4_GROUPED` (`w4a4_grouped`).
    pub w4a4_grouped: bool,
    /// 2026-10-14: `METRALE_GLM_MOE_W4A4_RASTER` (`w4a4_grouped_raster`): `None` off.
    pub raster: Option<RasterGroups>,
    /// 2026-10-14: `METRALE_GLM_MOE_W4A4_GU_FUSED` (`w4a4_grouped_gu`).
    pub gu_fused: bool,
    /// 2026-10-15: `METRALE_GLM_MOE_W4A4_NC` (`w4a4_grouped_nc`): `None` off.
    pub nc: Option<u32>,
    /// 2026-10-21: `METRALE_GLM_MOE_SWEEP_SERIAL` (on unless switched off,
    /// `sweep_entries`).
    pub sweep_serial: bool,
    /// 2026-10-21: `METRALE_GLM_MOE_ZERO_IN_SWIGLU` (on unless switched off,
    /// `w4a4::zeroes_in_swiglu`).
    pub zero_in_swiglu: bool,
}

/// 2026-10-17: A row width: unset is `default`; a negative value is an error.
fn width(env: &str, v: Option<i64>, default: usize) -> Result<usize> {
    v.map_or(Ok(default), |n| {
        usize::try_from(n).map_err(|_| anyhow!("{env}={n}: expected a row count"))
    })
}

/// 2026-10-17: A count lever: unset or `0` off, else a value `ok` accepts; another value is an
/// error that names what was `expected`.
pub(crate) fn count(
    env: &str,
    v: Option<i64>,
    ok: impl Fn(usize) -> bool,
    expected: &str,
) -> Result<Option<usize>> {
    match v {
        None | Some(0) => Ok(None),
        Some(n) => usize::try_from(n)
            .ok()
            .filter(|r| ok(*r))
            .map(Some)
            .ok_or_else(|| anyhow!("{env}={n}: expected 0 or {expected}")),
    }
}

/// 2026-10-17: The grouped-GEMM tile: unset is `DEFAULT_GEMM_TILE`; a name `select_gemm_tile`
/// does not know is an error.
fn gemm_tile(v: Option<&str>) -> Result<GemmTile> {
    let Some(v) = v else {
        return Ok(DEFAULT_GEMM_TILE);
    };
    select_gemm_tile(v).ok_or_else(|| {
        anyhow!(
            "METRALE_GLM_MOE_GEMM_TILE=`{v}` is not a known tile. Known: {:?}",
            GEMM_TILES.iter().map(|t| t.name).collect::<Vec<_>>()
        )
    })
}

impl MlpLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing the values in the module
    /// invariants.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let (k, sc) = (&s.kernels, &s.scheduler);
        let row_batch_max = width(
            "METRALE_GLM_MOE_ROW_BATCH_MAX",
            sc.glm_moe_row_batch_max,
            MOE_ROW_BATCH_MAX_ROWS,
        )?;
        let min_rows = width(
            "METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS",
            sc.glm_moe_prefill_gemm_min_rows,
            DEFAULT_GEMM_MIN_ROWS,
        )?;
        let fork = count(
            "METRALE_GLM_SHARED_FORK",
            k.glm_shared_fork,
            |r| r <= SHARED_FORK_MAX_ROWS,
            &format!("1..={SHARED_FORK_MAX_ROWS}"),
        )?;
        let nc = count(
            "METRALE_GLM_MOE_W4A4_NC",
            k.glm_moe_w4a4_nc,
            |n| n <= MAX_NC as usize,
            &format!("a column tile count 1..={MAX_NC}"),
        )?;
        Ok(Self {
            row_batch_off: sc.no_glm_moe_row_batch,
            row_batch_max: row_batch_max.clamp(1, MOE_ROW_BATCH_MAX_ROWS),
            host_dispatch: k.glm_moe_host_dispatch,
            gemm_tile: gemm_tile(k.glm_moe_gemm_tile.as_deref())?,
            prefill_gemm: sc.glm_moe_prefill_gemm,
            prefill_gemm_min_rows: min_rows.max(1),
            exact_tiles: sc.glm_moe_prefill_gemm_exact_tiles,
            dense_grouped_min_rows: count(
                "METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS",
                k.glm_dense_w4a4_grouped_min_rows,
                |r| r >= MIN_ROWS_FLOOR,
                &format!("a row count of at least {MIN_ROWS_FLOOR}"),
            )?,
            shared_fork: fork.unwrap_or(0),
            w4a4_grouped: k.glm_moe_w4a4_grouped,
            raster: parse_raster(&k.glm_moe_w4a4_raster).map_err(anyhow::Error::msg)?,
            gu_fused: k.glm_moe_w4a4_gu_fused,
            nc: nc.map(|n| n as u32),
            sweep_serial: k.glm_moe_sweep_serial,
            zero_in_swiglu: k.glm_moe_zero_in_swiglu,
        })
    }

    /// 2026-10-17: The log lines the readers used to print at their first read, printed once by
    /// the text-stack loader (`GlmLevers::announce`).
    pub(crate) fn announce(&self) {
        let m = self.row_batch_max;
        if m != MOE_ROW_BATCH_MAX_ROWS {
            tracing::warn!(
                "GLM MoE row-batch width capped at {m} (default {MOE_ROW_BATCH_MAX_ROWS})"
            );
        }
        self.announce_prefill_gemm();
    }
}

#[cfg(test)]
impl MlpLevers {
    /// 2026-10-17: The levers of the declared defaults.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
