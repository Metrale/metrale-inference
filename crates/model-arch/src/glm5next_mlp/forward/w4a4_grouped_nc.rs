// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MOE_W4A4_NC=<nc>`: the grouped W4A4 routed experts' m128 down launch
//! runs `w4a4_grouped_mx_m128[_k64]_n`, where one CTA computes `nc` adjacent column tiles of its
//! slot tile as one continuous load stream instead of one column tile per CTA, so the CTA's
//! prologue (expert lookup, pointers, slot rows and globals) and pipeline fill are paid once per
//! `nc` tiles, and each finished tile is stored straight from registers while the next tile's
//! chunks load. Every output's K chain, fold order and rounding are the plain entry's, so every
//! byte is the plain entry's (the GPU gate is the `glm5next_moe_epi_bench` example). The down
//! raster group (`METRALE_GLM_MOE_W4A4_RASTER`) walks the narrower grid the same way.
//!
//! The lever acts only where its launch runs: the grouped W4A4 path
//! (`METRALE_GLM_MOE_W4A4_GROUPED=1`) at the m128 tile; the m32 / m64 tiles, the fused gate/up
//! and the separate gate and up GEMMs keep their entries.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset or `0` is off. On the grouped path an active lever needs both `_n` entries, else the
//!   load is refused ([`check_at_load`]).

use anyhow::Result;

use super::w4a4_grouped_raster::GroupedEntry;
use crate::glm5next_mlp::Glm5NextMlpKernels;

const LEVER: &str = "METRALE_GLM_MOE_W4A4_NC";
/// 2026-10-15: The most column tiles one CTA takes (GLM-5.3's hidden 4096 is 64 tiles of 64).
const MAX_NC: u32 = 4096;
/// 2026-10-15: The m128 tile entry (`W4A4_GROUPED_MT[2]`), the only one with `_n` twins.
const TILE: usize = 2;

/// 2026-10-15: `METRALE_GLM_MOE_W4A4_NC`, read once.
pub fn moe_w4a4_nc() -> Result<Option<u32>> {
    static V: std::sync::OnceLock<Result<Option<u32>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_nc(std::env::var(LEVER).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-15: [`moe_w4a4_nc`]'s parse, pure: unset or `0` is off, `1..=4096` is on; anything
/// else is an error.
pub(crate) fn parse_nc(v: Option<&str>) -> Result<Option<u32>, String> {
    match v {
        None | Some("0") => Ok(None),
        Some(s) => s
            .parse::<u32>()
            .ok()
            .filter(|n| (1..=MAX_NC).contains(n))
            .map(Some)
            .ok_or_else(|| {
                format!("{LEVER}={s:?}: expected 0 or a column tile count 1..={MAX_NC}")
            }),
    }
}

/// 2026-10-15: Whether the lever can run on this load: `grouped` is
/// `METRALE_GLM_MOE_W4A4_GROUPED`, `ready` whether both `_n` entries resolved.
pub(crate) fn check_ready(grouped: bool, ready: bool) -> Result<(), String> {
    if grouped && !ready {
        return Err(format!(
            "{LEVER}: w4a4_grouped_mx_m128_n/_m128_k64_n is absent from this target's PTX"
        ));
    }
    Ok(())
}

/// 2026-10-15: The lever's load readback, logged once per process on every rank.
pub(crate) fn readback(nc: Option<u32>, grouped: bool) -> String {
    match (nc, grouped) {
        (Some(n), true) => format!("{LEVER}: active, {n} column tiles per m128 down CTA"),
        (Some(n), false) => format!(
            "{LEVER}: dormant (METRALE_GLM_MOE_W4A4_GROUPED off), {n} column tiles per m128 down CTA"
        ),
        (None, _) => format!("{LEVER}: off"),
    }
}

/// 2026-10-15: The load-time readback and refusal of a lever this load cannot run.
pub fn check_at_load(k: &Glm5NextMlpKernels) -> Result<()> {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    let nc = moe_w4a4_nc()?;
    let grouped = super::moe_w4a4_grouped()?;
    LOGGED.call_once(|| tracing::info!("{}", readback(nc, grouped)));
    if nc.is_none() {
        return Ok(());
    }
    check_ready(grouped, k.w4a4_grouped.nc.iter().all(|h| h.0 != 0)).map_err(anyhow::Error::msg)
}

/// 2026-10-15: The down entry of tile `ti` (`k64`: the `_k64` down) under the lever: at the m128
/// tile its `_n` twin with `nc` and the raster group `down` already carries (0 when plain), else
/// `down` unchanged.
pub(super) fn down_entry(
    k: &Glm5NextMlpKernels,
    nc: Option<u32>,
    ti: usize,
    k64: bool,
    down: GroupedEntry,
) -> GroupedEntry {
    match nc {
        Some(n) if ti == TILE => GroupedEntry {
            kern: k.w4a4_grouped.nc[usize::from(k64)],
            raster: Some(down.raster.unwrap_or(0)),
            nc: Some(n),
        },
        _ => down,
    }
}
