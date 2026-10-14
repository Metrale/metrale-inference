// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MOE_W4A4_RASTER=<gu>,<down>`: the grouped W4A4 routed experts' m128
//! launches (the fused gate/up `w4a4_grouped_gu_m128` and the down `w4a4_grouped_mx_m128[_k64]`)
//! run their `_r` twins, which walk the CTA grid in bands of `<gu>` / `<down>` slot tiles (all
//! column tiles of a band before the next band) instead of row-major, so a band's expert weight
//! tiles stay in L2 across the column sweep. Only the CTA that computes each (tile, column tile)
//! changes; every output's K chain is the same, so every byte is the plain entry's (the GPU gate
//! is the `glm5next_moe_epi_bench` example).
//!
//! A group of 0 keeps that projection's plain entry; `0` alone keeps both. The lever acts only
//! where its launches run: the grouped W4A4 path (`METRALE_GLM_MOE_W4A4_GROUPED=1`) at the m128
//! tile (launches of 48 or more slots per expert), the gate/up group only on the fused launch
//! (`METRALE_GLM_MOE_W4A4_GU_FUSED=1`); the m32 / m64 tiles and the separate gate and up GEMMs
//! have no `_r` twin and keep their order.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset is [`DEFAULT`]. On the grouped path an active lever needs the three `_r` entries,
//!   else the load is refused ([`check_at_load`]).

use anyhow::Result;
use metrale_gpu_runtime::gpu::KernelHandle;

use crate::glm5next_mlp::Glm5NextMlpKernels;

const LEVER: &str = "METRALE_GLM_MOE_W4A4_RASTER";
/// 2026-10-14: The widest group the lever takes (a whole 8192-row chunk is 800 m128 tiles).
const MAX_GROUP: u32 = 4096;
/// 2026-10-14: The lever's value when unset: off until the c32k A/B arm measures it.
const DEFAULT: Option<RasterGroups> = None;
/// 2026-10-14: The m128 tile entry (`W4A4_GROUPED_MT[2]`), the only one with `_r` twins.
const TILE: usize = 2;

/// 2026-10-14: The raster groups of the fused gate/up and the down launch; 0 = plain entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterGroups {
    pub gu: u32,
    pub down: u32,
}

/// 2026-10-14: A grouped launch's entry: the kernel and, for an `_r` twin, its raster group
/// (the one extra trailing argument).
#[derive(Clone, Copy, Debug)]
pub struct GroupedEntry {
    pub kern: KernelHandle,
    pub raster: Option<u32>,
}

impl GroupedEntry {
    /// 2026-10-14: An entry without a raster argument.
    pub fn plain(kern: KernelHandle) -> Self {
        Self { kern, raster: None }
    }
}

/// 2026-10-14: `METRALE_GLM_MOE_W4A4_RASTER`, read once.
pub fn moe_w4a4_raster() -> Result<Option<RasterGroups>> {
    static V: std::sync::OnceLock<Result<Option<RasterGroups>, String>> =
        std::sync::OnceLock::new();
    V.get_or_init(|| parse_raster(std::env::var(LEVER).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`moe_w4a4_raster`]'s parse, pure: unset is [`DEFAULT`], `0` is off,
/// `<gu>,<down>` with each in 0..=4096 is on; anything else is an error.
pub(crate) fn parse_raster(v: Option<&str>) -> Result<Option<RasterGroups>, String> {
    let s = match v {
        None => return Ok(DEFAULT),
        Some("0") => return Ok(None),
        Some(s) => s,
    };
    let bad = || format!("{LEVER}={s:?}: expected 0 or <gu>,<down> with each in 0..={MAX_GROUP}");
    let (gu, down) = s.split_once(',').ok_or_else(bad)?;
    let group = |t: &str| {
        t.parse::<u32>()
            .ok()
            .filter(|g| *g <= MAX_GROUP)
            .ok_or_else(bad)
    };
    Ok(Some(RasterGroups {
        gu: group(gu)?,
        down: group(down)?,
    }))
}

/// 2026-10-14: Whether the lever can run on this load: `grouped` is
/// `METRALE_GLM_MOE_W4A4_GROUPED` (without it no launch the lever touches runs), `ready` whether
/// every `_r` entry resolved.
pub(crate) fn check_ready(grouped: bool, ready: bool) -> Result<(), String> {
    if grouped && !ready {
        return Err(format!(
            "{LEVER}: w4a4_grouped_mx_m128_r/_m128_k64_r/gu_m128_r is absent from this target's PTX"
        ));
    }
    Ok(())
}

/// 2026-10-14: The load-time refusal of a lever this load cannot run.
pub fn check_at_load(k: &Glm5NextMlpKernels) -> Result<()> {
    if moe_w4a4_raster()?.is_none() {
        return Ok(());
    }
    check_ready(
        super::moe_w4a4_grouped()?,
        k.w4a4_grouped.raster.iter().all(|h| h.0 != 0),
    )
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: The fused gate/up entry of tile `ti`: its `_r` twin at the m128 tile with a
/// non-zero gate/up group, else `plain`.
pub(super) fn gu_entry(
    k: &Glm5NextMlpKernels,
    r: Option<RasterGroups>,
    ti: usize,
    plain: KernelHandle,
) -> GroupedEntry {
    twin(r.map(|r| r.gu), ti, k.w4a4_grouped.raster[2], plain)
}

/// 2026-10-14: The down entry of tile `ti` (`k64`: the `_k64` down): its `_r` twin at the m128
/// tile with a non-zero down group, else `plain`.
pub(super) fn down_entry(
    k: &Glm5NextMlpKernels,
    r: Option<RasterGroups>,
    ti: usize,
    k64: bool,
    plain: KernelHandle,
) -> GroupedEntry {
    let r_kern = k.w4a4_grouped.raster[usize::from(k64)];
    twin(r.map(|r| r.down), ti, r_kern, plain)
}

fn twin(group: Option<u32>, ti: usize, r_kern: KernelHandle, plain: KernelHandle) -> GroupedEntry {
    match group {
        Some(g) if g != 0 && ti == TILE => GroupedEntry {
            kern: r_kern,
            raster: Some(g),
        },
        _ => GroupedEntry::plain(plain),
    }
}
