// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MHC_POST_PRE_TILE=1`: in the SP-mHC layer body
//! (`glm5next_layer/steps/forward_sp.rs`, `sp_mid_piece`) the attention site's reduce + post-mix
//! (`glm5next_hc_post_rank_sum_vec8`) and the FFN site's pre-norm (`glm5next_hc_pre_norm_tile4`)
//! of a row piece run as one launch (`glm5next_mhc_post_tile.cu`). One 256-thread group per token
//! forms the token's four new streams in registers, writes them once, and runs the pre-norm's
//! chains from those registers; only `hc_fn` goes through a cp.async ring. The pair wrote the
//! streams and read them back from DRAM. The ring entry points ([`Entry`],
//! `glm5next_mhc_post_ring.cu`) keep tile4's four-token cp.async ring instead and read back, in
//! the same launch, the streams they have just formed; the layer launches [`WIRED`].
//!
//! Numerics: bit-identical (class B). Each stream element is vec8's expression in its order and
//! each pre-norm output tile4's chains, trees and tails in theirs, so `block_out` (with
//! `write_x`), the streams, `y`, `post`, `comb`, `normed` and the `mix` scratch hold the pair's
//! bytes (`examples/glm5next_hc_post_pre_tile_gate.rs` checks every one against the pair).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off unless the lever is `1`. On, the load fails (`Glm5NextMhcKernels::resolve`) unless
//!   `METRALE_GLM_SP_MHC=1`, `METRALE_GLM_MHC_FUSED=1`, `METRALE_GLM_MHC_PRE_TILE=1` and
//!   `METRALE_GLM_MHC_POST_VEC=1` (the pair it replaces, which narrower pieces keep running),
//!   `METRALE_GLM_MHC_POST_PRE_FUSED` is off, and the build has the kernel: no silent fallback.
//! - Taken per piece from [`MIN_TOKENS_ENV`] rows, a threshold with no default (the lever on
//!   without it fails the load) and no lower than `HC_PRE_NORM_ROWS_MIN_TOKENS`, tile4's floor;
//!   a narrower piece runs the pair. 2026-10-16: tile1 is slower than the pair at 347 rows and
//!   faster at 684 (crossover measured between 390 and 510 rows on GB10), so the recipe sets it.
//! - Hidden [`HIDDEN`], hc [`HC`], at most [`MAX_RANKS`] ranks, BF16 `hc_fn` 16-byte aligned (and
//!   `streams_out` for a ring entry); [`launch`] refuses anything else before launching
//!   (`refusal`).

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{
    GLM5NEXT_MHC_POST_RING_MODULE, GLM5NEXT_MHC_POST_TILE_MODULE, Glm5NextMhcKernels,
    mhc_mix_max_tokens,
};
use crate::glm5next_mhc_fused::HC_PRE_NORM_ROWS_MIN_TOKENS;
use crate::glm5next_mhc_post_pre_tok::PostPreTok;

/// 2026-10-15: The lever's environment variable.
pub const LEVER: &str = "METRALE_GLM_MHC_POST_PRE_TILE";
/// 2026-10-16: The narrowest piece the lever takes, required when it is on.
pub const MIN_TOKENS_ENV: &str = "METRALE_GLM_MHC_POST_PRE_TILE_MIN_TOKENS";
/// 2026-10-15: `GLM_HCPT_H`: the only hidden size the kernels are compiled for.
pub const HIDDEN: u32 = 4096;
/// 2026-10-15: `GLM_HCT_HC`: the only `hc_mult`.
pub const HC: u32 = 4;
/// 2026-10-15: `GLM_HCPT_MAX_RANKS`: the most ranks summed in registers.
pub const MAX_RANKS: usize = 4;
/// 2026-10-15: `GLM_HCPT_STAGES`.
pub const STAGES: u32 = 4;

/// 2026-10-15: The entry points. `Tile1` and `Tile2` keep each token's streams in registers
/// (one or two tokens per block, `glm5next_mhc_post_tile.cu`); the ring entries run tile4's
/// four-token cp.async ring on the streams they have just written (`glm5next_mhc_post_ring.cu`,
/// at most one block per SM, each block looping over its token groups): `Ring0` forms each
/// group's own streams, `Ring1` forms the next group's during the current group's chains, and
/// `Ring1e` is `Ring1` with its partial and residual loads marked L2 evict_first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    Tile1,
    Tile2,
    Ring0,
    Ring1,
    Ring1e,
}

/// 2026-10-15: Every entry point, in [`Entry`]'s order.
pub const ENTRIES: [Entry; 5] = [
    Entry::Tile1,
    Entry::Tile2,
    Entry::Ring0,
    Entry::Ring1,
    Entry::Ring1e,
];

/// 2026-10-15: The entry point the layer launches.
pub const WIRED: Entry = Entry::Tile1;

impl Entry {
    /// 2026-10-15: The kernel's name in its module.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tile1 => "glm5next_hc_post_pre_tile1",
            Self::Tile2 => "glm5next_hc_post_pre_tile2",
            Self::Ring0 => "glm5next_hc_post_pre_ring0",
            Self::Ring1 => "glm5next_hc_post_pre_ring1",
            Self::Ring1e => "glm5next_hc_post_pre_ring1e",
        }
    }

    /// 2026-10-15: The PTX module that holds it.
    pub const fn module(self) -> &'static str {
        if self.is_ring() {
            GLM5NEXT_MHC_POST_RING_MODULE
        } else {
            GLM5NEXT_MHC_POST_TILE_MODULE
        }
    }

    /// 2026-10-15: A ring entry: reads the streams it wrote back through cp.async.
    pub const fn is_ring(self) -> bool {
        matches!(self, Self::Ring0 | Self::Ring1 | Self::Ring1e)
    }

    /// 2026-10-15: Threads per block: 256 per token group.
    pub const fn block(self) -> u32 {
        match self {
            Self::Tile2 => 512,
            _ => 256,
        }
    }

    /// 2026-10-15: Dynamic shared memory per block. Tiles: `GlmHcptLayout<NG>::BYTES`, the
    /// `hc_fn` ring (per stage the 24 BF16 rows of one 256-wide step) or the groups' tree
    /// scratch, whichever is larger, then per group the mixes, the pre weights, the norm's warp
    /// sums and the token's post and comb rows. Rings: `GlmHcprLayout::BYTES`, tile4's layout
    /// rounded up to 16 bytes, then the post and comb rows of four tokens.
    pub const fn smem_bytes(self) -> u32 {
        let mix = (2 + HC) * HC;
        let groups = match self {
            Self::Tile1 => 1,
            Self::Tile2 => 2,
            Self::Ring0 | Self::Ring1 | Self::Ring1e => {
                let tile = crate::glm5next_mhc_tile::tile_smem_bytes();
                return tile.div_ceil(16) * 16 + 4 * (HC + HC * HC) * 4;
            }
        };
        let ring = STAGES * mix * 256 * 2;
        let red = groups * (mix + 1) * 256 * 4;
        let reuse = if ring > red { ring } else { red };
        reuse + groups * (mix + HC + 33 + HC + HC * HC) * 4
    }

    /// 2026-10-15: Blocks for a `num_tokens`-row piece on a GPU of `sm_count` SMs: one per
    /// token (`Tile1`), per two (`Tile2`), or per four-token group up to one per SM (rings).
    pub const fn grid(self, num_tokens: u32, sm_count: u32) -> u32 {
        match self {
            Self::Tile1 => num_tokens,
            Self::Tile2 => num_tokens.div_ceil(2),
            Self::Ring0 | Self::Ring1 | Self::Ring1e => {
                let groups = num_tokens.div_ceil(4);
                if groups < sm_count { groups } else { sm_count }
            }
        }
    }
}

/// 2026-10-15: The other levers the load check reads.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Requires {
    pub sp_mhc: bool,
    pub fused: bool,
    pub pre_tile: bool,
    pub post_vec: bool,
    pub post_pre_fused: bool,
}

/// 2026-10-15: [`enabled`]'s parse, pure: unset or `0` off, `1` on when every lever it needs is
/// on and the `METRALE_GLM_MHC_POST_PRE_FUSED` kernel is not; anything else is an error.
/// 2026-10-16: On, the threshold `min` ([`MIN_TOKENS_ENV`]) is required and returned; set with
/// the lever off it is an error too.
pub(crate) fn parse(
    v: Option<&str>,
    min: Option<&str>,
    r: Requires,
) -> Result<Option<u32>, String> {
    match v {
        None | Some("0") => {
            return match min {
                None => Ok(None),
                Some(m) => Err(format!("{MIN_TOKENS_ENV}={m:?} is set but {LEVER} is off")),
            };
        }
        Some("1") => {}
        Some(s) => return Err(format!("{LEVER}={s:?}: expected 0 (off) or 1 (on)")),
    }
    let missing: Vec<&str> = [
        (r.sp_mhc, "METRALE_GLM_SP_MHC=1"),
        (r.fused, "METRALE_GLM_MHC_FUSED=1"),
        (r.pre_tile, "METRALE_GLM_MHC_PRE_TILE=1"),
        (r.post_vec, "METRALE_GLM_MHC_POST_VEC=1"),
    ]
    .iter()
    .filter(|(set, _)| !set)
    .map(|&(_, name)| name)
    .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{LEVER}=1 needs {} (the pair it replaces on the pieces it takes)",
            missing.join(", ")
        ));
    }
    if r.post_pre_fused {
        return Err(format!(
            "{LEVER}=1 and METRALE_GLM_MHC_POST_PRE_FUSED=1 replace the same pair; set one"
        ));
    }
    let floor = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    let Some(m) = min else {
        return Err(format!(
            "{LEVER}=1 needs {MIN_TOKENS_ENV}=N, the narrowest piece it takes (no default: \
             narrower pieces are faster as the pair)"
        ));
    };
    match m.parse::<u32>() {
        Ok(n) if n >= floor => Ok(Some(n)),
        _ => Err(format!(
            "{MIN_TOKENS_ENV}={m:?}: expected a row count of at least {floor}"
        )),
    }
}

/// 2026-10-15: The lever, read once with the levers it needs, and a one-time readback when on.
/// 2026-10-16: `Some(threshold)` when on, `None` off.
pub fn enabled() -> Result<Option<u32>> {
    static V: std::sync::OnceLock<Result<Option<u32>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let v = std::env::var(LEVER).ok();
        let min = std::env::var(MIN_TOKENS_ENV).ok();
        // 2026-10-15: The other levers are read only when this one is `1`, so a load with it off
        // reads nothing new.
        let s = |e: anyhow::Error| e.to_string();
        let r = if v.as_deref() == Some("1") {
            Requires {
                sp_mhc: crate::glm5next_sp_mhc::enabled().map_err(s)?,
                fused: crate::glm5next_layer::levers::mhc_fused(),
                pre_tile: crate::glm5next_mhc_tile::pre_tile().map_err(s)?,
                post_vec: crate::glm5next_hc_post_vec::enabled().map_err(s)?,
                post_pre_fused: crate::glm5next_mhc_post_pre_tok::enabled().map_err(s)?,
            }
        } else {
            Requires::default()
        };
        let on = parse(v.as_deref(), min.as_deref(), r)?;
        if let Some(n) = on {
            tracing::info!(
                "{LEVER}=1: SP-mHC pieces of {n}+ rows ({MIN_TOKENS_ENV}) run the post-mix and \
                 the FFN pre-norm as one launch ({}, {} B shared; same bytes as vec8 + tile4)",
                WIRED.name(),
                WIRED.smem_bytes()
            );
        }
        Ok(on)
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-15: The kernel handle for `Glm5NextMhcKernels::resolve` (0 when the build lacks it),
/// after the load check: the lever on with the handle 0 is an error.
pub fn resolve(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let k = metrale_model_layers::layers::try_kernel(gpu, WIRED.module(), WIRED.name());
    load_check(enabled()?.is_some(), k)?;
    Ok(k)
}

/// 2026-10-15: On without the kernel in the build fails the load.
pub(crate) fn load_check(on: bool, kernel: KernelHandle) -> Result<()> {
    ensure!(
        !on || kernel.0 != 0,
        "{LEVER}=1 needs {} on this target",
        WIRED.name()
    );
    Ok(())
}

/// 2026-10-15: The fused kernel for a `num_tokens`-row piece: `None` off or narrower than
/// `HC_PRE_NORM_ROWS_MIN_TOKENS` (the caller runs the pair).
/// 2026-10-16: `min_tokens` is [`enabled`]'s: `None` off, else the narrowest piece taken.
pub fn kernel_for(
    kernels: &Glm5NextMhcKernels,
    min_tokens: Option<u32>,
    num_tokens: u32,
) -> Result<Option<KernelHandle>> {
    let Some(min) = min_tokens else {
        return Ok(None);
    };
    if num_tokens < min || (num_tokens as usize) < HC_PRE_NORM_ROWS_MIN_TOKENS {
        return Ok(None);
    }
    load_check(true, kernels.hc_post_pre_tile)?;
    Ok(Some(kernels.hc_post_pre_tile))
}

/// 2026-10-15: What a launch is checked on, pure.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub hidden: u32,
    pub hc_mult: u32,
    pub hc_fn_bf16: bool,
    pub hc_fn: u64,
    pub ring: bool,
    pub streams_out: u64,
    pub num_tokens: u32,
    pub mix_cap: usize,
    pub slot_elems: usize,
    pub world_size: usize,
    pub rank: usize,
}

/// 2026-10-15: Why the kernel cannot take `s`, or `None`.
pub(crate) fn refusal(s: &Shape) -> Option<String> {
    if s.hidden != HIDDEN || s.hc_mult != HC {
        return Some(format!(
            "hidden {} and hc {}: the kernel is built for hidden {HIDDEN}, hc {HC}",
            s.hidden, s.hc_mult
        ));
    }
    if !s.hc_fn_bf16 || !s.hc_fn.is_multiple_of(16) {
        return Some("hc_fn must be BF16 and 16-byte aligned".into());
    }
    if s.ring && !s.streams_out.is_multiple_of(16) {
        return Some("the ring reads the streams back by cp.async: 16-byte aligned".into());
    }
    if s.num_tokens == 0 || s.num_tokens as usize > s.mix_cap {
        return Some(format!(
            "{} tokens: 1 to the {}-token mix scratch",
            s.num_tokens, s.mix_cap
        ));
    }
    let elems = s.num_tokens as usize * s.hidden as usize;
    if elems > s.slot_elems {
        return Some(format!(
            "{elems} elements exceed the peer slot of {}",
            s.slot_elems
        ));
    }
    if !(1..=MAX_RANKS).contains(&s.world_size) || s.rank >= s.world_size {
        return Some(format!(
            "rank {} of {} ranks (1 to {MAX_RANKS})",
            s.rank, s.world_size
        ));
    }
    None
}

/// 2026-10-15: One launch of `kernel` (the `entry` point) on [`Entry::grid`] blocks. `write_x`
/// stores the summed row to `block_out` (vec8's flag). Refused before launching when
/// [`refusal`] names a reason.
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    entry: Entry,
    a: &PostPreTok,
    write_x: bool,
    stream: u64,
) -> Result<()> {
    let [hidden, hc_mult, sinkhorn_iters] = a.dims;
    let [norm_eps, hc_eps, rms_eps] = a.eps;
    let name = entry.name();
    ensure!(kernel.0 != 0, "{name} is not loaded on this target");
    let shape = Shape {
        hidden,
        hc_mult,
        hc_fn_bf16: a.site.hc_fn_bf16,
        hc_fn: a.site.hc_fn.0,
        ring: entry.is_ring(),
        streams_out: a.streams_out.0,
        num_tokens: a.num_tokens,
        mix_cap: mhc_mix_max_tokens(),
        slot_elems: a.partials.slot_elems,
        world_size: a.partials.world_size,
        rank: a.partials.rank,
    };
    if let Some(why) = refusal(&shape) {
        bail!("{name}: {why}");
    }
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!("{name} launched: {} rows, write_x={write_x}", a.num_tokens);
    }
    // 2026-10-15: Only the rings size their grid by the SM count.
    let sms = if entry.is_ring() { gpu.sm_count()? } else { 0 };
    ensure!(
        !entry.is_ring() || sms > 0,
        "{name}: the GPU reports no SMs"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([entry.grid(a.num_tokens, sms), 1, 1])
        .block([entry.block(), 1, 1])
        .shared_mem(entry.smem_bytes())
        .arg_ptr(a.block_out)
        .arg_ptr(DevicePtr(a.partials.recv))
        .arg_u64(a.partials.slot_elems as u64)
        .arg_i32(a.partials.world_size as i32)
        .arg_i32(a.partials.rank as i32)
        .arg_ptr(a.residual)
        .arg_ptr(a.post)
        .arg_ptr(a.comb)
        .arg_ptr(a.streams_out)
        .arg_i32(i32::from(write_x))
        .arg_ptr(a.site.hc_fn)
        .arg_ptr(a.site.hc_scale)
        .arg_ptr(a.site.hc_base)
        .arg_ptr(a.site.mix)
        .arg_ptr(a.norm_w)
        .arg_ptr(a.y)
        .arg_ptr(a.post_out)
        .arg_ptr(a.comb_out)
        .arg_ptr(a.normed)
        .arg_u32(hidden)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .arg_f32(rms_eps)
        .arg_u32(a.num_tokens)
        .launch(stream)
}

#[cfg(test)]
#[path = "glm5next_mhc_post_tile_tests.rs"]
mod tests;
