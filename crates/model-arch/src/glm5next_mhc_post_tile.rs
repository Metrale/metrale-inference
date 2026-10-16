// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MHC_POST_PRE_TILE=1`: in the SP-mHC layer body
//! (`glm5next_layer/steps/forward_sp.rs`, `sp_mid_piece`) the attention site's reduce + post-mix
//! (`glm5next_hc_post_rank_sum_vec8`) and the FFN site's pre-norm (`glm5next_hc_pre_norm_tile4`)
//! of a row piece run as one launch (`glm5next_mhc_post_tile.cu`). One 256-thread group per token
//! forms the token's four new streams in registers, writes them once, and runs the pre-norm's
//! chains from those registers; only `hc_fn` goes through a cp.async ring. The pair wrote the
//! streams and read them back from DRAM.
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
//! - Taken per piece from `HC_PRE_NORM_ROWS_MIN_TOKENS` rows, as tile4 is.
//! - Hidden [`HIDDEN`], hc [`HC`], at most [`MAX_RANKS`] ranks, BF16 `hc_fn` 16-byte aligned;
//!   [`launch`] refuses anything else before launching (`refusal`).

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{GLM5NEXT_MHC_POST_TILE_MODULE, Glm5NextMhcKernels, mhc_mix_max_tokens};
use crate::glm5next_mhc_fused::HC_PRE_NORM_ROWS_MIN_TOKENS;
use crate::glm5next_mhc_post_pre_tok::PostPreTok;

/// 2026-10-15: The lever's environment variable.
pub const LEVER: &str = "METRALE_GLM_MHC_POST_PRE_TILE";
/// 2026-10-15: The entry points by `NG` (tokens per block, one per 256-thread group).
pub const KERNELS: [&str; 2] = ["glm5next_hc_post_pre_tile1", "glm5next_hc_post_pre_tile2"];
/// 2026-10-15: The `NG` the layer launches.
pub const GROUPS: u32 = 1;
/// 2026-10-15: `GLM_HCPT_H`: the only hidden size the kernel is compiled for.
pub const HIDDEN: u32 = 4096;
/// 2026-10-15: `GLM_HCT_HC`: the only `hc_mult`.
pub const HC: u32 = 4;
/// 2026-10-15: `GLM_HCPT_MAX_RANKS`: the most ranks summed in registers.
pub const MAX_RANKS: usize = 4;
/// 2026-10-15: `GLM_HCPT_STAGES`.
pub const STAGES: u32 = 4;

/// 2026-10-15: `GlmHcptLayout<NG>::BYTES`: the `hc_fn` ring (per stage the 24 BF16 rows of one
/// 256-wide step) or the groups' tree scratch, whichever is larger; then per group the mixes,
/// the pre weights, the norm's warp sums, and the token's post and comb rows.
pub const fn smem_bytes(groups: u32) -> u32 {
    let mix = (2 + HC) * HC;
    let ring = STAGES * mix * 256 * 2;
    let red = groups * (mix + 1) * 256 * 4;
    let reuse = if ring > red { ring } else { red };
    reuse + groups * (mix + HC + 33 + HC + HC * HC) * 4
}

/// 2026-10-15: The entry point for `groups` tokens per block.
pub fn kernel_name(groups: u32) -> Result<&'static str> {
    match groups {
        1 | 2 => Ok(KERNELS[groups as usize - 1]),
        g => bail!("{LEVER}: no entry point for {g} tokens per block"),
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
pub(crate) fn parse(v: Option<&str>, r: Requires) -> Result<bool, String> {
    let on = match v {
        None | Some("0") => return Ok(false),
        Some("1") => true,
        Some(s) => return Err(format!("{LEVER}={s:?}: expected 0 (off) or 1 (on)")),
    };
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
    Ok(on)
}

/// 2026-10-15: The lever, read once with the levers it needs, and a one-time readback when on.
pub fn enabled() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let v = std::env::var(LEVER).ok();
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
        let on = parse(v.as_deref(), r)?;
        if on {
            tracing::info!(
                "{LEVER}=1: SP-mHC pieces of {HC_PRE_NORM_ROWS_MIN_TOKENS}+ rows run the post-mix \
                 and the FFN pre-norm as one launch ({}, {GROUPS} token(s) per block, streams in \
                 registers, {} B shared; same bytes as vec8 + tile4)",
                KERNELS[GROUPS as usize - 1],
                smem_bytes(GROUPS)
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
    let k = metrale_model_layers::layers::try_kernel(
        gpu,
        GLM5NEXT_MHC_POST_TILE_MODULE,
        kernel_name(GROUPS)?,
    );
    load_check(enabled()?, k)?;
    Ok(k)
}

/// 2026-10-15: On without the kernel in the build fails the load.
pub(crate) fn load_check(on: bool, kernel: KernelHandle) -> Result<()> {
    ensure!(
        !on || kernel.0 != 0,
        "{LEVER}=1 needs {} on this target",
        KERNELS[GROUPS as usize - 1]
    );
    Ok(())
}

/// 2026-10-15: The fused kernel for a `num_tokens`-row piece: `None` off or narrower than
/// `HC_PRE_NORM_ROWS_MIN_TOKENS` (the caller runs the pair).
pub fn kernel_for(
    kernels: &Glm5NextMhcKernels,
    on: bool,
    num_tokens: u32,
) -> Result<Option<KernelHandle>> {
    if !on || (num_tokens as usize) < HC_PRE_NORM_ROWS_MIN_TOKENS {
        return Ok(None);
    }
    load_check(on, kernels.hc_post_pre_tile)?;
    Ok(Some(kernels.hc_post_pre_tile))
}

/// 2026-10-15: What a launch is checked on, pure.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub hidden: u32,
    pub hc_mult: u32,
    pub hc_fn_bf16: bool,
    pub hc_fn: u64,
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

/// 2026-10-15: One launch of `kernel` (the `groups`-token entry) on `ceil(num_tokens / groups)`
/// blocks. `write_x` stores the summed row to `block_out` (vec8's flag). Refused before
/// launching when [`refusal`] names a reason.
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    groups: u32,
    a: &PostPreTok,
    write_x: bool,
    stream: u64,
) -> Result<()> {
    let [hidden, hc_mult, sinkhorn_iters] = a.dims;
    let [norm_eps, hc_eps, rms_eps] = a.eps;
    let name = kernel_name(groups)?;
    ensure!(kernel.0 != 0, "{name} is not loaded on this target");
    let shape = Shape {
        hidden,
        hc_mult,
        hc_fn_bf16: a.site.hc_fn_bf16,
        hc_fn: a.site.hc_fn.0,
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
    KernelLaunch::new(gpu, kernel)
        .grid([a.num_tokens.div_ceil(groups), 1, 1])
        .block([256 * groups, 1, 1])
        .shared_mem(smem_bytes(groups))
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
