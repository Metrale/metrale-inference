// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MHC_POST_VEC=1`: the SP-mHC prefill post-mix (the reduce step of the
//! one-shot exchange plus `glm_hc_post`, `glm5next_layer/steps/forward_sp.rs`) runs
//! `glm5next_hc_post_rank_sum_vec8` instead of `glm5next_hc_post_rank_sum`: eight columns per
//! thread in 16-byte accesses, 128-thread blocks, and the summed row written back to `block_out`
//! only when the caller asks (`write_x`).
//!
//! Numerics: bit-identical (class B). Every stream element is the scalar kernel's expression in
//! its order (`glm5next_hc_post_sum.cu`), and with `write_x` the summed row is its bytes too
//! (`examples/glm5next_hc_post_sum_gate.rs` checks both against the scalar kernel).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off unless `MhcLevers::post_vec` (resolved once at startup, `glm5next_mhc_levers`).
//! - On, a build without the kernel fails the first post-mix (no silent fallback).
//! - `write_x = false` leaves `block_out` holding this rank's partial. The SP prefill body reads
//!   the summed row after the post-mix only through a dump tap, so its callers pass
//!   `tap.is_some()`; the decode's `AR_FUSED_POST` path does not come here.
//! - `METRALE_GLM_MHC_POST_SUM_CTAS` caps only the scalar kernel's grid.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, ensure};
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{GLM5NEXT_HC_POST_SUM_MODULE, Glm5NextMhcKernels};
use crate::glm5next_mhc_fused::glm_hc_post_rank_sum;

/// 2026-10-15: The lever's environment variable.
pub const LEVER: &str = "METRALE_GLM_MHC_POST_VEC";
/// 2026-10-15: The vectorized kernel's entry point in [`GLM5NEXT_HC_POST_SUM_MODULE`].
pub const KERNEL: &str = "glm5next_hc_post_rank_sum_vec8";
/// 2026-10-15: Threads per block: at 4096 columns a row is four blocks, so an 86-row piece (a
/// 1k prefill over three ranks, four pieces) still launches 344 blocks.
pub const BLOCK: u32 = 128;
/// 2026-10-15: Columns per thread.
pub const VEC: u32 = 8;
/// 2026-10-15: The most ranks the kernel holds in registers (`GLM_HCPS_MAX_RANKS`).
pub const MAX_RANKS: usize = 8;
/// 2026-10-15: The most residual streams the kernel mixes (`GLM_HCPS_MAX_MULT`).
pub const MAX_HC: u32 = 4;

/// 2026-10-15: The optional kernel handle (0 when the build lacks it), for
/// `Glm5NextMhcKernels::resolve`.
pub fn resolve(gpu: &dyn GpuBackend) -> KernelHandle {
    metrale_model_layers::layers::try_kernel(gpu, GLM5NEXT_HC_POST_SUM_MODULE, KERNEL)
}

/// 2026-10-15: The kernel to launch: `None` off, the vectorized kernel on; on without it in the
/// build is an error.
pub fn kernel_for(kernels: &Glm5NextMhcKernels, on: bool) -> Result<Option<KernelHandle>> {
    if !on {
        return Ok(None);
    }
    ensure!(
        kernels.hc_post_rank_sum_vec8.0 != 0,
        "{LEVER}=1 needs {KERNEL} on this target"
    );
    Ok(Some(kernels.hc_post_rank_sum_vec8))
}

/// 2026-10-15: `(grid, block)` of a `num_tokens`-row launch at `hidden` columns: one block row
/// per token, each block [`BLOCK`] threads of [`VEC`] columns.
pub fn launch_shape(num_tokens: u32, hidden: u32) -> ([u32; 3], [u32; 3]) {
    let per_block = BLOCK * VEC;
    (
        [num_tokens.max(1), hidden.div_ceil(per_block).max(1), 1],
        [BLOCK, 1, 1],
    )
}

/// 2026-10-15: The SP-mHC post-mix of `num_tokens` rows: the vectorized kernel under the lever,
/// else `glm_hc_post_rank_sum` (which always writes the summed row back). `streams` is
/// `[residual, post, comb, out]`, `dims` `[num_tokens, hidden_size, hc_mult]`.
#[allow(clippy::too_many_arguments)]
pub fn post_rank_sum(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    block_out: DevicePtr,
    partials: &PeerPartials,
    streams: [DevicePtr; 4],
    dims: [u32; 3],
    write_x: bool,
    stream: u64,
) -> Result<()> {
    match kernel_for(kernels, kernels.levers.post_vec)? {
        Some(kernel) => launch(
            gpu, kernel, block_out, partials, streams, dims, write_x, stream,
        ),
        None => {
            let [residual, post, comb, out] = streams;
            let [num_tokens, hidden_size, hc_mult] = dims;
            glm_hc_post_rank_sum(
                gpu,
                kernels,
                block_out,
                partials,
                residual,
                post,
                comb,
                out,
                num_tokens,
                hidden_size,
                hc_mult,
                stream,
            )
        }
    }
}

/// 2026-10-15: One `glm5next_hc_post_rank_sum_vec8` launch. Refuses, before launching, a zero
/// handle, rows past a peer slot, and any shape the kernel's 16-byte accesses cannot take.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    partials: &PeerPartials,
    [residual, post, comb, out]: [DevicePtr; 4],
    [num_tokens, hidden_size, hc_mult]: [u32; 3],
    write_x: bool,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "{KERNEL} is not loaded on this target");
    let elems = num_tokens as usize * hidden_size as usize;
    ensure!(
        elems <= partials.slot_elems,
        "{elems} elements exceed the one-shot peer slot of {}",
        partials.slot_elems
    );
    ensure!(
        hidden_size.is_multiple_of(VEC) && partials.slot_elems.is_multiple_of(VEC as usize),
        "{KERNEL}: hidden {hidden_size} and peer slot {} must be multiples of {VEC}",
        partials.slot_elems
    );
    ensure!(
        (1..=MAX_RANKS).contains(&partials.world_size) && (1..=MAX_HC).contains(&hc_mult),
        "{KERNEL}: {} ranks (max {MAX_RANKS}) and hc {hc_mult} (max {MAX_HC})",
        partials.world_size
    );
    ensure!(
        [block_out.0, partials.recv, residual.0, out.0]
            .iter()
            .all(|p| p % 16 == 0),
        "{KERNEL}: block_out, the peer slots, residual and out must be 16-byte aligned"
    );
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!("{KERNEL} launched: {num_tokens} rows, write_x={write_x}");
    }
    let (grid, block) = launch_shape(num_tokens, hidden_size);
    KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
        .arg_ptr(block_out)
        .arg_ptr(DevicePtr(partials.recv))
        .arg_u64(partials.slot_elems as u64)
        .arg_i32(partials.world_size as i32)
        .arg_i32(partials.rank as i32)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(num_tokens)
        .arg_i32(i32::from(write_x))
        .launch(stream)
}

#[cfg(test)]
#[path = "glm5next_hc_post_vec_tests.rs"]
mod tests;
