// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_DSA_SCORES_PIPE`: `dsa_index_scores_mma`'s scores and candidacy,
//! byte for byte, from per-pool BF16 keys and visibility positions written once per pass after
//! `dsa_kpool_compress` (`dsa_pool_prep_bf16`), with the key tiles streamed through a 3-stage
//! cp.async ring (`dsa_index_scores_mma_pipe`, `dsa_index_scores_pipe.cu`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The pipe runs only where the tensor-core scores run ([`scores_mma_for`]) and replaces that
//!   launch; every other kernel of the pass is unchanged.
//! - [`DsaPipeRegions`] are allocated only when the lever is on at allocation; a pass that takes
//!   the pipe over more pools than they hold fails before any launch.

use super::*;

/// 2026-10-14: `DSA_SP_STAGES` and the dynamic shared memory of `dsa_index_scores_mma_pipe`:
/// per stage one 64-pool BF16 key tile at a row stride of `D + 8` halves and 64 i32 positions.
pub const SCORES_PIPE_STAGES: usize = 3;
pub const SCORES_PIPE_SMEM: u32 =
    (SCORES_PIPE_STAGES * (64 * (SCORES_MMA_D + 8) * 2 + 64 * 4)) as u32;

/// 2026-10-14: Pools per block of `dsa_pool_prep_bf16` (32 threads of 4 channels per pool).
const PREP_POOLS_PER_BLOCK: usize = 8;

/// 2026-10-14: `METRALE_GLM_DSA_SCORES_PIPE=1`, read once: where the tensor-core scores run
/// (`METRALE_GLM_DSA_SCORES_MMA`), run them streamed. Off unless `1`.
pub fn scores_pipe() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_DSA_SCORES_PIPE").as_deref() == Ok("1"))
}

/// 2026-10-15: The readback of the prefill selection levers as the first exact multi-row pass
/// resolved them (lever set AND kernel resolved AND shape eligible), logged once per process so
/// every rank's serve log shows which scores path its prefill ran.
/// 2026-10-17: `radix` is `METRALE_GLM_DSA_TOPK_RADIX` as set; a fused pass selects in its own
/// kernel, so the line says when the radix lever is set but unused there.
pub fn readback_select_levers(mma: bool, pipe: bool, fused: bool, radix: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let note = if fused && radix {
            " (topk_radix unused: the fused pass selects)"
        } else {
            ""
        };
        tracing::info!(
            "DSA select levers: scores_mma={} scores_pipe={} select_fused={} topk_radix={}{}",
            u8::from(mma),
            u8::from(pipe),
            u8::from(fused),
            u8::from(radix),
            note
        )
    });
}

/// 2026-10-14: Whether a pass that takes the tensor-core scores (`mma`) streams them: the lever
/// on and both kernels resolved.
pub fn scores_pipe_for(mma: bool, kernels: &super::super::DsaRowsKernels, lever: bool) -> bool {
    lever && mma && kernels.pool_prep.0 != 0 && kernels.scores_pipe.0 != 0
}

/// 2026-10-14: `[pools, D]` BF16 keys and `[pools]` i32 visibility positions; empty (no
/// allocation) when the pipe is off.
#[derive(Debug, Clone, Copy)]
pub struct DsaPipeRegions {
    pub(super) keys_bf16: DevicePtr,
    pub(super) vis: DevicePtr,
    pools: usize,
}

impl DsaPipeRegions {
    /// 2026-10-14: Bytes the regions take for `pools` pools of `d` channels.
    pub fn bytes(pools: usize, d: usize) -> usize {
        pools * d * 2 + pools * 4
    }

    /// 2026-10-14: Sized for `geom.n_pools_full` pools when `on`, else empty.
    pub(crate) fn alloc(gpu: &dyn GpuBackend, geom: &DsaSelectGeometry, on: bool) -> Result<Self> {
        if !on {
            return Ok(Self {
                keys_bf16: DevicePtr(0),
                vis: DevicePtr(0),
                pools: 0,
            });
        }
        let n = geom.n_pools_full;
        Ok(Self {
            keys_bf16: gpu.alloc(n * geom.index_head_dim * 2)?,
            vis: gpu.alloc(n * 4)?,
            pools: n,
        })
    }

    pub(crate) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.pools > 0 {
            gpu.free(self.keys_bf16)?;
            gpu.free(self.vis)?;
        }
        Ok(())
    }

    pub(super) fn check(&self, pools: usize) -> Result<()> {
        if pools > self.pools {
            bail!(
                "DSA select: the streamed scores need {pools} prepared pools but {} were \
                 reserved (METRALE_GLM_DSA_SCORES_PIPE must be set before the workspace is built)",
                self.pools
            );
        }
        Ok(())
    }
}

/// 2026-10-14: `dsa_pool_prep_bf16` over the pass's `pools` complete pools, after
/// `dsa_kpool_compress` on the same stream.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_prep(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    pools: usize,
    seq: usize,
    stream: u64,
) -> Result<()> {
    scratch.pipe.check(pools)?;
    KernelLaunch::new(gpu, kernels.rows.pool_prep)
        .grid([pools.div_ceil(PREP_POOLS_PER_BLOCK) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(scratch.pool_keys)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.pool_valid)
        .arg_ptr(inputs.rows.valid)
        .arg_ptr(scratch.pipe.keys_bf16)
        .arg_ptr(scratch.pipe.vis)
        .arg_u32(pools as u32)
        .arg_u32(geom.index_head_dim as u32)
        .arg_u32(geom.index_kpool as u32)
        .arg_u32(seq as u32)
        .launch(stream)?;
    Ok(())
}

/// 2026-10-14: `dsa_index_scores_mma_pipe` in place of `dsa_index_scores_mma`: the same grid
/// and outputs, from the regions [`launch_prep`] wrote.
pub(super) fn launch_scores(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    pools: usize,
    stream: u64,
) -> Result<()> {
    scratch.pipe.check(pools)?;
    let d = geom.index_head_dim;
    KernelLaunch::new(gpu, kernels.rows.scores_pipe)
        .grid([geom.q_rows.div_ceil(SCORES_MMA_ROWS) as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SCORES_PIPE_SMEM)
        .arg_ptr(inputs.q)
        .arg_ptr(scratch.pipe.keys_bf16)
        .arg_ptr(inputs.weights)
        .arg_ptr(scratch.pipe.vis)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(scratch.scores)
        .arg_ptr(scratch.valid_cand)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(pools as u32)
        .arg_u32(geom.index_heads as u32)
        .arg_u32(d as u32)
        .arg_f32((d as f32).powf(-0.5))
        .launch(stream)?;
    Ok(())
}
