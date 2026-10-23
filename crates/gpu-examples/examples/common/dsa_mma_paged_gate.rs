// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `dsa_index_scores_mma_paged_bf16` (`METRALE_GLM_DSA_DECODE_SCORES_MMA`, the
//! persisted BF16 keys on tensor cores) against `dsa_index_scores_mma_pipe`, the prefill
//! selection's tensor-core scores, given the same BF16 keys contiguously and each pool's
//! visibility (its last token, so `vis <= q_pos` is the persisted candidacy): row `r`'s score and
//! candidacy bytes must be the pipe's for every pool below its live count `n_r`, and the bytes
//! past `n_r` the sentinel fill (not written).
//!
//! Owner: model-arch examples.
//! Invariants: returns an error unless every compared byte matches, at every grid.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{Case, D, H, KP, MAX_POOLS, read, run_on};

/// 2026-10-23: `dsa_index_scores_mma_pipe`'s dynamic shared memory: 3 stages of a 64-pool BF16
/// tile (row stride 136 halves) and 64 visibility words.
const PIPE_SMEM: u32 = 3 * (64 * 136 * 2 + 256);

fn f32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}

pub fn check(
    gpu: &dyn GpuBackend,
    paged: KernelHandle,
    pipe: KernelHandle,
    c: &Case,
    grids: &[[u32; 3]],
) -> Result<()> {
    let (rows, p) = (c.rows, c.pools);
    let out = gpu.alloc((rows * p * 4).max(16))?;
    let cand = gpu.alloc(rows * p.max(16))?;
    KernelLaunch::new(gpu, pipe)
        .grid([rows.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(PIPE_SMEM)
        .arg_ptr(c.q)
        .arg_ptr(c.keys16)
        .arg_ptr(c.w)
        .arg_ptr(c.vis)
        .arg_ptr(c.qpos)
        .arg_ptr(out)
        .arg_ptr(cand)
        .arg_u32(rows as u32)
        .arg_u32(p as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_f32((D as f32).powf(-0.5))
        .launch(0)?;
    let ro = read(gpu, out, rows * p * 4)?;
    let rc = read(gpu, cand, rows * p)?;
    gpu.free(out)?;
    gpu.free(cand)?;
    let sb = rows * MAX_POOLS * 4;
    for &g in grids {
        let b = run_on(gpu, paged, c, c.pk16, g)?;
        for r in 0..rows {
            let n = (c.s - (rows - 1 - r)) / KP;
            for q in 0..MAX_POOLS {
                let (got_s, got_c) = (f32_at(&b, r * MAX_POOLS + q), b[sb + r * MAX_POOLS + q]);
                let (want_s, want_c) = if q < n {
                    (f32_at(&ro, r * p + q), rc[r * p + q])
                } else {
                    (0x5a5a_5a5a, 0xa5)
                };
                ensure!(
                    (got_s, got_c) == (want_s, want_c),
                    "S={} rows={rows} grid {g:?}: dsa_index_scores_mma_paged_bf16 row {r} pool {q} \
                     (live {n}): score {got_s:#010x} cand {got_c}, expected {want_s:#010x} {want_c}",
                    c.s
                );
            }
        }
    }
    Ok(())
}
