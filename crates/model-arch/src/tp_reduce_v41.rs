// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The DeepSeek-V4.1 tensor-parallel reduction. A rank writes f32 partial sums
//! (`[rows, dim]`) into [`TpReduce::partial`]; `reduce` all-gathers every rank's partials and
//! sums them on each rank with `moe_v41_rank_combine`, which rounds each expert's (or the one
//! projection's) total to bf16 once, whole, as the single-device reference does. Every rank
//! computes the same sums in the same order, so their outputs are identical bytes.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - One world: `world` fixed at construction; one rank reads its own partials (no gather).
//! - The all-gather runs on the communicator's stream, the default stream. A caller on another
//!   stream is drained before it and the default stream after it, so the combine never reads a
//!   slot before it lands.

use anyhow::{Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-19: Which rows of the gathered partials make up each token's output.
#[derive(Clone, Copy, Debug)]
pub struct CombineRows {
    /// 2026-10-19: Rows the partial buffer holds this call (every rank the same count).
    pub rows: usize,
    /// 2026-10-19: Device `[m * topk]` i32 rows, each token's in ascending expert order;
    /// ignored when `topk` is 0.
    pub pos: DevicePtr,
    pub topk: usize,
    /// 2026-10-19: Token `t`'s one remaining row is `shared_base + t`.
    pub shared_base: usize,
    pub m: usize,
}

pub struct TpReduce {
    world: usize,
    dim: usize,
    cap_rows: usize,
    p: DevicePtr,
    /// 2026-10-19: `world` slots of `cap_rows * dim` f32 (only the first `rows * dim` of each
    /// slot used per call); the partial buffer itself at one rank.
    g: DevicePtr,
    combine: KernelHandle,
}

impl TpReduce {
    pub fn new(gpu: &dyn GpuBackend, world: usize, cap_rows: usize, dim: usize) -> Result<Self> {
        ensure!(world >= 1, "tensor-parallel world of 0 ranks");
        let bytes = cap_rows * dim * 4;
        let p = gpu.alloc(bytes.max(16))?;
        let g = if world == 1 {
            p
        } else {
            gpu.alloc(world * bytes)?
        };
        Ok(Self {
            world,
            dim,
            cap_rows,
            p,
            g,
            combine: gpu.kernel("moe_v41", "moe_v41_rank_combine")?,
        })
    }

    pub fn world(&self) -> usize {
        self.world
    }

    /// 2026-10-19: This rank's f32 `[cap_rows, dim]` partial-sum buffer.
    pub fn partial(&self) -> DevicePtr {
        self.p
    }

    pub fn cap_rows(&self) -> usize {
        self.cap_rows
    }

    /// 2026-10-19: Sum the ranks' partials into bf16 `out` (`[m, dim]`). `comm` must span
    /// `world` ranks when `world > 1`.
    pub fn reduce(
        &self,
        gpu: &dyn GpuBackend,
        comm: Option<&dyn CommBackend>,
        rows: CombineRows,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows.rows <= self.cap_rows && rows.shared_base + rows.m <= rows.rows,
            "tensor-parallel reduce of {} rows (shared at {} + {}) exceeds {} rows",
            rows.rows,
            rows.shared_base,
            rows.m,
            self.cap_rows
        );
        let slot_elems = rows.rows * self.dim;
        if self.world > 1 {
            let comm = comm.ok_or_else(|| {
                anyhow::anyhow!("tensor-parallel reduce over {} ranks without a communicator", self.world)
            })?;
            ensure!(
                comm.world_size() == self.world,
                "communicator spans {} ranks, the reduce {}",
                comm.world_size(),
                self.world
            );
            let foreign = stream != gpu.default_stream();
            if foreign {
                gpu.synchronize(stream)?;
            }
            comm.all_gather(self.p.0, self.g.0, slot_elems * 4)?;
            if foreign {
                gpu.synchronize(gpu.default_stream())?;
            }
        }
        if rows.m == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.combine)
            .grid([(self.dim as u32).div_ceil(256), rows.m as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.g)
            .arg_u64(slot_elems as u64)
            .arg_u32(self.world as u32)
            .arg_ptr(if rows.topk == 0 { self.g } else { rows.pos })
            .arg_u32(rows.topk as u32)
            .arg_u32(rows.shared_base as u32)
            .arg_ptr(out)
            .arg_u32(self.dim as u32)
            .launch(stream)
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.g != self.p {
            gpu.free(self.g)?;
        }
        gpu.free(self.p)
    }
}

#[cfg(test)]
#[path = "tp_reduce_v41_tests.rs"]
mod tests;
