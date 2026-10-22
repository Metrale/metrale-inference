// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `METRALE_GLM_SHARED_FORK=<rows>`: the MoE site's shared expert on a side stream,
//! beside the router and the routed experts, for launches of up to `<rows>` rows. It forks after
//! the site's input is ready (the shared expert reads only `x`) and joins before the combine, which
//! adds it exactly as before; under a CUDA graph capture the fork and join are graph edges.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `0` (2026-10-22: the default is 2): no side stream, no extra buffers, every launch as
//!   before.
//! - Under the lever the shared expert runs the same kernels on the same inputs into the same
//!   `shared_out`, with its own gate/up/act scratch (the routed experts use the workspace's), so
//!   every output byte is unchanged; only the order of independent launches changes.
//! - The side stream and both events exist from load (never created inside a capture).

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;

use super::dense::DenseScratch;
use crate::glm5next_mlp::Glm5NextMlpConfig;

/// 2026-10-11: The widest launch the lever may name.
pub const SHARED_FORK_MAX_ROWS: usize = 64;

/// 2026-10-11: The side stream, the fork and join events, and the shared expert's own scratch
/// for up to `max_rows` rows.
pub(crate) struct SharedFork {
    side: u64,
    ev_fork: u64,
    ev_join: u64,
    scratch: DenseScratch,
}

impl SharedFork {
    /// 2026-10-11: The fork state for launches of up to `min(fork_rows, workspace_rows)` rows,
    /// or `None` when `fork_rows` is 0 or (2026-10-22, reported) the backend has no side
    /// stream. Refuses more than [`SHARED_FORK_MAX_ROWS`].
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextMlpConfig,
        workspace_rows: usize,
        fork_rows: usize,
    ) -> Result<Option<Self>> {
        ensure!(
            fork_rows <= SHARED_FORK_MAX_ROWS,
            "GLM shared fork: {fork_rows} rows (at most {SHARED_FORK_MAX_ROWS})"
        );
        if fork_rows == 0 {
            return Ok(None);
        }
        let rows = fork_rows.min(workspace_rows).max(1);
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "GLM shared fork: MoE sites of up to {rows} rows run the shared expert on a side \
                 stream (METRALE_GLM_SHARED_FORK={fork_rows}; 0 turns it off)"
            )
        });
        let inter = cfg.local_shared_intermediate.max(1);
        let bytes = rows * inter * 2;
        let side = gpu.create_stream()?;
        // 2026-10-22: Default on since today, so a backend without side streams runs the
        // shared expert on the main stream (reported once) instead of refusing the load.
        if side == 0 {
            tracing::warn!(
                "METRALE_GLM_SHARED_FORK={fork_rows}: this backend has no side stream \
                 (create_stream gave the default stream); the shared expert stays on the main \
                 stream"
            );
            return Ok(None);
        }
        Ok(Some(Self {
            side,
            ev_fork: gpu.create_event()?,
            ev_join: gpu.create_event()?,
            scratch: DenseScratch {
                gate: gpu.alloc(bytes)?,
                up: gpu.alloc(bytes)?,
                act: gpu.alloc(bytes)?,
                max_inter: inter,
                max_rows: rows,
            },
        }))
    }

    /// 2026-10-11: Whether a launch of `rows` rows forks.
    pub(super) fn takes(&self, rows: usize) -> bool {
        rows <= self.scratch.max_rows
    }

    /// 2026-10-11: Run `shared(side, scratch)` on the side stream after everything `stream` has
    /// enqueued so far; [`Self::join`] makes `stream` wait for it. Errors when `stream` is the
    /// side stream.
    pub(super) fn fork(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        shared: impl FnOnce(u64, DenseScratch) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            self.side != stream,
            "GLM shared fork: side stream is the main stream"
        );
        gpu.record_event(self.ev_fork, stream)?;
        gpu.stream_wait_event(self.side, self.ev_fork)?;
        shared(self.side, self.scratch)?;
        gpu.record_event(self.ev_join, self.side)
    }

    /// 2026-10-11: `stream` waits for the forked shared expert.
    pub(super) fn join(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        gpu.stream_wait_event(stream, self.ev_join)
    }
}

#[cfg(test)]
mod tests {
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn cfg() -> Glm5NextMlpConfig {
        Glm5NextMlpConfig {
            hidden: 4096,
            local_dense_intermediate: 4096,
            dense_start: 0,
            moe_intermediate: 704,
            local_shared_intermediate: 768,
            shared_start: 0,
            num_experts: 288,
            local_experts: 288,
            ep_rank: 0,
            top_k: 8,
            routed_scale: 2.5,
            renormalize: true,
            swiglu_limit: 10.0,
            router_bf16_ladder: false,
            tp_world_size: 3,
            ep_world_size: 1,
            expert_shard: crate::glm5next_mlp::ExpertShard::Whole,
        }
    }

    /// 2026-10-11: 0 rows builds nothing (no stream, no allocation); more than the maximum is
    /// refused; a backend without side streams (the mock's default stream 0) never forks onto
    /// the main stream: 2026-10-22, it builds no fork (reported) instead of refusing.
    #[test]
    fn construction_refusals() {
        let gpu = MockGpuBackend::new();
        assert!(SharedFork::new(&gpu, &cfg(), 16, 0).unwrap().is_none());
        assert!(gpu.launches_snapshot().is_empty());
        let e = SharedFork::new(&gpu, &cfg(), 16, SHARED_FORK_MAX_ROWS + 1)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("at most 64"), "{e}");
        assert!(SharedFork::new(&gpu, &cfg(), 16, 4).unwrap().is_none());
    }
}
