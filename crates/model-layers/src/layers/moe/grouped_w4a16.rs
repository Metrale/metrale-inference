// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The routed grouped W4A16 GEMM of the untransposed MoE prefill: its kernels (the
//! base `moe_w4a16_grouped_gemm_ptrtable`, its opt-in k32 and m256 twins and the target's
//! pipelined twins) and the launcher that picks one. Moved out of `init.rs` and `helpers_b.rs`
//! (500-line cap) when the pipelined twin was added.
//!
//! Owner: model-layers (MoE).
//! Invariants: the pipelined twins run only the shapes `ops::moe_w4a16_pipe_admits` admits, and
//! their output bytes equal the base kernel's (`ops/moe_w4a16_pipe_tests.rs`).

use super::*;
use crate::layers::{try_kernel, try_target_kernel};

/// 2026-10-13: The grouped W4A16 kernels `MoeLayer::launch_grouped_gemm` chooses from.
pub(super) struct GroupedW4a16Kernels {
    base: KernelHandle,
    /// 2026-09-25: Wider-K twin of `base`. Loaded only when `METRALE_MOE_GROUPED_K32=1` and the
    /// target ships it; preferred over `base` whenever it is non-zero.
    k32: KernelHandle,
    /// 2026-09-25: M_TILE=256 twin of `base`. Loaded only when `METRALE_MOE_GROUPED_M256=1` and
    /// the target ships it; used whenever it is non-zero.
    m256: KernelHandle,
    /// 2026-10-13: The cp.async-pipelined twin (`moe_w4a16_grouped_pipe_hopper`), byte-identical
    /// to `base`; resolved on a target that compiles it (Hopper), zero elsewhere.
    pipe: KernelHandle,
    /// 2026-10-14: The pipelined twin over a persistent grid (same module), preferred over
    /// `pipe`: it reads the tile count on the device, so the prefill needs no host copy of the
    /// expert offsets.
    persist: KernelHandle,
    /// 2026-10-14: The warpgroup-MMA twin (`moe_w4a16_grouped_wgmma_hopper`). Loaded only when
    /// `METRALE_MOE_PREFILL_WGMMA=1` and the target compiles it (Hopper); preferred over
    /// `persist` for the shapes it admits on calls of at least `wgmma_min_rows` rows, below
    /// which it is slower. It wrote the base kernel's bytes on the H100
    /// (`ops/moe_w4a16_wgmma_tests.rs`, wgmma-01); it stays opt-in until the chunk gate
    /// (`ops/moe_w4a16_wgmma_chunk_tests.rs`), the crossover sweep and a serve A/B report.
    wgmma: KernelHandle,
    /// 2026-10-14: The recipe's routed-rows minimum for `wgmma`
    /// (`METRALE_MOE_PREFILL_WGMMA_MIN_ROWS`, required with the lever); `None` with the lever off.
    wgmma_min_rows: Option<u32>,
}

impl GroupedW4a16Kernels {
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let opt_in = |env: &str, func: &str| {
            if std::env::var(env).as_deref() == Ok("1") {
                try_kernel(gpu, "moe_w4a16", func)
            } else {
                KernelHandle(0)
            }
        };
        let wgmma_on = std::env::var("METRALE_MOE_PREFILL_WGMMA").as_deref() == Ok("1");
        let wgmma_min_rows = if wgmma_on {
            let raw = std::env::var(ops::MOE_W4A16_WGMMA_MIN_ROWS_ENV).ok();
            Some(ops::moe_w4a16_wgmma_min_rows(raw.as_deref())?)
        } else {
            None
        };
        Ok(Self {
            base: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?,
            k32: opt_in(
                "METRALE_MOE_GROUPED_K32",
                "moe_w4a16_grouped_gemm_ptrtable_k32",
            ),
            m256: opt_in(
                "METRALE_MOE_GROUPED_M256",
                "moe_w4a16_grouped_gemm_ptrtable_m256",
            ),
            pipe: try_target_kernel(
                gpu,
                "moe_w4a16_grouped_pipe_hopper",
                "moe_w4a16_grouped_gemm_ptrtable_pipe",
            ),
            persist: try_target_kernel(
                gpu,
                "moe_w4a16_grouped_pipe_hopper",
                "moe_w4a16_grouped_gemm_ptrtable_persist",
            ),
            wgmma: if wgmma_on {
                try_target_kernel(
                    gpu,
                    "moe_w4a16_grouped_wgmma_hopper",
                    "moe_w4a16_grouped_gemm_ptrtable_wgmma",
                )
            } else {
                KernelHandle(0)
            },
            wgmma_min_rows,
        })
    }

    /// 2026-10-14: Whether `launch_grouped_gemm` takes the persistent twin for an `n_out` x `k`
    /// projection: resolved, admitted, and no opt-in twin ahead of it.
    pub(super) fn persistent(&self, gpu: &dyn GpuBackend, n_out: u32, k: u32) -> bool {
        self.m256.0 == 0
            && self.k32.0 == 0
            && ops::moe_w4a16_pipe_admits(gpu, self.persist, n_out, k)
    }

    /// 2026-10-14: Whether `launch_grouped_gemm` takes the wgmma twin for a call of
    /// `total_rows` expert-sorted rows: loaded (the lever), admitted for `num_experts`, routed
    /// by `ops::moe_w4a16_wgmma_routes` at `wgmma_min_rows`, and no opt-in twin ahead of it.
    pub(super) fn wgmma(
        &self,
        gpu: &dyn GpuBackend,
        n_out: u32,
        k: u32,
        num_experts: u32,
        total_rows: u32,
    ) -> bool {
        self.m256.0 == 0
            && self.k32.0 == 0
            && self
                .wgmma_min_rows
                .is_some_and(|min| ops::moe_w4a16_wgmma_routes(total_rows, min))
            && ops::moe_w4a16_wgmma_admits(gpu, self.wgmma, n_out, k, num_experts)
    }
}

impl MoeLayer {
    /// 2026-09-25: Launch the routed grouped GEMM. With the M_TILE=256 kernel
    /// (METRALE_MOE_GROUPED_M256=1 and resolved) the M=64 tile count `max_m_tiles` is divided by
    /// 4, rounded up; else the k32 twin when it resolved (METRALE_MOE_GROUPED_K32=1); else
    /// (2026-10-13) the pipelined twin when it resolved and admits the shape; else the base kernel.
    /// The two opt-ins keep priority over the pipelined twin. 2026-10-14: the persistent pipelined
    /// twin goes first when `GroupedW4a16Kernels::persistent`; it ignores `max_m_tiles` and sizes
    /// its grid from `total_rows`, the expert-sorted row count. Ahead of it, the wgmma twin when
    /// `GroupedW4a16Kernels::wgmma` (METRALE_MOE_PREFILL_WGMMA=1 and `total_rows` at least
    /// METRALE_MOE_PREFILL_WGMMA_MIN_ROWS), sized the same way.
    ///
    /// The k32 and m256 kernels are opt-in. Measured 2026-08-27 on GB10 (qwen4_exp, 28K
    /// prefill, back to back): k32 +1.0% and m256 +1.4% over the arm without them,
    /// inside run-to-run noise, and m256 averaged 31.30 ms per call against 26.17 ms for the base kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_grouped_gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
        c: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_experts: u32,
        n_out: u32,
        k: u32,
        max_m_tiles: u32,
        total_rows: u32,
        stream: u64,
    ) -> Result<()> {
        let kn = &self.w4a16_grouped;
        if kn.wgmma(gpu, n_out, k, num_experts, total_rows) {
            return ops::moe_w4a16_grouped_gemm_ptrtable_wgmma(
                gpu,
                kn.wgmma,
                a,
                packed_ptrs,
                scale_ptrs,
                scale2_vals,
                c,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                n_out,
                k,
                total_rows,
                stream,
            );
        }
        if kn.persistent(gpu, n_out, k) {
            return ops::moe_w4a16_grouped_gemm_ptrtable_persist(
                gpu,
                kn.persist,
                a,
                packed_ptrs,
                scale_ptrs,
                scale2_vals,
                c,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                n_out,
                k,
                total_rows,
                stream,
            );
        }
        let pipe =
            kn.m256.0 == 0 && kn.k32.0 == 0 && ops::moe_w4a16_pipe_admits(gpu, kn.pipe, n_out, k);
        let (kernel, m_tiles) = if kn.m256.0 != 0 {
            (kn.m256, max_m_tiles.div_ceil(4).max(1))
        } else if pipe {
            (kn.pipe, max_m_tiles)
        } else if kn.k32.0 != 0 {
            (kn.k32, max_m_tiles)
        } else {
            (kn.base, max_m_tiles)
        };
        let launch = if kn.m256.0 != 0 {
            ops::moe_w4a16_grouped_gemm_ptrtable_m256
        } else if pipe {
            ops::moe_w4a16_grouped_gemm_ptrtable_pipe
        } else {
            ops::moe_w4a16_grouped_gemm_ptrtable
        };
        launch(
            gpu,
            kernel,
            a,
            packed_ptrs,
            scale_ptrs,
            scale2_vals,
            c,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            n_out,
            k,
            m_tiles,
            stream,
        )
    }
}
