// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The routed grouped W4A16 GEMM of the untransposed MoE prefill: its kernels (the
//! base `moe_w4a16_grouped_gemm_ptrtable`, its opt-in k32 and m256 twins and the target's
//! pipelined twin) and the launcher that picks one. Moved out of `init.rs` and `helpers_b.rs`
//! (500-line cap) when the pipelined twin was added.
//!
//! Owner: model-layers (MoE).
//! Invariants: the pipelined twin runs only the shapes `ops::moe_w4a16_pipe_admits` admits, and
//! its output bytes equal the base kernel's (`ops/moe_w4a16_pipe_tests.rs`).

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
        })
    }
}

impl MoeLayer {
    /// 2026-09-25: Launch the routed grouped GEMM. With the M_TILE=256 kernel
    /// (METRALE_MOE_GROUPED_M256=1 and resolved) the M=64 tile count `max_m_tiles` is divided by
    /// 4, rounded up; else the k32 twin when it resolved (METRALE_MOE_GROUPED_K32=1); else
    /// (2026-10-13) the pipelined twin when it resolved and admits the shape; else the base kernel.
    /// The two opt-ins keep priority over the pipelined twin.
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
        stream: u64,
    ) -> Result<()> {
        let kn = &self.w4a16_grouped;
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
