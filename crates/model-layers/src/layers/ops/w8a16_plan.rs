// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The levers the W8A16 tile launchers run under ([`W8a16Plan`], carried on
//! `OpsLevers::w8a16`), and the 32- and 64-row tiles as lever-free function pointers for the
//! callers that pick their arm as a pointer shared with launchers that take no lever (the
//! multi-seq attention O and Q/K/V projections).
//!
//! Owner: model-layers ops.
//! Invariants: the pointer an `*_arm` method returns launches exactly what the launcher it
//! names launches with `tc_rows` set to [`W8a16Plan::tc_rows`].

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// 2026-10-17: The W8A16 tile levers, resolved once from the startup settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W8a16Plan {
    /// 2026-09-28: On unless `METRALE_NO_W8A16_PIPE128` is present at any value: from
    /// `W8A16_PIPE128_MIN_ROWS` rows `w8a16_gemm_pipelined` launches its 128x128 twin.
    pub pipe128: bool,
    /// 2026-09-28: On unless `METRALE_NO_W8A16_TC_ROWS` is present at any value: under the
    /// canonical tiers the 32- and 64-row tiles launch `w8a16_tc_rows` (`w8a16_tc_rows.rs`).
    pub tc_rows: bool,
}

/// 2026-10-17: The signature of the strided batched W8A16 launchers
/// (`w8a16_gemv_batch{4,16}_strided`, the `_ncol*_strided` and `m16` rungs and the tile arms
/// below), so a tier choice is a function pointer.
pub type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

impl W8a16Plan {
    /// 2026-10-17: Both levers on, as unset variables leave them.
    pub const fn declared() -> Self {
        Self {
            pipe128: true,
            tc_rows: true,
        }
    }

    /// 2026-10-17: [`super::w8a16_gemm_pipelined_m32`] under this plan.
    pub fn m32_arm(self) -> super::ContiguousBatchGemv {
        if self.tc_rows {
            m32::<true>
        } else {
            m32::<false>
        }
    }

    /// 2026-10-17: [`super::w8a16_gemm_pipelined_m64`] under this plan.
    pub fn m64_arm(self) -> super::ContiguousBatchGemv {
        if self.tc_rows {
            m64::<true>
        } else {
            m64::<false>
        }
    }

    /// 2026-10-17: [`super::w8a16_gemm_pipelined_m32_strided`] under this plan.
    pub fn m32_strided_arm(self) -> StridedBatchGemv {
        if self.tc_rows {
            m32_strided::<true>
        } else {
            m32_strided::<false>
        }
    }

    /// 2026-10-17: [`super::w8a16_gemm_pipelined_m64_strided`] under this plan.
    pub fn m64_strided_arm(self) -> StridedBatchGemv {
        if self.tc_rows {
            m64_strided::<true>
        } else {
            m64_strided::<false>
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn m32<const TC_ROWS: bool>(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::w8a16_gemm_pipelined_m32(gpu, kernel, TC_ROWS, a, b, scale, c, m, n, k, stream)
}

#[allow(clippy::too_many_arguments)]
fn m64<const TC_ROWS: bool>(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::w8a16_gemm_pipelined_m64(gpu, kernel, TC_ROWS, a, b, scale, c, m, n, k, stream)
}

#[allow(clippy::too_many_arguments)]
fn m32_strided<const TC_ROWS: bool>(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    super::w8a16_gemm_pipelined_m32_strided(
        gpu, kernel, TC_ROWS, a, b, scale, c, m, n, k, lda, ldc, stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn m64_strided<const TC_ROWS: bool>(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    super::w8a16_gemm_pipelined_m64_strided(
        gpu, kernel, TC_ROWS, a, b, scale, c, m, n, k, lda, ldc, stream,
    )
}
