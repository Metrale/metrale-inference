// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The n-row MTP draft's kernels the executor launches (`forward_batch_position`):
//! the BF16 tensor-core GEMV tiers (`dense_gemv_tc`), the wider NVFP4 LM-head tiers
//! (`MtpHead::lm_head_batch_kernel`) and the argmax that also writes the confidences, looked up
//! literally as their legacy sites do.
//!
//! Owner: model-layers circuit executor.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::try_kernel;

/// 2026-09-30: `(module, func, handle)` of each n-row draft kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 6] {
    let one = |m: &'static str, f: &'static str| (m, f, look(m, f, &|| try_kernel(gpu, m, f)));
    [
        one("dense_gemv_bf16_tc", "dense_gemv_bf16_tc8"),
        one("dense_gemv_bf16_tc", "dense_gemv_bf16_tc16"),
        one("dense_gemv_bf16_tc", "dense_gemv_bf16_tc32"),
        one("w4a16_gemv_tc", "w4a16_gemv_tc16"),
        one("w4a16_gemv", "w4a16_gemv_batch32"),
        one("argmax", "argmax_bf16_batch_lp"),
    ]
}
