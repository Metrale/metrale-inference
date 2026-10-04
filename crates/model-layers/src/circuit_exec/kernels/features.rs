// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The kernels the FEATURES workstream's emitters launch (LoRA's routed fold and the
//! pair's scaled add), looked up literally as in the parent.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::try_kernel;

/// 2026-10-03: `(module, func, handle)` of each kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 3] {
    [
        (
            "lora_bgmv",
            "lora_bgmv_shrink",
            look("lora_bgmv", "lora_bgmv_shrink", &|| {
                try_kernel(gpu, "lora_bgmv", "lora_bgmv_shrink")
            }),
        ),
        (
            "lora_bgmv",
            "lora_bgmv_expand_fold",
            look("lora_bgmv", "lora_bgmv_expand_fold", &|| {
                try_kernel(gpu, "lora_bgmv", "lora_bgmv_expand_fold")
            }),
        ),
        (
            "residual_add",
            "bf16_scaled_add",
            look("residual_add", "bf16_scaled_add", &|| {
                try_kernel(gpu, "residual_add", "bf16_scaled_add")
            }),
        ),
    ]
}
