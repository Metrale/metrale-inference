// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The batched MTP verify's kernels the executor launches: the carried-state GDN
//! kernels (`qwen3_ssm/carry.rs`) and the one-launch row argmax (`verify_rows_argmax`), looked
//! up literally as their legacy sites do.
//!
//! Owner: model-layers circuit executor.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::try_kernel;

/// 2026-09-30: `(module, func, handle)` of each batched-verify kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 10] {
    const C: &str = "gated_delta_rule_carry";
    let carry = |f: &'static str| (C, f, look(C, f, &|| try_kernel(gpu, C, f)));
    [
        carry("gdn_carry_conv"),
        carry("gdn_carry_wy2"),
        carry("gdn_carry_wy3"),
        carry("gdn_carry_wy4"),
        carry("gdn_carry_wy2_lazy"),
        carry("gdn_carry_wy3_lazy"),
        carry("gdn_carry_wy4_lazy"),
        carry("gdn_carry_flush"),
        carry("gdn_carry_conv_flush"),
        (
            "argmax",
            "argmax_bf16_batch",
            look("argmax", "argmax_bf16_batch", &|| {
                try_kernel(gpu, "argmax", "argmax_bf16_batch")
            }),
        ),
    ]
}
