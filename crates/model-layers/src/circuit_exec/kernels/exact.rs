// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The exact MTP verify chain's kernels (`emitters/gdn_exact.rs`), looked up as the
//! legacy layer does (`qwen3_ssm/carry.rs` `carry_kernels`, `carry_flush_kernel`): the model
//! directory's `gdn_exact_carry` twins through `try_target_kernel`, the common carry module's
//! FP32 conv twins through `try_kernel`.
//!
//! Owner: model-layers circuit executor.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::{try_kernel, try_target_kernel};

/// 2026-10-03: `(module, func, handle)` of each exact-verify kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 12] {
    const C: &str = "gated_delta_rule_carry";
    const X: &str = "gdn_exact_carry";
    let conv = |f: &'static str| (C, f, look(C, f, &|| try_kernel(gpu, C, f)));
    let exact = |f: &'static str| (X, f, look(X, f, &|| try_target_kernel(gpu, X, f)));
    [
        conv("gdn_conv_chain_f32"),
        conv("gdn_carry_conv_f32"),
        exact("gdn_exact_chain2"),
        exact("gdn_exact_chain3"),
        exact("gdn_exact_chain4"),
        exact("gdn_exact_carry2"),
        exact("gdn_exact_carry3"),
        exact("gdn_exact_carry4"),
        exact("gdn_exact_carry2_lazy"),
        exact("gdn_exact_carry3_lazy"),
        exact("gdn_exact_carry4_lazy"),
        exact("gdn_exact_carry_flush"),
    ]
}
