// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The decode-step pre-norm `glm5next_hc_pre_norm_c1` (`glm5next_mhc_c1.cu`): when
//! `glm_hc_pre_norm` takes it. The kernel writes `glm5next_hc_pre_norm`'s bytes with the last
//! block's tail shortened; on GB10 at 2 tokens (90 distinct sites in a graph) 11.8 -> 9.5 us a
//! site, at 4 tokens it loses (12.1 -> 12.4: 158 registers a thread leave one block per SM, and
//! 96 blocks no longer fit one wave), so it is taken up to [`HC_PRE_C1_MAX_TOKENS`] only.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: [`c1_kernel`] is `Some` only with the lever on, 1..=[`HC_PRE_C1_MAX_TOKENS`]
//! tokens, hc 4, H 4096 and the kernel in the build; its grid is the per-row kernel's.

use metrale_gpu_runtime::gpu::KernelHandle;

use crate::glm5next_mhc::Glm5NextMhcKernels;

/// 2026-10-21: The kernel module and entry (`kernels/gb10/common/glm5next_mhc_c1.cu`).
pub const HC_PRE_C1_MODULE: &str = "glm5next_mhc_c1";
pub const HC_PRE_C1_KERNEL: &str = "glm5next_hc_pre_norm_c1";
/// 2026-10-21: The most tokens a launch takes (measured, see the module doc).
pub const HC_PRE_C1_MAX_TOKENS: u32 = 2;
/// 2026-10-21: The one shape the kernel is compiled for (GLM-5.3).
pub const HC_PRE_C1_HC: u32 = 4;
pub const HC_PRE_C1_HIDDEN: u32 = 4096;

/// 2026-10-21: The decode-step kernel when `on` and the step and shape are the ones it serves.
pub fn c1_kernel(
    kernels: &Glm5NextMhcKernels,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    on: bool,
) -> Option<KernelHandle> {
    (on && (1..=HC_PRE_C1_MAX_TOKENS).contains(&num_tokens)
        && hc_mult == HC_PRE_C1_HC
        && hidden_size == HC_PRE_C1_HIDDEN
        && kernels.hc_pre_norm_c1.0 != 0)
        .then_some(kernels.hc_pre_norm_c1)
}

#[cfg(test)]
#[path = "glm5next_mhc_c1_tests.rs"]
mod tests;
