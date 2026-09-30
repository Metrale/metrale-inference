// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The declared-precision kernels the executor launches (the W8A8 quantizers and
//! GEMV entries, the W4A4 quantizer and MX entries), looked up literally as in the parent.
//!
//! Owner: model-layers circuit executor.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use crate::layers::try_kernel;

/// 2026-09-30: A lookup that is issued only for a kernel the target has (the parent's `look`).
pub(super) type Look<'a> = dyn Fn(&str, &str, &dyn Fn() -> KernelHandle) -> KernelHandle + 'a;

/// 2026-09-30: `(module, func, handle)` of each declared-precision kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 13] {
    [
        (
            "w8a8_act_quant",
            "w8a8_act_quant_row",
            look("w8a8_act_quant", "w8a8_act_quant_row", &|| {
                try_kernel(gpu, "w8a8_act_quant", "w8a8_act_quant_row")
            }),
        ),
        (
            "w8a8_act_quant",
            "w8a8_act_quant_silu_row",
            look("w8a8_act_quant", "w8a8_act_quant_silu_row", &|| {
                try_kernel(gpu, "w8a8_act_quant", "w8a8_act_quant_silu_row")
            }),
        ),
        (
            "w8a8_gemv",
            "w8a8_gemv_rowscale_mb1_ku8",
            look("w8a8_gemv", "w8a8_gemv_rowscale_mb1_ku8", &|| {
                try_kernel(gpu, "w8a8_gemv", "w8a8_gemv_rowscale_mb1_ku8")
            }),
        ),
        (
            "w8a8_gemv",
            "w8a8_gemv_rowscale_mb2",
            look("w8a8_gemv", "w8a8_gemv_rowscale_mb2", &|| {
                try_kernel(gpu, "w8a8_gemv", "w8a8_gemv_rowscale_mb2")
            }),
        ),
        (
            "w8a8_gemv",
            "w8a8_gemv_rowscale_mb4",
            look("w8a8_gemv", "w8a8_gemv_rowscale_mb4", &|| {
                try_kernel(gpu, "w8a8_gemv", "w8a8_gemv_rowscale_mb4")
            }),
        ),
        (
            "w8a8_gemv",
            "w8a8_gemv_rowscale_mb8",
            look("w8a8_gemv", "w8a8_gemv_rowscale_mb8", &|| {
                try_kernel(gpu, "w8a8_gemv", "w8a8_gemv_rowscale_mb8")
            }),
        ),
        (
            "w8a8_gemv",
            "w8a8_gemv_rowscale_mb16",
            look("w8a8_gemv", "w8a8_gemv_rowscale_mb16", &|| {
                try_kernel(gpu, "w8a8_gemv", "w8a8_gemv_rowscale_mb16")
            }),
        ),
        (
            "w4a4_gemv_mx",
            "w4a4_quant_rows",
            look("w4a4_gemv_mx", "w4a4_quant_rows", &|| {
                try_kernel(gpu, "w4a4_gemv_mx", "w4a4_quant_rows")
            }),
        ),
        (
            "w4a4_gemv_mx",
            "w4a4_gemv_mx8",
            look("w4a4_gemv_mx", "w4a4_gemv_mx8", &|| {
                try_kernel(gpu, "w4a4_gemv_mx", "w4a4_gemv_mx8")
            }),
        ),
        (
            "w4a4_gemv_mx",
            "w4a4_gemv_mx16_nt2",
            look("w4a4_gemv_mx", "w4a4_gemv_mx16_nt2", &|| {
                try_kernel(gpu, "w4a4_gemv_mx", "w4a4_gemv_mx16_nt2")
            }),
        ),
        (
            "w4a4_gemv_mx",
            "w4a4_gemv_mx32_nt4",
            look("w4a4_gemv_mx", "w4a4_gemv_mx32_nt4", &|| {
                try_kernel(gpu, "w4a4_gemv_mx", "w4a4_gemv_mx32_nt4")
            }),
        ),
        (
            "w4a4_gemv_mx",
            "w4a4_gemv_mx16_ps",
            look("w4a4_gemv_mx", "w4a4_gemv_mx16_ps", &|| {
                try_kernel(gpu, "w4a4_gemv_mx", "w4a4_gemv_mx16_ps")
            }),
        ),
        (
            "w4a4_gemv_mx",
            "w4a4_gemv_mx32_ps",
            look("w4a4_gemv_mx", "w4a4_gemv_mx32_ps", &|| {
                try_kernel(gpu, "w4a4_gemv_mx", "w4a4_gemv_mx32_ps")
            }),
        ),
    ]
}
