// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The FP8 W8A16 projection kernels the executor launches (the row-tile entries), looked up literally as their launchers do.
//!
//! Owner: model-layers (MoE) circuit emitters.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::try_kernel;

/// 2026-10-03: `(module, func, handle)` of each FP8 projection kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 4] {
    [
        (
            "w8a16_tc_rows",
            "w8a16_tc_rows_16",
            look("w8a16_tc_rows", "w8a16_tc_rows_16", &|| {
                try_kernel(gpu, "w8a16_tc_rows", "w8a16_tc_rows_16")
            }),
        ),
        (
            "w8a16_tc_rows",
            "w8a16_tc_rows_32",
            look("w8a16_tc_rows", "w8a16_tc_rows_32", &|| {
                try_kernel(gpu, "w8a16_tc_rows", "w8a16_tc_rows_32")
            }),
        ),
        (
            "w8a16_tc_rows",
            "w8a16_tc_rows_64",
            look("w8a16_tc_rows", "w8a16_tc_rows_64", &|| {
                try_kernel(gpu, "w8a16_tc_rows", "w8a16_tc_rows_64")
            }),
        ),
        (
            "w8a16_tc_rows",
            "w8a16_tc_rows_64c",
            look("w8a16_tc_rows", "w8a16_tc_rows_64c", &|| {
                try_kernel(gpu, "w8a16_tc_rows", "w8a16_tc_rows_64c")
            }),
        ),
    ]
}
