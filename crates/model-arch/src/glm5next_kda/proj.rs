// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `Glm5NextKdaLayer::gemm`, the BF16 projection dispatch every KDA projection
//! launches through (moved out of `mod.rs` verbatim, plus the `DenseCallSite`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Every call names its [`DenseCallSite`]; the 17..=32-row launches follow
//!   `ops::dense_batchm_32_at`, so a prefill projection never takes them.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::DenseCallSite;
use metrale_model_layers::weight_map::DenseWeight;

use super::Glm5NextKdaLayer;

impl Glm5NextKdaLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        site: DenseCallSite,
        stream: u64,
    ) -> Result<()> {
        // 2026-10-09: `--dense-quantization fp8` (registered weights, W8A8) is checked first in
        // `glm_mm`, as for every GLM projection.
        // 2026-09-25: Only M above `DENSE_GEMV_BATCHM_MAX_M` goes to cuBLASLt. Below it
        // `dense_mm_bf16` runs the M = 1 GEMV or the batched GEMV, whose rows carry the same
        // bits, so the cuBLASLt switch never changes those widths. 2026-10-09: the dispatch is
        // `glm_mm`'s, with the register-resident batched GEMV at 9..=16 rows.
        crate::glm5next_layer::wide_gemv::glm_mm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            crate::glm5next_layer::wide_gemv::Batchm {
                narrow: self.kernels.gemv_batchm,
                // 2026-10-16: No reg8 entry: every KDA projection is in `dense_projections`, so
                // under `--dense-quantization w4a16|fp8` `glm_mm` never reaches the GEMV here.
                reg8: KernelHandle(0),
                wide: self.kernels.gemv_batchm_wide,
                out_bytes: 2,
            },
            input,
            weight.weight,
            out,
            m,
            n,
            k,
            site,
            stream,
        )
    }
}
