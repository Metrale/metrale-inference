// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The batched-GEMV sets the MLP hands `glm_mm` (moved out of `mod.rs` to keep it
//! under 500 lines; the bodies are as they were, plus the reg8 entries).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond `glm5next_layer::wide_gemv`'s.

use super::Glm5NextMlpKernels;

impl Glm5NextMlpKernels {
    /// 2026-10-09: The BF16-out batched GEMV pair `glm_mm` takes.
    pub(crate) fn batchm(&self) -> crate::glm5next_layer::wide_gemv::Batchm {
        crate::glm5next_layer::wide_gemv::Batchm {
            narrow: self.gemv_batchm,
            reg8: self.gemv_batchm_reg8,
            wide: self.gemv_batchm_wide,
            out_bytes: 2,
        }
    }

    /// 2026-10-09: The FP32-out batched GEMV pair, for the router logits.
    pub(crate) fn batchm_f32(&self) -> crate::glm5next_layer::wide_gemv::Batchm {
        crate::glm5next_layer::wide_gemv::Batchm {
            narrow: self.gemv_batchm_f32,
            reg8: self.gemv_batchm_reg8_f32,
            wide: self.gemv_batchm_wide_f32,
            out_bytes: 4,
        }
    }
}
