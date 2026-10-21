// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_DSA_SELECTOR_PAIR` (default on): a batched DSA step's selector query
//! (`wq_b` over `q_resid`) and head weights (`weights_proj` over the layer input) as one
//! `dense_gemv_bf16_batchm_few_fp32out_pair` launch instead of two
//! `dense_gemv_bf16_batchm_few_fp32out` launches. The pair's blocks run that entry's body on each
//! operand set, so each output keeps its bytes (`examples/glm5next_dsa_proj_in_gate`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Taken only where `glm_mm` would launch the FP32-out few-row entry for both (2..=8 rows,
//!   `METRALE_GLM_BATCHM_FEW` on, neither weight served by a quantized registry); otherwise
//!   `selector_query_rows` launches each as before.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Glm5NextDsaLayer;
use crate::glm5next_layer::wide_gemv::fixed_entry;

impl Glm5NextDsaLayer {
    /// 2026-10-21: Whether [`Self::selector_pair`] takes `n` rows (nothing launched).
    pub(super) fn selector_pair_takes(&self, n: usize) -> bool {
        let batchm = self.kernels.batchm_f32();
        let unquantized = [self.weights.wq_b, self.weights.weights_proj].iter().all(|&w| {
            crate::glm5next_w4a16_dense::footprint(w).is_none()
                && !crate::glm5next_fp8_dense::holds(w)
        });
        self.levers.dsa_selector_pair
            && self.kernels.selector_pair.0 != 0
            && batchm.few.0 != 0
            && fixed_entry(&batchm, n, &self.levers.dense).map(|h| h.0) == Some(batchm.few.0)
            && unquantized
    }

    /// 2026-10-21: The selector query and head weights of `n` rows as one launch where
    /// [`Self::selector_pair_takes`]; `Ok(false)`, with nothing launched, otherwise.
    pub(super) fn selector_pair(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<bool> {
        if !self.selector_pair_takes(n) {
            return Ok(false);
        }
        let w = &self.workspace;
        let c = &self.cfg;
        let (heads, idx_row) = (c.index_heads as u32, (c.index_heads * c.index_head_dim) as u32);
        KernelLaunch::new(gpu, self.kernels.selector_pair)
            .grid([div_ceil(idx_row, 4) + div_ceil(heads, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.q_resid)
            .arg_ptr(self.weights.wq_b)
            .arg_ptr(w.q_idx_rows)
            .arg_u32(idx_row)
            .arg_u32(c.q_lora_rank as u32)
            .arg_u32(idx_row)
            .arg_ptr(hidden)
            .arg_ptr(self.weights.weights_proj)
            .arg_ptr(w.head_weights_rows)
            .arg_u32(heads)
            .arg_u32(c.hidden as u32)
            .arg_u32(heads)
            .arg_u32(n as u32)
            .launch(stream)?;
        Ok(true)
    }
}

#[cfg(test)]
#[path = "selector_pair_tests.rs"]
mod tests;
