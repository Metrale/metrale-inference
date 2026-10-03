// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The single-pass prefill's per-layer diagnostics under `--profile`, split from
//! `prefill_a.rs`.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-03: After layer `i` of a `proc_count`-row single-pass prefill: the Mistral
    /// hidden-norm log and, when `diag_prefill`, the last row's readback.
    pub(super) fn prefill_a_layer_diag(
        &self,
        i: usize,
        proc_count: usize,
        hidden: DevicePtr,
        diag_prefill: bool,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let fp32 = 2usize;
        // 2026-09-25: Mistral diagnostic under `self.profile`: log each layer's
        // hidden-state norm, once per model (`stats.dumped` is per model).
        if self.profile
            && self.config.model_type == "mistral"
            && self.stats.dumped.keyed("mla_prefill_norms")
        {
            self.gpu.synchronize(stream)?;
            let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
            let h_sz = self.config.hidden_size;
            let mut buf = vec![0u16; h_sz];
            // 2026-09-25: SAFETY: `buf` is `vec![0u16; h_sz]` on the line above, so it
            // owns exactly `h_sz * size_of::<u16>()` initialised bytes and
            // the length matches its capacity. `bytes` is the only live
            // reference to that allocation for its whole lifetime — it is
            // last used on the `copy_d2h` line below, and `buf` is not read
            // again until after that.
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, h_sz * 2) };
            if self.gpu.copy_d2h(hidden.offset(last_offset), bytes).is_ok() {
                let vals: Vec<f32> = buf
                    .iter()
                    .map(|&b| f32::from_bits((b as u32) << 16))
                    .collect();
                let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                tracing::info!("LAYER_NORM L{i}: hidden_norm={norm:.4}");
                if i == self.layers.len() - 1 {}
            }
        }

        // 2026-09-25: Profile diagnostic: read back the last processed token's
        // hidden state after each layer.
        if diag_prefill {
            self.gpu.synchronize(stream)?;
            let last_start = (proc_count - 1) * h;
            let (last_vals, last_norm) =
                self.readback_bf16(hidden.offset(last_start * fp32), h.min(64))?;
            let last_nan = last_vals.iter().filter(|v| v.is_nan()).count();
            let last_inf = last_vals.iter().filter(|v| v.is_infinite()).count();
            let lt = self.config.layer_type(i);
            if i % 4 == 0 || i == self.layers.len() - 1 || last_nan > 0 || last_inf > 0 {
                tracing::warn!(
                    "DIAG L{i} ({lt:?}) last_tok: norm={last_norm:.4} nan={last_nan} inf={last_inf} first4={:.4?}",
                    &last_vals[..4.min(last_vals.len())]
                );
            }
        }
        Ok(())
    }
}
