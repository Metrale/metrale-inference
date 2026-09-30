// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The Mamba-2 h-state guard: a non-finite count in place of the per-head norm
//! clamp, which Mamba-2 models no longer take.
//!
//! Owner: model-engine (SSM state).
//! Invariants: runs where `normalize_ssm_states_dispatch` runs (after every prefill chunk and
//! every 64 decode tokens), on the stream that wrote the state, outside graph capture.

use anyhow::{Result, bail};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-09-29: Count the non-finite values of `seq`'s Mamba-2 h states and fail when
    /// there are any. The state is left as it is: the reference bounds nothing, and the
    /// clamp it replaces rescaled prefill states of real prompts (per-head norms up to
    /// 11275 on Nemotron-3-Nano) to 200.
    pub(super) fn mamba2_state_finite_guard(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        let num_ssm = self.ssm_pool.num_ssm_layers;
        let gpu = self.gpu.as_ref();
        let cache = gpu.op_cache();
        let kernel = cache.kernel(gpu, "ssm_state_norm", "ssm_state_nonfinite_count")?;
        let count = cache.scratch(gpu, "mamba2_nonfinite_count", 4)?;
        let ptrs: Vec<u8> = (0..num_ssm)
            .flat_map(|i| self.ssm_pool.h_state(i, seq.slot_idx).0.to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&ptrs, self.ssm_norm_ptrs_buf, stream)?;
        gpu.memset_async(count, 0, 4, stream)?;
        let (num_heads, k_dim, v_dim) = self.config.ssm_state_norm_dims();
        KernelLaunch::new(gpu, kernel)
            .grid([num_heads as u32, num_ssm as u32, 1])
            .block([v_dim as u32, 1, 1])
            .arg_ptr(self.ssm_norm_ptrs_buf)
            .arg_u32(num_heads as u32)
            .arg_u32(k_dim as u32)
            .arg_u32(v_dim as u32)
            .arg_ptr(count)
            .launch(stream)?;
        let mut host = [0u8; 4];
        gpu.copy_d2h_on_stream(count, &mut host, stream)?;
        let bad = u32::from_le_bytes(host);
        if bad > 0 {
            bail!(
                "Mamba-2 h state of sequence slot {} holds {bad} non-finite values at seq_len {}",
                seq.slot_idx,
                seq.seq_len
            );
        }
        Ok(())
    }
}
