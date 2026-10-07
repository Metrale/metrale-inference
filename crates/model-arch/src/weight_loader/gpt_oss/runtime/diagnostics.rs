// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read-only host snapshots of existing live buffers; no diagnostic math path.
use super::*;

/// Owned little-endian bytes copied after a successful forward on its stream.
pub struct DiagnosticTensor {
    pub name: &'static str,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}
impl GptOssLayer {
    /// Inspect the latest completed token before the state is reused or released.
    /// The caller supplies the same backend and stream used by `forward_token`.
    pub fn diagnostic_snapshot(
        &self,
        state: &dyn LayerState,
        position: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<DiagnosticTensor>> {
        let s = state
            .as_any()
            .downcast_ref::<State>()
            .context("GPT state type")?;
        ensure!(
            !s.allocation.is_null() && !s.failed,
            "GPT state is not readable"
        );
        ensure!(
            position.checked_add(1) == Some(s.next_position),
            "GPT snapshot position is stale or unwritten"
        );
        let buffers = [
            ("q_post_rope", s.q, vec![64, 64], "BF16", 2),
            ("k_post_rope", s.k, vec![8, 64], "BF16", 2),
            ("v", s.v, vec![8, 64], "BF16", 2),
            ("attention_pre_o", s.attn, vec![64, 64], "BF16", 2),
            ("attention_post_o", s.projection, vec![2880], "BF16", 2),
            ("post_attention_norm", s.norm, vec![2880], "BF16", 2),
            ("router_logits", s.logits, vec![32], "BF16", 2),
            ("router_scores", s.scores, vec![32], "BF16", 2),
            ("router_ids", s.ids, vec![4], "U32", 4),
            ("selected_experts", s.selected, vec![4, 2880], "BF16", 2),
            ("moe", s.moe, vec![2880], "BF16", 2),
        ];
        buffers
            .into_iter()
            .map(|(name, ptr, shape, dtype, width)| {
                let mut bytes = vec![0; shape.iter().product::<usize>() * width];
                gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
                Ok(DiagnosticTensor {
                    name,
                    dtype,
                    shape,
                    bytes,
                })
            })
            .collect()
    }
}
