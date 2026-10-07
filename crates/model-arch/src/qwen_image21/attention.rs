// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Slow diagnostic block-causal attention using unchanged kernels.
//! Stable valid-key gathers reduce each query's mask to one contiguous prefix.
//! This is O(sequence) launches, not an image-serving performance path.
use super::ImageQkvBuffers;
use anyhow::{Result, ensure};
use metrale_circuit::image_attention::{Layout, PackedPrefixes, Token};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};
use metrale_model_layers::layers::ops;

/// Owns compact K/V, attention output and immutable gather indices. Inputs must
/// remain alive through the stream; output remains owned until this is dropped.
pub struct DiagnosticBlockAttention<'a> {
    gpu: &'a dyn GpuBackend,
    allocation: DevicePtr,
    compact_k: DevicePtr,
    compact_v: DevicePtr,
    output: DevicePtr,
    indices: DevicePtr,
    plan: PackedPrefixes,
    gather_kernel: KernelHandle,
    attention_kernel: KernelHandle,
    samples: u32,
    tokens: u32,
}
impl<'a> DiagnosticBlockAttention<'a> {
    /// Image IDs are shared across samples: -1 is text, nonnegative is a block.
    /// Key validity is sample-major. Padding removes keys, never query rows.
    pub fn new(
        gpu: &'a dyn GpuBackend,
        samples: u32,
        image_ids: &[i32],
        key_valid: &[bool],
    ) -> Result<Self> {
        ensure!(
            samples > 0 && !image_ids.is_empty() && image_ids.len() <= 8192,
            "diagnostic attention requires 1..8192 tokens and nonempty batch"
        );
        let tokens = u32::try_from(image_ids.len())?;
        let rows = samples
            .checked_mul(tokens)
            .ok_or_else(|| anyhow::anyhow!("attention row overflow"))?;
        ensure!(
            rows <= i32::MAX as u32 && key_valid.len() == rows as usize,
            "attention validity shape differs"
        );
        ensure!(
            image_ids.iter().all(|id| *id >= -1),
            "invalid negative image ID"
        );
        let mut metadata = Vec::with_capacity(rows as usize);
        for sample in 0..samples {
            for (position, id) in image_ids.iter().enumerate() {
                metadata.push(Token {
                    sample,
                    image: (*id >= 0).then_some(*id as u32),
                    key_valid: key_valid[sample as usize * tokens as usize + position],
                });
            }
        }
        let plan = Layout::new(metadata)?.packed_prefixes()?;
        let row_bytes = (rows as usize)
            .checked_mul(4096 * 2)
            .ok_or_else(|| anyhow::anyhow!("attention storage overflow"))?;
        let bytes = row_bytes
            .checked_mul(3)
            .and_then(|n| n.checked_add(rows as usize * 4))
            .ok_or_else(|| anyhow::anyhow!("attention allocation overflow"))?;
        let gather_kernel = gpu.kernel("embed_from_argmax", "batched_embed")?;
        let attention_kernel = gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?;
        let allocation = gpu.alloc(bytes)?;
        let result = Self {
            gpu,
            allocation,
            compact_k: allocation,
            compact_v: allocation.offset(row_bytes),
            output: allocation.offset(2 * row_bytes),
            indices: allocation.offset(3 * row_bytes),
            plan,
            gather_kernel,
            attention_kernel,
            samples,
            tokens,
        };
        let indices: Vec<u8> = result
            .plan
            .gather
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        if !indices.is_empty() {
            gpu.copy_h2d(&indices, result.indices)?;
        }
        // Fully masked queries never launch attention, and remain exactly zero.
        gpu.copy_h2d(&vec![0u8; row_bytes], result.output)?;
        Ok(result)
    }
    pub fn geometry(&self) -> (u32, u32) {
        (self.samples, self.tokens)
    }
    pub(super) fn uses_backend(&self, gpu: &dyn GpuBackend) -> bool {
        std::ptr::addr_eq(self.gpu, gpu)
    }
    /// Caller supplies BF16 `[samples,tokens,32,128]` Q/K/V; Q/K must already
    /// have their intended normalization/rotation. No prefix KV cache is used.
    pub fn forward(&mut self, qkv: ImageQkvBuffers, stream: u64) -> Result<DevicePtr> {
        ensure!(
            [qkv.q, qkv.k, qkv.v]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(2)),
            "invalid attention input pointer"
        );
        if !self.plan.gather.is_empty() {
            for (input, output) in [(qkv.k, self.compact_k), (qkv.v, self.compact_v)] {
                ops::batched_embed(
                    self.gpu,
                    self.gather_kernel,
                    self.indices,
                    input,
                    output,
                    self.plan.gather.len() as u32,
                    4096,
                    stream,
                )?;
            }
        }
        for (query, &(start, count)) in self.plan.spans.iter().enumerate() {
            if count == 0 {
                continue;
            }
            KernelLaunch::new(self.gpu, self.attention_kernel)
                .grid([32, 1, 1])
                .block([128, 1, 1])
                .shared_mem((count + 128) * 4)
                .arg_ptr(qkv.q.offset(query * 8192))
                .arg_ptr(self.compact_k.offset(start as usize * 8192))
                .arg_ptr(self.compact_v.offset(start as usize * 8192))
                .arg_ptr(self.output.offset(query * 8192))
                .arg_u32(1)
                .arg_u32(count)
                .arg_u32(32)
                .arg_u32(128)
                .arg_f32(1.0 / 128.0f32.sqrt())
                .arg_u32(0)
                .launch(stream)?;
        }
        Ok(self.output)
    }
}
impl Drop for DiagnosticBlockAttention<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}
