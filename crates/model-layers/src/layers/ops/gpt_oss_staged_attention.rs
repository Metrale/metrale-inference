// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit staged-BF16 correctness policy; does not replace online FP32 attention.
use super::paged_sink::{PagedSinkGeometry, paged_decode_attn_bf16_sink};
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// 2026-10-06: Finite Q/K/V operands only; nonfinite masked-operand propagation is unqualified.
/// Existing paged-sink geometry and buffers, with a device-enforced 4096-token cap.
/// The caller must provide the `gpt_oss_staged_attention_bf16` kernel handle.
/// Above-cap lengths return NaN; this is an eager correctness policy, not a fast path.
pub fn gpt_oss_staged_attention_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: [DevicePtr; 7],
    geometry: &PagedSinkGeometry,
    stream: u64,
) -> Result<()> {
    ensure!(
        geometry.head_dim == 64,
        "staged sink attention requires HD64"
    );
    ensure!(
        geometry
            .max_blocks
            .checked_mul(geometry.block_size)
            .is_some_and(|n| n <= 4096),
        "staged sink attention capacity exceeds 4096-token correctness bound"
    );
    paged_decode_attn_bf16_sink(gpu, kernel, buffers, geometry, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    #[test]
    fn bounded_policy_reuses_abi_and_refuses_oversized_capacity() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu
            .kernel("gpt_oss_staged_attention", "gpt_oss_staged_attention_bf16")
            .unwrap();
        let buffers = [DevicePtr(4096); 7];
        let mut g = PagedSinkGeometry {
            sequences: 1,
            max_blocks: 256,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            block_size: 16,
            scale: 0.125,
            q_stride: 4096,
            window: 128,
        };
        gpt_oss_staged_attention_bf16(&gpu, kernel, buffers, &g, 0).unwrap();
        g.max_blocks = 257;
        assert!(gpt_oss_staged_attention_bf16(&gpu, kernel, buffers, &g, 0).is_err());
        g.max_blocks = u32::MAX;
        assert!(gpt_oss_staged_attention_bf16(&gpu, kernel, buffers, &g, 0).is_err());
        g.max_blocks = 1;
        g.head_dim = 128;
        assert!(gpt_oss_staged_attention_bf16(&gpu, kernel, buffers, &g, 0).is_err());
    }
}
