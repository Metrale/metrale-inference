// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: BF16 HD64 paged attention with one learned denominator-only sink.
//! Owner: model-layers ops. FP32 online-softmax policy, not staged BF16 parity.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: Explicit paged cache geometry. Callers own buffer extents and valid
/// nonnegative sequence lengths/block indices; no implicit KV dtype or head size.
pub struct PagedSinkGeometry {
    pub sequences: u32,
    pub max_blocks: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub scale: f32,
    pub q_stride: u32,
    pub window: u32,
}

/// 2026-10-07: Launch `paged_decode_attn_sink`, compiled with HDIM=64. Buffers:
/// Q, K-cache, V-cache, output, block tables, sequence lengths, BF16 sinks.
/// Sink is one per query head. Empty sequences output zero; -inf disables the
/// sink; NaN/+inf produces NaN on nonempty sequences. Existing ABI is unchanged.
pub fn paged_decode_attn_bf16_sink(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: [DevicePtr; 7],
    g: &PagedSinkGeometry,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "sink attention: missing kernel");
    ensure!(g.head_dim == 64, "sink attention: requires HDIM=64 kernel");
    ensure!(
        g.sequences > 0 && g.sequences <= 65535,
        "sink attention: invalid sequence grid"
    );
    ensure!(
        g.q_heads > 0 && g.kv_heads > 0 && g.q_heads.is_multiple_of(g.kv_heads),
        "sink attention: invalid GQA ratio"
    );
    ensure!(
        g.max_blocks > 0 && g.block_size > 0,
        "sink attention: empty cache geometry"
    );
    ensure!(
        g.scale.is_finite() && g.scale > 0.0,
        "sink attention: invalid scale"
    );
    let width = g
        .q_heads
        .checked_mul(g.head_dim)
        .ok_or_else(|| anyhow::anyhow!("sink attention: Q width overflow"))?;
    ensure!(
        g.q_stride >= width && g.q_stride.is_multiple_of(2),
        "sink attention: invalid Q stride"
    );
    ensure!(
        g.sequences.checked_mul(g.q_stride).is_some()
            && g.sequences.checked_mul(g.max_blocks).is_some()
            && g.max_blocks.checked_mul(g.block_size).is_some(),
        "sink attention: geometry overflow"
    );
    for (i, ptr) in buffers.iter().enumerate() {
        let alignment = if i == 6 { 2 } else { 4 };
        ensure!(
            ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
            "sink attention: null or misaligned buffer {i}"
        );
    }
    KernelLaunch::new(gpu, kernel)
        .grid([g.q_heads, g.sequences, 1])
        .block([256, 1, 1])
        .arg_ptr(buffers[0])
        .arg_ptr(buffers[1])
        .arg_ptr(buffers[2])
        .arg_ptr(buffers[3])
        .arg_ptr(buffers[4])
        .arg_ptr(buffers[5])
        .arg_u32(g.max_blocks)
        .arg_u32(g.q_heads)
        .arg_u32(g.kv_heads)
        .arg_u32(g.head_dim)
        .arg_u32(g.block_size)
        .arg_f32(g.scale)
        .arg_u32(g.q_stride)
        .arg_u32(g.window)
        .arg_ptr(buffers[6])
        .launch(stream)
}
