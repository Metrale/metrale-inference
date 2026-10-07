// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Explicit D1152 FP32 noncausal attention, a diagnostic wide-head residual.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kernel_args::KernelLaunch,
};
/// 2026-10-06: Contiguous QKV `[3,1152,pixels]` to output `[1152,pixels]`.
/// Callers retain device allocations and finite operands. No masks or batches.
pub fn attention_f32(
    gpu: &dyn GpuBackend,
    qkv: DevicePtr,
    output: DevicePtr,
    channels: u32,
    pixels: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        channels == 1152 && (1..=16384).contains(&pixels),
        "unsupported VAE attention geometry"
    );
    ensure!(
        [qkv, output]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
        "invalid VAE attention pointers"
    );
    let bytes = u64::from(channels) * u64::from(pixels) * 4;
    let qkv_end = qkv
        .0
        .checked_add(3 * bytes)
        .ok_or_else(|| anyhow::anyhow!("QKV address overflow"))?;
    let out_end = output
        .0
        .checked_add(bytes)
        .ok_or_else(|| anyhow::anyhow!("output address overflow"))?;
    ensure!(
        qkv_end <= output.0 || out_end <= qkv.0,
        "overlapping VAE attention buffers"
    );
    KernelLaunch::new(
        gpu,
        gpu.kernel("image_vae_attention", "image_vae_attention_f32")?,
    )
    .grid([pixels, 1, 1])
    .block([128, 1, 1])
    .arg_ptr(qkv)
    .arg_ptr(output)
    .arg_u32(pixels)
    .launch(stream)?;
    Ok(())
}
