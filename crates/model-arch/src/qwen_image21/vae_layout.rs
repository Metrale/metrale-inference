// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Exact first-frame spatial upsampling and channel-shuffle layout.
use super::vae::ImageShape;
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kernel_args::KernelLaunch,
};
/// New spatial dimensions are exactly twice the input. `temporal_factor=1`
/// with equal channels is nearest-exact; other points implement DupUp3D's
/// repeat-interleave/pixel-shuffle and first-chunk temporal selection.
pub fn upsample_shape(
    input: ImageShape,
    channels: u32,
    temporal_factor: u32,
) -> Result<ImageShape> {
    let [ic, h, w] = input.dimensions();
    ensure!(
        [1, 2].contains(&temporal_factor) && channels > 0 && channels <= 4096,
        "invalid VAE upsample parameters"
    );
    ensure!(
        (channels * temporal_factor * 4).is_multiple_of(ic),
        "fractional VAE channel repeat"
    );
    ImageShape::new(
        channels,
        h.checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("height overflow"))?,
        w.checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("width overflow"))?,
    )
}
pub fn upsample_f32(
    gpu: &dyn GpuBackend,
    input_shape: ImageShape,
    channels: u32,
    temporal_factor: u32,
    input: DevicePtr,
    output: DevicePtr,
    stream: u64,
) -> Result<ImageShape> {
    let shape = upsample_shape(input_shape, channels, temporal_factor)?;
    ensure!(
        [input, output]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
        "invalid upsample pointers"
    );
    let end_i = input
        .0
        .checked_add(u64::from(input_shape.elements()) * 4)
        .ok_or_else(|| anyhow::anyhow!("input address overflow"))?;
    let end_o = output
        .0
        .checked_add(u64::from(shape.elements()) * 4)
        .ok_or_else(|| anyhow::anyhow!("output address overflow"))?;
    ensure!(
        end_i <= output.0 || end_o <= input.0,
        "overlapping upsample buffers"
    );
    let [ic, h, w] = input_shape.dimensions();
    KernelLaunch::new(gpu, gpu.kernel("image_vae", "image_vae_upsample_f32")?)
        .grid([shape.elements().div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(ic)
        .arg_u32(channels)
        .arg_u32(h)
        .arg_u32(w)
        .arg_u32(temporal_factor)
        .launch(stream)?;
    Ok(shape)
}
