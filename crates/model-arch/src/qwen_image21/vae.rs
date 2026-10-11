// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Single-frame decoder FP32 operations; no VAE/pipeline registration.
//! Original checkpoint precision is retained. Earlier BF16 reference image runs
//! are a different numerical policy and are not this component's exact oracle.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
#[derive(Clone, Copy, Debug)]
pub struct ImageShape {
    channels: u32,
    height: u32,
    width: u32,
    elements: u32,
}
impl ImageShape {
    pub fn new(channels: u32, height: u32, width: u32) -> Result<Self> {
        ensure!(
            channels > 0 && channels <= 4096 && height > 0 && width > 0,
            "invalid VAE image shape"
        );
        let elements = channels
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .filter(|n| *n <= i32::MAX as u32)
            .ok_or_else(|| anyhow::anyhow!("VAE image shape overflow"))?;
        Ok(Self {
            channels,
            height,
            width,
            elements,
        })
    }
    pub fn dimensions(&self) -> [u32; 3] {
        [self.channels, self.height, self.width]
    }
    pub fn elements(&self) -> u32 {
        self.elements
    }
}
pub struct DiagnosticVaeConv<'a> {
    gpu: &'a dyn GpuBackend,
    input: ImageShape,
    output: ImageShape,
    weight: &'a WeightTensor,
    bias: &'a WeightTensor,
    kernel_size: u32,
    kernel: KernelHandle,
    allocation: DevicePtr,
}
impl<'a> DiagnosticVaeConv<'a> {
    pub fn new(
        gpu: &'a dyn GpuBackend,
        input: ImageShape,
        weight: &'a WeightTensor,
        bias: &'a WeightTensor,
    ) -> Result<Self> {
        ensure!(
            weight.dtype == WeightDtype::FP32
                && weight.shape.len() == 4
                && weight.shape[1] == input.channels as usize
                && weight.shape[2] == weight.shape[3]
                && [1, 3].contains(&weight.shape[2]),
            "unsupported VAE convolution weight"
        );
        let output = ImageShape::new(u32::try_from(weight.shape[0])?, input.height, input.width)?;
        ensure!(
            bias.dtype == WeightDtype::FP32
                && bias.shape == [output.channels as usize]
                && [weight.ptr, bias.ptr]
                    .iter()
                    .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
            "invalid VAE convolution bias/pointers"
        );
        let kernel = gpu.kernel("image_vae", "image_vae_conv2d_f32")?;
        let allocation = gpu.alloc(output.elements as usize * 4)?;
        Ok(Self {
            gpu,
            input,
            output,
            weight,
            bias,
            kernel_size: u32::try_from(weight.shape[2])?,
            kernel,
            allocation,
        })
    }
    /// Input is contiguous FP32 `[C,H,W]`; output has the same spatial size.
    /// This slow diagnostic supports only stride1, odd kernel1/3, symmetric pad.
    pub fn forward(&mut self, input: DevicePtr, stream: u64) -> Result<DevicePtr> {
        ensure!(
            !input.is_null() && input.0.is_multiple_of(4) && input != self.allocation,
            "invalid or aliasing VAE input"
        );
        let input_end = input
            .0
            .checked_add(u64::from(self.input.elements) * 4)
            .ok_or_else(|| anyhow::anyhow!("VAE input address overflow"))?;
        let output_end = self
            .allocation
            .0
            .checked_add(u64::from(self.output.elements) * 4)
            .ok_or_else(|| anyhow::anyhow!("VAE output address overflow"))?;
        ensure!(
            input_end <= self.allocation.0 || input.0 >= output_end,
            "overlapping VAE convolution input/output"
        );
        KernelLaunch::new(self.gpu, self.kernel)
            .grid([self.output.elements.div_ceil(128), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(self.weight.ptr)
            .arg_ptr(self.bias.ptr)
            .arg_ptr(self.allocation)
            .arg_u32(self.input.channels)
            .arg_u32(self.output.channels)
            .arg_u32(self.input.height)
            .arg_u32(self.input.width)
            .arg_u32(self.kernel_size)
            .launch(stream)?;
        Ok(self.allocation)
    }
    pub fn output_shape(&self) -> ImageShape {
        self.output
    }
    pub(crate) fn overlaps_output(&self, input: DevicePtr, elements: u32) -> bool {
        let Some(end) = input.0.checked_add(u64::from(elements) * 4) else {
            return true;
        };
        let Some(out_end) = self
            .allocation
            .0
            .checked_add(u64::from(self.output.elements) * 4)
        else {
            return true;
        };
        input.0 < out_end && self.allocation.0 < end
    }
}
impl Drop for DiagnosticVaeConv<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}
/// Gamma must be the pinned channel-wise affine tensor. No epsilon under sqrt:
/// source uses L2 clamp, which has a different tiny-input contract from RMSNorm.
pub fn channel_norm_f32(
    gpu: &dyn GpuBackend,
    shape: ImageShape,
    input: DevicePtr,
    gamma: &WeightTensor,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        gamma.dtype == WeightDtype::FP32
            && [
                vec![shape.channels as usize, 1, 1],
                vec![shape.channels as usize, 1, 1, 1]
            ]
            .contains(&gamma.shape),
        "invalid VAE channel gamma"
    );
    ensure!(
        [input, gamma.ptr, output]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
        "invalid VAE norm pointer"
    );
    KernelLaunch::new(gpu, gpu.kernel("image_vae", "image_vae_norm_f32")?)
        .grid([shape.height * shape.width, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gamma.ptr)
        .arg_ptr(output)
        .arg_u32(shape.channels)
        .arg_u32(shape.height * shape.width)
        .launch(stream)
}
pub fn silu_f32(
    gpu: &dyn GpuBackend,
    shape: ImageShape,
    input: DevicePtr,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        [input, output]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
        "invalid VAE activation pointers"
    );
    KernelLaunch::new(gpu, gpu.kernel("image_vae", "image_vae_silu_f32")?)
        .grid([shape.elements.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(shape.elements)
        .launch(stream)
}
