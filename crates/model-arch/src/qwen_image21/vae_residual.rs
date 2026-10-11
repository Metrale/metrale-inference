// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Original-FP32 single-frame VAE residual block composition.
//! Reuses existing FP32 residual addition; no decoder/model registration.
use super::vae::{DiagnosticVaeConv, ImageShape, channel_norm_f32, silu_f32};
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kernel_args::KernelLaunch,
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};

pub struct ResidualWeights<'a> {
    pub norm1: &'a WeightTensor,
    pub conv1: &'a WeightTensor,
    pub bias1: &'a WeightTensor,
    pub norm2: &'a WeightTensor,
    pub conv2: &'a WeightTensor,
    pub bias2: &'a WeightTensor,
    pub shortcut: Option<(&'a WeightTensor, &'a WeightTensor)>,
}
pub struct DiagnosticVaeResidual<'a> {
    gpu: &'a dyn GpuBackend,
    input: ImageShape,
    output: ImageShape,
    norm1: &'a WeightTensor,
    norm2: &'a WeightTensor,
    conv1: DiagnosticVaeConv<'a>,
    conv2: DiagnosticVaeConv<'a>,
    shortcut: Option<DiagnosticVaeConv<'a>>,
    scratch: DevicePtr,
}
fn gamma(weight: &WeightTensor, channels: u32) -> bool {
    weight.dtype == WeightDtype::FP32
        && weight.shape == [channels as usize, 1, 1, 1]
        && !weight.ptr.is_null()
        && weight.ptr.0.is_multiple_of(4)
}
impl<'a> DiagnosticVaeResidual<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, input: ImageShape, w: ResidualWeights<'a>) -> Result<Self> {
        ensure!(
            w.conv1.shape.len() == 4 && w.conv1.shape[2..] == [3, 3],
            "VAE residual conv1 kernel differs"
        );
        let conv1 = DiagnosticVaeConv::new(gpu, input, w.conv1, w.bias1)?;
        let output = conv1.output_shape();
        ensure!(
            w.conv2.shape
                == [
                    output.dimensions()[0] as usize,
                    output.dimensions()[0] as usize,
                    3,
                    3
                ],
            "VAE residual conv2 differs"
        );
        ensure!(
            gamma(w.norm1, input.dimensions()[0]) && gamma(w.norm2, output.dimensions()[0]),
            "VAE residual gamma differs"
        );
        let conv2 = DiagnosticVaeConv::new(gpu, output, w.conv2, w.bias2)?;
        let shortcut = if input.dimensions()[0] == output.dimensions()[0] {
            ensure!(
                w.shortcut.is_none(),
                "unexpected VAE identity shortcut weights"
            );
            None
        } else {
            let (weight, bias) = w
                .shortcut
                .ok_or_else(|| anyhow::anyhow!("missing VAE shortcut"))?;
            ensure!(
                weight.shape
                    == [
                        output.dimensions()[0] as usize,
                        input.dimensions()[0] as usize,
                        1,
                        1
                    ],
                "VAE shortcut shape differs"
            );
            Some(DiagnosticVaeConv::new(gpu, input, weight, bias)?)
        };
        let scratch = gpu.alloc(input.elements().max(output.elements()) as usize * 4)?;
        Ok(Self {
            gpu,
            input,
            output,
            norm1: w.norm1,
            norm2: w.norm2,
            conv1,
            conv2,
            shortcut,
            scratch,
        })
    }
    /// Single contiguous FP32 frame, no dropout or temporal cache. Returned
    /// storage belongs to this block and remains valid until its next call/drop.
    pub fn forward(&mut self, input: DevicePtr, stream: u64) -> Result<DevicePtr> {
        ensure!(
            !input.is_null() && input.0.is_multiple_of(4),
            "invalid VAE residual input"
        );
        let count = self.input.elements();
        let end = input
            .0
            .checked_add(u64::from(count) * 4)
            .ok_or_else(|| anyhow::anyhow!("input address overflow"))?;
        let scratch_end = self
            .scratch
            .0
            .checked_add(u64::from(count.max(self.output.elements())) * 4)
            .ok_or_else(|| anyhow::anyhow!("scratch address overflow"))?;
        ensure!(
            (end <= self.scratch.0 || input.0 >= scratch_end)
                && !self.conv1.overlaps_output(input, count)
                && !self.conv2.overlaps_output(input, count)
                && self
                    .shortcut
                    .as_ref()
                    .is_none_or(|c| !c.overlaps_output(input, count)),
            "VAE residual input overlaps owned storage"
        );
        let shortcut = if let Some(conv) = self.shortcut.as_mut() {
            conv.forward(input, stream)?
        } else {
            input
        };
        channel_norm_f32(
            self.gpu,
            self.input,
            input,
            self.norm1,
            self.scratch,
            stream,
        )?;
        silu_f32(self.gpu, self.input, self.scratch, self.scratch, stream)?;
        let x = self.conv1.forward(self.scratch, stream)?;
        channel_norm_f32(self.gpu, self.output, x, self.norm2, self.scratch, stream)?;
        silu_f32(self.gpu, self.output, self.scratch, self.scratch, stream)?;
        let result = self.conv2.forward(self.scratch, stream)?;
        add_f32(self.gpu, self.output, result, shortcut, stream)?;
        Ok(result)
    }
    pub fn output_shape(&self) -> ImageShape {
        self.output
    }
}
impl Drop for DiagnosticVaeResidual<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.scratch);
    }
}
/// Existing FP32 residual entry supports in-place destination explicitly.
pub fn add_f32(
    gpu: &dyn GpuBackend,
    shape: ImageShape,
    destination: DevicePtr,
    source: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        [destination, source]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
        "invalid FP32 residual pointers"
    );
    KernelLaunch::new(gpu, gpu.kernel("nllb_encoder", "nllb_add_inplace")?)
        .grid([shape.elements().div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(destination)
        .arg_ptr(source)
        .arg_u32(shape.elements())
        .launch(stream)
}
