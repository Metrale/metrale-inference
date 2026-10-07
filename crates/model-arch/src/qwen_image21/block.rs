// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Complete diagnostic native block; recorded precision gates fail.
//! This composes real operations, but does not register a model or claim speed.
use super::{DiagnosticImagePrelude, attention::DiagnosticBlockAttention, rope::ImageRopeLayout};
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};
use metrale_model_layers::{
    layers::ops::{self, ImageModulationLayout},
    weight_map::DenseWeight,
};
use metrale_model_weights::{
    qwen_image21::{Block, Config},
    weights::{WeightDtype, WeightTensor},
};

pub struct DiagnosticImageBlock<'a> {
    prelude: DiagnosticImagePrelude<'a>,
    attention: DiagnosticBlockAttention<'a>,
    weights: [&'a WeightTensor; 6],
    allocation: DevicePtr,
    residual: DevicePtr,
    down: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    activated: DevicePtr,
    layouts: [ImageModulationLayout; 3],
    residual_kernel: KernelHandle,
    activation_kernel: KernelHandle,
    image_ids: Vec<i32>,
}
impl<'a> DiagnosticImageBlock<'a> {
    /// Geometry is fixed. IDs describe all image blocks; target-mask separately
    /// selects real-timestep modulation versus the trailing t=0 conditioning row.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &Config,
        weights: &Block<'a, WeightTensor>,
        gpu: &'a dyn GpuBackend,
        samples: u32,
        image_ids: &[i32],
        key_valid: &[bool],
        target_mask: Option<&[bool]>,
    ) -> Result<Self> {
        ensure!(
            config.intermediate() == 12288,
            "unsupported image FFN width"
        );
        for (weight, shape) in [
            (weights.o, [4096, 4096]),
            (weights.gate, [12288, 4096]),
            (weights.up, [12288, 4096]),
            (weights.down, [4096, 12288]),
        ] {
            ensure!(
                weight.dtype == WeightDtype::BF16
                    && weight.shape == shape
                    && !weight.ptr.is_null()
                    && weight.ptr.0.is_multiple_of(16),
                "invalid image block matrix"
            );
        }
        for norm in [weights.q_norm, weights.k_norm] {
            ensure!(
                norm.dtype == WeightDtype::BF16
                    && norm.shape == [128]
                    && !norm.ptr.is_null()
                    && norm.ptr.0.is_multiple_of(4),
                "invalid image block head norm"
            );
        }
        let tokens = u32::try_from(image_ids.len())?;
        let prelude =
            DiagnosticImagePrelude::new(config, weights, gpu, samples, tokens, target_mask)?;
        let attention = DiagnosticBlockAttention::new(gpu, samples, image_ids, key_valid)?;
        let rows = prelude.rows;
        rows.checked_mul(12288)
            .ok_or_else(|| anyhow::anyhow!("image activation count overflow"))?;
        let row_bytes = rows as usize * 8192;
        let bytes = row_bytes
            .checked_mul(11)
            .ok_or_else(|| anyhow::anyhow!("image block allocation overflow"))?;
        let layouts = [
            ImageModulationLayout::new(samples, tokens, 4096, 4, 1, target_mask)?,
            ImageModulationLayout::new(samples, tokens, 4096, 4, 2, target_mask)?,
            ImageModulationLayout::new(samples, tokens, 4096, 4, 3, target_mask)?,
        ];
        let residual_kernel = gpu.kernel("image_modulation", "image_modulation_residual_bf16")?;
        let activation_kernel = gpu.kernel("image_modulation", "image_silu_staged_mul_bf16")?;
        let allocation = gpu.alloc(bytes)?;
        Ok(Self {
            prelude,
            attention,
            weights: [
                weights.q_norm,
                weights.k_norm,
                weights.o,
                weights.gate,
                weights.up,
                weights.down,
            ],
            allocation,
            residual: allocation,
            down: allocation.offset(row_bytes),
            gate: allocation.offset(2 * row_bytes),
            up: allocation.offset(5 * row_bytes),
            activated: allocation.offset(8 * row_bytes),
            layouts,
            residual_kernel,
            activation_kernel,
            image_ids: image_ids.to_vec(),
        })
    }
    /// BF16 hidden `[samples,tokens,4096]`, modulation `[samples(+1),16384]`.
    /// Output is owned/overwritten by this block; caller preserves input storage
    /// through the stream. No denoising cache or model registration is implied.
    pub fn forward(
        &mut self,
        hidden: DevicePtr,
        modulation: DevicePtr,
        rope: &ImageRopeLayout,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            rope.matches_image_ids(&self.image_ids),
            "rotary image blocks differ from attention metadata"
        );
        self.prelude.project(hidden, modulation, stream)?;
        self.prelude
            .normalize_qk(self.weights[0], self.weights[1], stream)?;
        self.prelude.rotate_qk(rope, stream)?;
        let branch = self
            .prelude
            .attend_project(&mut self.attention, self.weights[2], stream)?;
        let p = &self.prelude;
        ops::image_modulation_residual_bf16(
            p.gpu,
            self.residual_kernel,
            &self.layouts[0],
            [hidden, branch, modulation, p.selection, self.residual],
            stream,
        )?;
        KernelLaunch::new(p.gpu, p.norm)
            .grid([p.rows, 1, 1])
            .block([256, 1, 1])
            .shared_mem(1024)
            .arg_ptr(self.residual)
            .arg_ptr(p.normalized)
            .arg_ptr(p.ones)
            .arg_ptr(p.zeros)
            .arg_u32(p.rows)
            .arg_u32(4096)
            .arg_f32(1e-6)
            .launch(stream)?;
        ops::image_modulation_scale_bf16(
            p.gpu,
            p.scale,
            &self.layouts[1],
            [p.normalized, modulation, p.selection, p.modulated],
            stream,
        )?;
        for (weight, output) in [(self.weights[3], self.gate), (self.weights[4], self.up)] {
            ops::dense_gemm_bf16_pipelined(
                p.gpu,
                p.gemm,
                p.modulated,
                &DenseWeight { weight: weight.ptr },
                output,
                p.rows,
                12288,
                4096,
                stream,
            )?;
        }
        ops::silu_mul(
            p.gpu,
            self.activation_kernel,
            self.gate,
            self.up,
            self.activated,
            p.rows * 12288,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            p.gpu,
            p.gemm,
            self.activated,
            &DenseWeight {
                weight: self.weights[5].ptr,
            },
            self.down,
            p.rows,
            4096,
            12288,
            stream,
        )?;
        ops::image_modulation_residual_bf16(
            p.gpu,
            self.residual_kernel,
            &self.layouts[2],
            [
                self.residual,
                self.down,
                modulation,
                p.selection,
                self.residual,
            ],
            stream,
        )?;
        Ok(self.residual)
    }
}
impl Drop for DiagnosticImageBlock<'_> {
    fn drop(&mut self) {
        let _ = self.prelude.gpu.free(self.allocation);
    }
}
