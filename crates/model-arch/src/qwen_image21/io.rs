// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native latent/text projections and final adaptive image output.
//! Text encoder outputs are inputs here; this does not implement the encoder.
use super::layout::JointLayout;
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};
use metrale_model_layers::{
    layers::ops::{self, ImageModulationLayout},
    weight_map::DenseWeight,
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
pub struct ProjectionWeights<'a> {
    pub image_in: &'a WeightTensor,
    pub text_norm: &'a WeightTensor,
    pub text_in: &'a WeightTensor,
    pub text_out: &'a WeightTensor,
    pub image_out: &'a WeightTensor,
}
pub struct DiagnosticVisualIo<'a> {
    gpu: &'a dyn GpuBackend,
    layout: &'a JointLayout,
    weights: ProjectionWeights<'a>,
    allocation: DevicePtr,
    text_norm: DevicePtr,
    text_hidden: DevicePtr,
    text_gelu: DevicePtr,
    projected: DevicePtr,
    joint: DevicePtr,
    final_norm: DevicePtr,
    final_scaled: DevicePtr,
    output: DevicePtr,
    target: DevicePtr,
    ones: DevicePtr,
    zeros: DevicePtr,
    gather_ids: DevicePtr,
    target_ids: DevicePtr,
    selection: DevicePtr,
    scale_layout: ImageModulationLayout,
    rms: KernelHandle,
    gelu: KernelHandle,
    gemm: KernelHandle,
    gather: KernelHandle,
    norm: KernelHandle,
    scale: KernelHandle,
}
impl<'a> DiagnosticVisualIo<'a> {
    pub fn new(
        gpu: &'a dyn GpuBackend,
        layout: &'a JointLayout,
        weights: ProjectionWeights<'a>,
    ) -> Result<Self> {
        for (w, shape) in [
            (weights.image_in, [4096, 64]),
            (weights.text_in, [4096, 4096]),
            (weights.text_out, [4096, 4096]),
            (weights.image_out, [64, 4096]),
        ] {
            ensure!(
                w.dtype == WeightDtype::BF16
                    && w.shape == shape
                    && !w.ptr.is_null()
                    && w.ptr.0.is_multiple_of(16),
                "invalid visual projection weight"
            );
        }
        ensure!(
            weights.text_norm.dtype == WeightDtype::BF16
                && weights.text_norm.shape == [4096]
                && !weights.text_norm.ptr.is_null()
                && weights.text_norm.ptr.0.is_multiple_of(4),
            "invalid zero-centered text norm"
        );
        let (b, t, i, j, target) = layout.geometry();
        let text_rows = b * t;
        let joint_rows = b * j;
        let target_rows = b * target;
        ensure!(
            text_rows.checked_mul(4096).is_some() && joint_rows.checked_mul(4096).is_some(),
            "visual projection launch overflow"
        );
        let sizes = [
            text_rows as usize * 8192,
            text_rows as usize * 8192,
            text_rows as usize * 8192,
            (b as usize * (t + i) as usize) * 8192,
            joint_rows as usize * 8192,
            joint_rows as usize * 8192,
            joint_rows as usize * 8192,
            joint_rows as usize * 128,
            target_rows as usize * 128,
            8192,
            8192,
            joint_rows as usize * 4,
            target_rows as usize * 4,
            joint_rows as usize * 4,
        ];
        let mut offsets = Vec::new();
        let mut bytes = 0usize;
        for size in sizes {
            offsets.push(bytes);
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| anyhow::anyhow!("visual IO allocation overflow"))?;
        }
        let rms = gpu.kernel("rms_norm", "rms_norm")?;
        let gelu = gpu.kernel("gelu", "gelu_tanh")?;
        let gemm = gpu.kernel("dense_gemm_bf16", "dense_gemm_bf16_pipelined")?;
        let gather = gpu.kernel("embed_from_argmax", "batched_embed")?;
        let norm = gpu.kernel("nllb_encoder", "nllb_layernorm_oop_bf16")?;
        let scale = gpu.kernel("image_modulation", "image_modulation_scale_bf16")?;
        let scale_layout =
            ImageModulationLayout::new(b, j, 4096, 1, 0, Some(layout.target_mask()))?;
        let allocation = gpu.alloc(bytes)?;
        let value = Self {
            gpu,
            layout,
            weights,
            allocation,
            text_norm: allocation.offset(offsets[0]),
            text_hidden: allocation.offset(offsets[1]),
            text_gelu: allocation.offset(offsets[2]),
            projected: allocation.offset(offsets[3]),
            joint: allocation.offset(offsets[4]),
            final_norm: allocation.offset(offsets[5]),
            final_scaled: allocation.offset(offsets[6]),
            output: allocation.offset(offsets[7]),
            target: allocation.offset(offsets[8]),
            ones: allocation.offset(offsets[9]),
            zeros: allocation.offset(offsets[10]),
            gather_ids: allocation.offset(offsets[11]),
            target_ids: allocation.offset(offsets[12]),
            selection: allocation.offset(offsets[13]),
            scale_layout,
            rms,
            gelu,
            gemm,
            gather,
            norm,
            scale,
        };
        for (ids, ptr) in [
            (layout.projection_gather(), value.gather_ids),
            (layout.target_gather(), value.target_ids),
            (value.scale_layout.selected_rows(), value.selection),
        ] {
            gpu.copy_h2d(
                &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                ptr,
            )?;
        }
        gpu.copy_h2d(
            &(0..4096)
                .flat_map(|_| 0x3f80u16.to_le_bytes())
                .collect::<Vec<_>>(),
            value.ones,
        )?;
        gpu.copy_h2d(&vec![0u8; 8192], value.zeros)?;
        Ok(value)
    }
    /// Input image BF16 `[B,image_tokens,64]` and encoder BF16 `[B,text_tokens,4096]`.
    pub fn input_project(
        &mut self,
        image: DevicePtr,
        text: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            [image, text]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
            "invalid visual input pointers"
        );
        let (b, t, i, j, _) = self.layout.geometry();
        ops::rms_norm(
            self.gpu,
            self.rms,
            text,
            &DenseWeight {
                weight: self.weights.text_norm.ptr,
            },
            self.text_norm,
            b * t,
            4096,
            1e-6,
            stream,
        )?;
        self.linear(
            self.text_norm,
            self.weights.text_in,
            self.text_hidden,
            b * t,
            4096,
            4096,
            stream,
        )?;
        KernelLaunch::new(self.gpu, self.gelu)
            .grid([(b * t * 4096).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.text_hidden)
            .arg_ptr(self.text_gelu)
            .arg_u32(b * t * 4096)
            .launch(stream)?;
        self.linear(
            self.text_gelu,
            self.weights.text_out,
            self.projected,
            b * t,
            4096,
            4096,
            stream,
        )?;
        self.linear(
            image,
            self.weights.image_in,
            self.projected.offset(b as usize * t as usize * 8192),
            b * i,
            4096,
            64,
            stream,
        )?;
        ops::batched_embed(
            self.gpu,
            self.gather,
            self.gather_ids,
            self.projected,
            self.joint,
            b * j,
            4096,
            stream,
        )?;
        Ok(self.joint)
    }
    /// Hidden BF16 `[B,joint_tokens,4096]`; final scale BF16 `[B+1,4096]` from
    /// native time conditioning. Returns only target latents `[B,target_tokens,64]`.
    pub fn output_project(
        &mut self,
        hidden: DevicePtr,
        final_scale: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            [hidden, final_scale]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
            "invalid visual output pointers"
        );
        let (b, _, _, j, target) = self.layout.geometry();
        let rows = b * j;
        KernelLaunch::new(self.gpu, self.norm)
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .shared_mem(1024)
            .arg_ptr(hidden)
            .arg_ptr(self.final_norm)
            .arg_ptr(self.ones)
            .arg_ptr(self.zeros)
            .arg_u32(rows)
            .arg_u32(4096)
            .arg_f32(1e-6)
            .launch(stream)?;
        ops::image_modulation_scale_bf16(
            self.gpu,
            self.scale,
            &self.scale_layout,
            [
                self.final_norm,
                final_scale,
                self.selection,
                self.final_scaled,
            ],
            stream,
        )?;
        self.linear(
            self.final_scaled,
            self.weights.image_out,
            self.output,
            rows,
            64,
            4096,
            stream,
        )?;
        ops::batched_embed(
            self.gpu,
            self.gather,
            self.target_ids,
            self.output,
            self.target,
            b * target,
            64,
            stream,
        )?;
        Ok(self.target)
    }
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        input: DevicePtr,
        weight: &WeightTensor,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemm_bf16_pipelined(
            self.gpu,
            self.gemm,
            input,
            &DenseWeight { weight: weight.ptr },
            output,
            m,
            n,
            k,
            stream,
        )
    }
}
impl Drop for DiagnosticVisualIo<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}
