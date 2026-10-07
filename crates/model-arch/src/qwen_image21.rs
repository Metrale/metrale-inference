// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Diagnostic native visual-block prelude; not a serving model.
//! LayerNorm uses the existing two-pass NLLB kernel, whose exact-reference gate
//! has three recorded near-zero BF16 mismatches. This path does not qualify it.
pub mod attention;
pub mod block;
pub mod conditioning;
pub mod io;
pub mod layout;
pub mod rope;
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

/// Outputs remain owned by the prelude and are overwritten on the next call.
/// The producing method states whether Q/K normalization has run. No rotary
/// transform, attention or KV reuse is implied.
#[derive(Clone, Copy)]
pub struct ImageQkvBuffers {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
}

#[derive(Clone, Copy, PartialEq)]
enum QkvStage {
    Empty,
    Raw,
    Normalized,
    Rotated,
    Attended,
}

/// Fixed-shape diagnostic composition of non-affine LayerNorm, exact native
/// scale modulation and tensor-core BF16 Q/K/V projections. Keep weights and
/// GPU alive; never use this as an already-qualified transformer block.
pub struct DiagnosticImagePrelude<'a> {
    gpu: &'a dyn GpuBackend,
    weights: [&'a WeightTensor; 3],
    allocation: DevicePtr,
    normalized: DevicePtr,
    modulated: DevicePtr,
    qkv: ImageQkvBuffers,
    ones: DevicePtr,
    zeros: DevicePtr,
    selection: DevicePtr,
    rows: u32,
    samples: u32,
    tokens: u32,
    stage: QkvStage,
    layout: ImageModulationLayout,
    norm: KernelHandle,
    scale: KernelHandle,
    gemm: KernelHandle,
}
impl<'a> DiagnosticImagePrelude<'a> {
    /// Allocate scratch for explicit sample/token geometry. `block` must refer to
    /// the validated pinned transformer component and retain its active storage.
    pub fn new(
        config: &Config,
        block: &Block<'a, WeightTensor>,
        gpu: &'a dyn GpuBackend,
        samples: u32,
        tokens: u32,
        target_mask: Option<&[bool]>,
    ) -> Result<Self> {
        let width = config.hidden();
        ensure!(width == 4096, "Qwen prelude requires pinned hidden width");
        let weights = [block.q, block.k, block.v];
        for w in weights {
            ensure!(
                w.dtype == WeightDtype::BF16
                    && w.shape == [width, width]
                    && !w.ptr.is_null()
                    && w.ptr.0.is_multiple_of(16),
                "invalid Qwen projection weight"
            );
        }
        let layout = ImageModulationLayout::new(samples, tokens, width as u32, 4, 0, target_mask)?;
        let rows = u32::try_from(layout.selected_rows().len())?;
        let row_bytes = (rows as usize)
            .checked_mul(width * 2)
            .ok_or_else(|| anyhow::anyhow!("Qwen scratch overflow"))?;
        let bytes = row_bytes
            .checked_mul(5)
            .and_then(|n| n.checked_add(width * 4 + rows as usize * 4))
            .ok_or_else(|| anyhow::anyhow!("Qwen scratch overflow"))?;
        let norm = gpu.kernel("nllb_encoder", "nllb_layernorm_oop_bf16")?;
        let scale = gpu.kernel("image_modulation", "image_modulation_scale_bf16")?;
        let gemm = gpu.kernel("dense_gemm_bf16", "dense_gemm_bf16_pipelined")?;
        let allocation = gpu.alloc(bytes)?;
        let result = Self {
            gpu,
            weights,
            allocation,
            normalized: allocation,
            modulated: allocation.offset(row_bytes),
            qkv: ImageQkvBuffers {
                q: allocation.offset(2 * row_bytes),
                k: allocation.offset(3 * row_bytes),
                v: allocation.offset(4 * row_bytes),
            },
            ones: allocation.offset(5 * row_bytes),
            zeros: allocation.offset(5 * row_bytes + width * 2),
            selection: allocation.offset(5 * row_bytes + width * 4),
            rows,
            samples,
            tokens,
            stage: QkvStage::Empty,
            layout,
            norm,
            scale,
            gemm,
        };
        let ones: Vec<u8> = (0..width).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
        gpu.copy_h2d(&ones, result.ones)?;
        gpu.copy_h2d(&vec![0u8; width * 2], result.zeros)?;
        let selected: Vec<u8> = result
            .layout
            .selected_rows()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        gpu.copy_h2d(&selected, result.selection)?;
        Ok(result)
    }
    /// Caller provides BF16 `hidden[rows,4096]` and `modulation[samples(+1),16384]`
    /// with sizes from construction. The selection map is owned and cannot drift.
    /// Enqueues work on stream. Norm discrepancy remains explicit and unqualified.
    pub fn project(
        &mut self,
        hidden: DevicePtr,
        modulation: DevicePtr,
        stream: u64,
    ) -> Result<ImageQkvBuffers> {
        ensure!(
            !hidden.is_null()
                && hidden.0.is_multiple_of(16)
                && !modulation.is_null()
                && modulation.0.is_multiple_of(2),
            "invalid Qwen prelude input pointers"
        );
        self.stage = QkvStage::Empty;
        KernelLaunch::new(self.gpu, self.norm)
            .grid([self.rows, 1, 1])
            .block([256, 1, 1])
            .shared_mem(1024)
            .arg_ptr(hidden)
            .arg_ptr(self.normalized)
            .arg_ptr(self.ones)
            .arg_ptr(self.zeros)
            .arg_u32(self.rows)
            .arg_u32(4096)
            .arg_f32(1e-6)
            .launch(stream)?;
        ops::image_modulation_scale_bf16(
            self.gpu,
            self.scale,
            &self.layout,
            [self.normalized, modulation, self.selection, self.modulated],
            stream,
        )?;
        for (weight, out) in self
            .weights
            .iter()
            .zip([self.qkv.q, self.qkv.k, self.qkv.v])
        {
            ops::dense_gemm_bf16_pipelined(
                self.gpu,
                self.gemm,
                self.modulated,
                &DenseWeight { weight: weight.ptr },
                out,
                self.rows,
                4096,
                4096,
                stream,
            )?;
        }
        self.stage = QkvStage::Raw;
        Ok(self.qkv)
    }
    /// 2026-10-07: Normalize each 128-wide head, rounding its activation to BF16
    /// before the separate BF16 weight multiplication. Must follow project once.
    /// Returned Q/K have no rotary transform yet; V is untouched.
    pub fn normalize_qk(
        &mut self,
        q_weight: &WeightTensor,
        k_weight: &WeightTensor,
        stream: u64,
    ) -> Result<ImageQkvBuffers> {
        ensure!(
            self.stage == QkvStage::Raw,
            "Qwen QK normalization requires fresh raw projections"
        );
        for weight in [q_weight, k_weight] {
            ensure!(
                weight.dtype == WeightDtype::BF16
                    && weight.shape == [128]
                    && !weight.ptr.is_null()
                    && weight.ptr.0.is_multiple_of(4),
                "invalid Qwen head norm weight"
            );
        }
        let rows = self
            .rows
            .checked_mul(32)
            .ok_or_else(|| anyhow::anyhow!("Qwen head row count overflow"))?;
        let norm = self.gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
        let apply = self
            .gpu
            .kernel("image_modulation", "image_head_weight_bf16")?;
        self.stage = QkvStage::Empty;
        for (input, weight) in [(self.qkv.q, q_weight), (self.qkv.k, k_weight)] {
            ops::rms_norm(
                self.gpu,
                norm,
                input,
                &DenseWeight { weight: self.ones },
                self.normalized,
                rows,
                128,
                1e-6,
                stream,
            )?;
            ops::image_head_weight_bf16(
                self.gpu,
                apply,
                [self.normalized, weight.ptr, input],
                rows,
                stream,
            )?;
        }
        self.stage = QkvStage::Normalized;
        Ok(self.qkv)
    }
    /// 2026-10-07: Apply native three-axis complex RoPE after staged Q/K norm.
    /// Geometry must describe the exact projected sequence shared by samples.
    pub fn rotate_qk(
        &mut self,
        layout: &rope::ImageRopeLayout,
        stream: u64,
    ) -> Result<ImageQkvBuffers> {
        ensure!(
            self.stage == QkvStage::Normalized,
            "image RoPE requires normalized fresh QK"
        );
        ensure!(
            layout.positions().len() == self.tokens as usize,
            "image rotary sequence differs from projections"
        );
        let kernel = self
            .gpu
            .kernel("image_modulation", "image_rope_complex_bf16")?;
        let table = layout.frequencies();
        let bytes: Vec<u8> = table.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, self.normalized, stream)?;
        self.stage = QkvStage::Empty;
        for qk in [self.qkv.q, self.qkv.k] {
            ops::image_rope_complex_bf16(
                self.gpu,
                kernel,
                [qk, self.normalized, qk],
                self.samples,
                self.tokens,
                stream,
            )?;
        }
        self.stage = QkvStage::Rotated;
        Ok(self.qkv)
    }

    /// 2026-10-07: Slow diagnostic block-causal attention and output projection.
    /// Prior LayerNorm/cis discrepancies remain; this does not qualify a block.
    pub fn attend_project(
        &mut self,
        attention: &mut attention::DiagnosticBlockAttention<'_>,
        weight: &WeightTensor,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            self.stage == QkvStage::Rotated,
            "attention requires fresh rotated QK"
        );
        ensure!(
            attention.geometry() == (self.samples, self.tokens),
            "attention geometry differs"
        );
        ensure!(
            attention.uses_backend(self.gpu),
            "attention backend differs"
        );
        ensure!(
            weight.dtype == WeightDtype::BF16
                && weight.shape == [4096, 4096]
                && !weight.ptr.is_null()
                && weight.ptr.0.is_multiple_of(16),
            "invalid attention output weight"
        );
        self.stage = QkvStage::Empty;
        let attended = attention.forward(self.qkv, stream)?;
        ops::dense_gemm_bf16_pipelined(
            self.gpu,
            self.gemm,
            attended,
            &DenseWeight { weight: weight.ptr },
            self.modulated,
            self.rows,
            4096,
            4096,
            stream,
        )?;
        self.stage = QkvStage::Attended;
        Ok(self.modulated)
    }
}
impl Drop for DiagnosticImagePrelude<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}
