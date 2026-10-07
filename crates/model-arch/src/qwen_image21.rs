// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Diagnostic native visual-block prelude; not a serving model.
//! LayerNorm uses the existing two-pass NLLB kernel, whose exact-reference gate
//! has three recorded near-zero BF16 mismatches. This path does not qualify it.
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
/// They are raw projections: no Q/K norm, RoPE, attention or KV reuse is implied.
#[derive(Clone, Copy)]
pub struct RawImageQkv {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
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
    qkv: RawImageQkv,
    ones: DevicePtr,
    zeros: DevicePtr,
    selection: DevicePtr,
    rows: u32,
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
            qkv: RawImageQkv {
                q: allocation.offset(2 * row_bytes),
                k: allocation.offset(3 * row_bytes),
                v: allocation.offset(4 * row_bytes),
            },
            ones: allocation.offset(5 * row_bytes),
            zeros: allocation.offset(5 * row_bytes + width * 2),
            selection: allocation.offset(5 * row_bytes + width * 4),
            rows,
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
    /// Caller provides BF16 hidden[rows,4096] and modulation[samples(+1),16384]
    /// with sizes from construction. The selection map is owned and cannot drift.
    /// Enqueues work on stream. Norm discrepancy remains explicit and unqualified.
    pub fn project(
        &mut self,
        hidden: DevicePtr,
        modulation: DevicePtr,
        stream: u64,
    ) -> Result<RawImageQkv> {
        ensure!(
            !hidden.is_null()
                && hidden.0.is_multiple_of(16)
                && !modulation.is_null()
                && modulation.0.is_multiple_of(2),
            "invalid Qwen prelude input pointers"
        );
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
        Ok(self.qkv)
    }
}
impl Drop for DiagnosticImagePrelude<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}
