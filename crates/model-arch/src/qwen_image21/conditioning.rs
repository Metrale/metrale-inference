// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native time MLP, shared block modulation and final adaptive scale.
//! This consumes scheduler timesteps; it does not substitute for the text encoder.
use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};
use metrale_model_layers::{layers::ops, weight_map::DenseWeight};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};

pub struct ConditioningWeights<'a> {
    pub time_in: &'a WeightTensor,
    pub time_out: &'a WeightTensor,
    pub modulation: &'a WeightTensor,
    pub final_scale: &'a WeightTensor,
}
/// Device outputs stay owned by the conditioner and are overwritten per step.
#[derive(Clone, Copy)]
pub struct ConditioningBuffers {
    pub time_embedding: DevicePtr,
    pub modulation: DevicePtr,
    pub final_scale: DevicePtr,
}
pub struct DiagnosticConditioning<'a> {
    gpu: &'a dyn GpuBackend,
    weights: ConditioningWeights<'a>,
    batch: u32,
    rows: u32,
    allocation: DevicePtr,
    timestep: DevicePtr,
    frequency: DevicePtr,
    projected: DevicePtr,
    hidden: DevicePtr,
    activated: DevicePtr,
    output: ConditioningBuffers,
    timestep_kernel: KernelHandle,
    activation_kernel: KernelHandle,
    gemm: KernelHandle,
}
/// Native FP32 constructor policy; compare its bytes independently to the pinned
/// CPU reference table instead of assuming libm and vectorized exp are identical.
pub fn timestep_frequencies() -> Vec<f32> {
    (0..128)
        .map(|i| (-(10000.0f64.ln() as f32) * i as f32 / 128.0).exp())
        .collect()
}
impl<'a> DiagnosticConditioning<'a> {
    pub fn new(
        gpu: &'a dyn GpuBackend,
        batch: u32,
        weights: ConditioningWeights<'a>,
    ) -> Result<Self> {
        ensure!(batch > 0, "empty timestep batch");
        let rows = batch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("timestep row overflow"))?;
        ensure!(
            rows.checked_mul(16384).is_some(),
            "conditioning launch overflow"
        );
        for (weight, shape) in [
            (weights.time_in, [4096, 256]),
            (weights.time_out, [4096, 4096]),
            (weights.modulation, [16384, 4096]),
            (weights.final_scale, [4096, 4096]),
        ] {
            ensure!(
                weight.dtype == WeightDtype::BF16
                    && weight.shape == shape
                    && !weight.ptr.is_null()
                    && weight.ptr.0.is_multiple_of(16),
                "invalid conditioning weight"
            );
        }
        let timestep_kernel = gpu.kernel("image_modulation", "image_timestep_bf16")?;
        let activation_kernel = gpu.kernel("image_modulation", "image_silu_staged_mul_bf16")?;
        let gemm = gpu.kernel("dense_gemm_bf16", "dense_gemm_bf16_pipelined")?;
        let row = rows as usize * 4096 * 2;
        let projection = rows as usize * 256 * 2;
        let bytes = row
            .checked_mul(8)
            .and_then(|n| n.checked_add(projection + 512 + rows as usize * 2))
            .ok_or_else(|| anyhow::anyhow!("conditioning allocation overflow"))?;
        let allocation = gpu.alloc(bytes)?;
        let value = Self {
            gpu,
            weights,
            batch,
            rows,
            allocation,
            projected: allocation,
            hidden: allocation.offset(projection),
            activated: allocation.offset(projection + row),
            output: ConditioningBuffers {
                time_embedding: allocation.offset(projection + 2 * row),
                modulation: allocation.offset(projection + 3 * row),
                final_scale: allocation.offset(projection + 7 * row),
            },
            frequency: allocation.offset(projection + 8 * row),
            timestep: allocation.offset(projection + 8 * row + 512),
            timestep_kernel,
            activation_kernel,
            gemm,
        };
        let frequencies: Vec<u8> = timestep_frequencies()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        gpu.copy_h2d(&frequencies, value.frequency)?;
        Ok(value)
    }
    /// Timesteps are finite in [0,1], cast to BF16 before time_factor1000.
    /// The appended t=0 row conditions all non-target tokens in every sample.
    pub fn forward(&mut self, timesteps: &[f32], stream: u64) -> Result<ConditioningBuffers> {
        ensure!(
            timesteps.len() == self.batch as usize
                && timesteps
                    .iter()
                    .all(|t| t.is_finite() && (0.0..=1.0).contains(t)),
            "invalid scheduler timesteps"
        );
        let bytes: Vec<u8> = timesteps
            .iter()
            .copied()
            .chain([0.0])
            .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        self.gpu.copy_h2d_async(&bytes, self.timestep, stream)?;
        KernelLaunch::new(self.gpu, self.timestep_kernel)
            .grid([self.rows, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.timestep)
            .arg_ptr(self.frequency)
            .arg_ptr(self.projected)
            .arg_u32(self.rows)
            .launch(stream)?;
        self.linear(
            self.projected,
            self.weights.time_in,
            self.hidden,
            4096,
            256,
            stream,
        )?;
        self.activate(self.hidden, self.activated, stream)?;
        self.linear(
            self.activated,
            self.weights.time_out,
            self.output.time_embedding,
            4096,
            4096,
            stream,
        )?;
        self.activate(self.output.time_embedding, self.activated, stream)?;
        self.linear(
            self.activated,
            self.weights.modulation,
            self.output.modulation,
            16384,
            4096,
            stream,
        )?;
        self.linear(
            self.activated,
            self.weights.final_scale,
            self.output.final_scale,
            4096,
            4096,
            stream,
        )?;
        Ok(self.output)
    }
    fn activate(&self, input: DevicePtr, output: DevicePtr, stream: u64) -> Result<()> {
        ops::silu_mul(
            self.gpu,
            self.activation_kernel,
            input,
            DevicePtr::NULL,
            output,
            self.rows * 4096,
            stream,
        )
    }
    fn linear(
        &self,
        input: DevicePtr,
        weight: &WeightTensor,
        output: DevicePtr,
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
            self.rows,
            n,
            k,
            stream,
        )
    }
}
impl Drop for DiagnosticConditioning<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}
