// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Shared fixture of the W8A8 GEMM examples (`w8a8_gemm_bitgate`,
//! `w8a8_shared_quant_gate`): deterministic BF16 data, host<->device copies, an FP8 per-row weight
//! segment quantized on the device, and a median per-launch timer.
//!
//! Owner: model-arch examples.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, WeightQuantFormat, quantize_to_fp8,
};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn bf16_bits(&mut self) -> u16 {
        let u = self.next();
        (((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f)) as u16
    }
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn download(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub fn seg(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    quant: KernelHandle,
    n: usize,
    k: usize,
) -> Result<Fp8Weight> {
    let bf: Vec<u8> = (0..n * k)
        .flat_map(|_| rng.bf16_bits().to_le_bytes())
        .collect();
    let dev = upload(gpu, &bf)?;
    let q = quantize_to_fp8(&DenseWeight { weight: dev }, n, k, gpu, quant, 0)?;
    gpu.free(dev)?;
    Ok(Fp8Weight {
        weight: q.weight,
        row_scale: q.row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

pub fn median_us(
    gpu: &dyn GpuBackend,
    launches: usize,
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    let mut t = Vec::new();
    for rep in 0..5 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for i in 0..launches {
            f(rep * launches + i)?;
        }
        gpu.synchronize(0)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / launches as f64);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[2])
}
