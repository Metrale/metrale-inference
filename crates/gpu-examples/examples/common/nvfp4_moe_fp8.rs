// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The FP8 expert matrices of `nvfp4_moe_grouped_microtest` (moved out of
//! nvfp4_moe_fixture.rs unchanged, to keep that file under 500 lines).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

use super::fixture::{HostDot, Rng, e4m3, upload};

/// 2026-09-27: One FP8 E4M3 matrix `[n, k]` with f32 block scales `[ceil(n/128), ceil(k/128)]`.
pub(crate) struct Fp8Mat {
    bytes: Vec<u8>,
    scales: Vec<f32>,
    pub(crate) w: Fp8Weight,
    k: usize,
}

impl Fp8Mat {
    pub(crate) fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Self> {
        let bytes: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 120) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let scales: Vec<f32> = (0..n.div_ceil(128) * k.div_ceil(128))
            .map(|_| ((rng.next() % 16 + 1) as f32) / 4096.0)
            .collect();
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let w = Fp8Weight {
            weight: upload(g, &bytes)?,
            row_scale: upload(g, &scale_bytes)?,
            n: n as u32,
            k: k as u32,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };
        Ok(Self {
            bytes,
            scales,
            w,
            k,
        })
    }
}

impl HostDot for Fp8Mat {
    fn dot(&self, row: usize, x: &[f64]) -> f64 {
        let kb = self.k.div_ceil(128);
        (0..self.k)
            .map(|c| {
                e4m3(self.bytes[row * self.k + c])
                    * self.scales[(row / 128) * kb + c / 128] as f64
                    * x[c]
            })
            .sum()
    }
}

/// 2026-09-27: Device pointer tables (weights, scales) of FP8 projections, one entry per
/// expert.
pub(crate) fn fp8_table(g: &dyn GpuBackend, mats: &[&Fp8Mat]) -> Result<(DevicePtr, DevicePtr)> {
    let ptrs = |f: &dyn Fn(&Fp8Mat) -> u64| -> Vec<u8> {
        mats.iter().flat_map(|m| f(m).to_le_bytes()).collect()
    };
    Ok((
        upload(g, &ptrs(&|m| m.w.weight.0))?,
        upload(g, &ptrs(&|m| m.w.row_scale.0))?,
    ))
}
