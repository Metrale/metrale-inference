// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fixture and byte oracle for `fp8_moe_grouped_decode_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

use super::{GUARD, H};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// 2026-09-26: Routed experts: the first argument, default 32 (rows share experts).
/// `256` is the model's count, where the distinct-expert bandwidth is representative.
pub fn experts() -> usize {
    std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(32)
}

/// 2026-09-26: The second argument, a Zipf exponent for the routing (default 0,
/// uniform): expert e is drawn with weight (e + 1)^-alpha. At 256 experts,
/// alpha 0.9 gives about the distinct-expert counts the 35B verify step
/// routes to (121 at 32 rows, 153 at 64, measured with
/// METRALE_DUMP_EXPERT_IDS=1 on the concurrency ladder).
pub fn zipf_alpha() -> f64 {
    std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.0)
}

/// 2026-09-26: `rows` rows of `top_k` distinct experts each, drawn from `e_count` experts
/// with weight (e + 1)^-alpha.
pub fn zipf_routing(
    rng: &mut Rng,
    e_count: usize,
    alpha: f64,
    rows: usize,
    top_k: usize,
) -> Vec<u32> {
    let weights: Vec<f64> = (0..e_count)
        .map(|e| ((e + 1) as f64).powf(-alpha))
        .collect();
    let total: f64 = weights.iter().sum();
    let cdf: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();
    let mut idx = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < top_k {
            let u = rng.next() as f64 / u32::MAX as f64;
            let e = cdf.partition_point(|&c| c < u).min(e_count - 1) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        idx.extend(row);
    }
    idx
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn fp8_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let x = rng.next();
            ((x % 120) as u8) | (((x >> 7) & 1) as u8 * 128)
        })
        .collect()
}

pub fn scale_bytes(rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    (0..n.div_ceil(128) * k.div_ceil(128))
        .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 512.0).to_le_bytes())
        .collect()
}

pub fn bf16_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0 * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

pub fn fp8w(weight: DevicePtr, row_scale: DevicePtr, n: usize, k: usize) -> Fp8Weight {
    Fp8Weight {
        weight,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// 2026-09-25: The oracle: the first `m` rows byte-equal between the two runs and
/// finite, every other byte still the sentinel on both.
pub fn check(observed: &[u8], baseline: &[u8], sentinel: &[u8], m: usize) -> Result<()> {
    ensure!(observed.len() == sentinel.len() && baseline.len() == sentinel.len());
    let live = GUARD..GUARD + m * H * 2;
    for i in 0..sentinel.len() {
        if !live.contains(&i) {
            ensure!(
                observed[i] == sentinel[i],
                "grouped wrote outside its rows at byte {i}"
            );
            ensure!(
                baseline[i] == sentinel[i],
                "loop wrote outside its rows at byte {i}"
            );
        }
    }
    let (a, b) = (&observed[live.clone()], &baseline[live]);
    for (i, (x, y)) in a.chunks_exact(2).zip(b.chunks_exact(2)).enumerate() {
        let (fx, fy) = (
            bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32(),
            bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32(),
        );
        ensure!(
            fx.is_finite() && fy.is_finite(),
            "nonfinite output at element {i}"
        );
        ensure!(
            x == y,
            "row {} col {}: grouped {fx} != loop {fy} (bf16 bits {x:?} vs {y:?})",
            i / H,
            i % H
        );
    }
    Ok(())
}
