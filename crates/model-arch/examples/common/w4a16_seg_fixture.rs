// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The shared fixture of the `w4a16_gemv_seg` benches: GLM-5.3's rank-0 W4A16
//! projection groups, a 128 MB pool of random NVFP4 weights rotated so each launch streams from
//! DRAM, and the host helpers.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
#![allow(dead_code)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::W4a16Seg;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-10: `(name, [n of each member], k, launches per decode step)`: the rank-0 W4A16
/// projections (34 KDA layers, 42 shared experts, 11 DSA layers under METRALE_GLM_DSA_W4A16=1).
pub const GROUPS: [(&str, &[u32], u32, u32); 13] = [
    (
        "kda q|k|v|f_a|b|g_a",
        &[2816, 2816, 2816, 128, 22, 128],
        4096,
        34,
    ),
    ("kda q|k|v", &[2816, 2816, 2816], 4096, 0),
    ("kda f_a|b|g_a", &[128, 22, 128], 4096, 0),
    ("kda q", &[2816], 4096, 0),
    ("kda f_a", &[128], 4096, 0),
    ("kda b", &[22], 4096, 0),
    ("kda o", &[4096], 2816, 34),
    ("shared gate|up", &[768, 768], 4096, 42),
    ("shared gate", &[768], 4096, 0),
    ("shared down", &[4096], 768, 42),
    ("dsa q_a", &[1536], 4096, 11),
    ("dsa q_absorb", &[11264], 1536, 11),
    ("dsa o_absorb", &[4096], 11264, 11),
];

pub const POOL: usize = 128 << 20;
pub const STREAM_BYTES: usize = 96 << 20;
pub const ROUNDS: usize = 5;

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

pub fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

pub fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-10: Random packed codes and E4M3 scale bytes (normal range, both signs of every code).
pub struct Pool {
    packed: DevicePtr,
    scale: DevicePtr,
}

impl Pool {
    pub fn new(g: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        let packed: Vec<u8> = (0..POOL).map(|_| rng.next() as u8).collect();
        let scale: Vec<u8> = (0..POOL / 8)
            .map(|_| (0x08 + rng.next() % 0x70) as u8)
            .collect();
        Ok(Self {
            packed: upload(g, &packed)?,
            scale: upload(g, &scale)?,
        })
    }

    /// 2026-10-10: Weight `[n, k]` number `i` in the rotation: consecutive slices of the pool.
    pub fn weight(&self, i: usize, n: u32, k: u32) -> QuantizedWeight {
        let pb = ((n as usize * k as usize / 2).div_ceil(256)) * 256;
        let slots = POOL / pb;
        let slot = i % slots;
        QuantizedWeight {
            weight: self.packed.offset(slot * pb),
            weight_scale: self.scale.offset(slot * pb / 8),
            weight_scale_2: 0.0123,
            ..QuantizedWeight::null()
        }
    }
}

pub fn nvfp4_bytes(ns: &[u32], k: u32) -> f64 {
    ns.iter()
        .map(|&n| f64::from(n) * f64::from(k) * (0.5 + 1.0 / 16.0))
        .sum()
}

pub fn segs(ws: &[QuantizedWeight], outs: &[DevicePtr], ns: &[u32]) -> Vec<W4a16Seg> {
    ws.iter()
        .zip(outs)
        .zip(ns)
        .map(|((&weight, &output), &n)| W4a16Seg { weight, output, n })
        .collect()
}

/// 2026-10-11: `m` random BF16 rows of `k` (exponents 2^-12..2^4, both signs).
pub fn bf16_rows(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    (0..m * k)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect()
}

/// 2026-10-11: Median, min and max of a sample.
pub fn stat(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0], v[v.len() - 1])
}
