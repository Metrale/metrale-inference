// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Inputs, device buffers and the byte comparison shared by the norm-fusion
//! microtests (`residual_add_rms_norm_exact_microtest`, `rms_norm_act_quant_microtest`).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Row r of every input uses pattern r % PATTERNS, so each row count from 1 to 128 covers
//!   every pattern once it passes PATTERNS rows, and row 0 is always the plain random row.
//! - Buffers are filled with SENTINEL before a leg runs; `compare` also checks that the rows
//!   past the launched ones still hold it.

#![allow(dead_code)]

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-28: The hidden sizes of the two circuit models: dense Qwen3.8-27B and the
/// Qwen3.6-35B-A3B MoE.
pub const HIDDENS: [usize; 2] = [5120, 2048];
/// 2026-09-28: Rows 1..=MAX_ROWS are launched.
pub const MAX_ROWS: usize = 128;
pub const SENTINEL: u8 = 0xA5;
pub const EPS: f32 = 1e-6;

/// 2026-09-28: The kernel targets the circuits run on; each compiled one is tested.
pub const TARGETS: [&str; 2] = ["qwen3.8-27b", "qwen3.6-35b-a3b"];

pub struct Rng(pub u64);
impl Rng {
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// 2026-09-28: Uniform on (-1, 1).
    pub fn sym(&mut self) -> f32 {
        2.0 * self.unit() - 1.0
    }
}

pub const PATTERNS: usize = 8;

/// 2026-09-28: One row of pattern `p`, `h` values: random, denormal, huge (the sum of squares
/// overflows to infinity), all equal, one-hot, mixed magnitudes, all zero, and exact
/// round-to-even ties for the residual add.
pub fn row(rng: &mut Rng, p: usize, h: usize, salt: usize) -> Vec<bf16> {
    (0..h)
        .map(|k| {
            let x = match p {
                0 => rng.sym() * 4.0,
                1 => rng.sym() * 1e-39,
                2 => rng.sym() * 3.0e38,
                3 => 0.75,
                4 => {
                    if k == (salt * 37) % h {
                        -9.5
                    } else {
                        0.0
                    }
                }
                5 => rng.sym() * 10f32.powi((k % 11) as i32 - 5),
                6 => 0.0,
                _ => 1.0 + (k % 4) as f32 * 2f32.powi(-8),
            };
            bf16::from_f32(x)
        })
        .collect()
}

/// 2026-09-28: `rows` rows of `h` values, row r of pattern r % PATTERNS.
pub fn matrix(rng: &mut Rng, rows: usize, h: usize, salt: usize) -> Vec<bf16> {
    (0..rows)
        .flat_map(|r| row(rng, r % PATTERNS, h, r + salt))
        .collect()
}

pub fn bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

pub fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

/// 2026-09-28: A device buffer of `n` bytes filled with SENTINEL.
pub fn sentinel(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    upload(g, &vec![SENTINEL; n])
}

pub fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-09-28: Mismatching bytes between two legs' buffers of `total` bytes of which `live`
/// were written; bytes past `live` must still hold SENTINEL in both.
pub fn compare(
    g: &dyn GpuBackend,
    a: DevicePtr,
    b: DevicePtr,
    live: usize,
    total: usize,
) -> Result<usize> {
    let (x, y) = (download(g, a, total)?, download(g, b, total)?);
    let mut bad = x[..live]
        .iter()
        .zip(&y[..live])
        .filter(|(p, q)| p != q)
        .count();
    bad += x[live..]
        .iter()
        .chain(&y[live..])
        .filter(|&&v| v != SENTINEL)
        .count();
    Ok(bad)
}

/// 2026-09-28: A backend on each compiled target of TARGETS; errors when none is compiled.
pub fn backends() -> Result<Vec<(&'static str, MetraleCudaBackend)>> {
    let mut out = Vec::new();
    for t in TARGETS {
        match metrale_kernels::ptx_for_exact_target(t, "nvfp4") {
            Some(set) => out.push((t, MetraleCudaBackend::new(0, &set.modules)?)),
            None => println!("target {t}/nvfp4 is not compiled in; skipped"),
        }
    }
    anyhow::ensure!(!out.is_empty(), "no circuit target is compiled in");
    Ok(out)
}

pub fn weight(g: &dyn GpuBackend, rng: &mut Rng, h: usize) -> Result<DevicePtr> {
    let w: Vec<bf16> = (0..h).map(|_| bf16::from_f32(rng.sym() * 0.5)).collect();
    upload(g, &bytes(&w)).context("weight")
}
