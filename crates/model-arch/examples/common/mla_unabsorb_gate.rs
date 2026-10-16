// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Shared by the two MLA-UNABSORB G2 gates (`glm5next_mla_head_bmm_gate`,
//! `glm5next_mla_unabsorb_layer_gate`): BF16 byte conversions, the f64 references of
//! `glm5next_dsa::unabsorb::host` run over row blocks on every core, the error statistics both
//! gates print, and the device memory in use.
//!
//! Owner: model-arch examples (GLM-5.3 DSA).
//! Invariants:
//! - The parallel references only split rows: each block is `host::head_bmm_ref` /
//!   `host::dense_ref` on its rows, so they equal one call over all rows exactly.
#![allow(dead_code)]

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::unabsorb::host::{dense_ref, head_bmm_ref};

pub struct Lcg(pub u64);
impl Lcg {
    /// 2026-10-15: Uniform in [-1, 1).
    pub fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// 2026-10-15: `n` values `scale * U[-1, 1)`, rounded to BF16 (so the f32 holds the
    /// uploaded bits exactly).
    pub fn bf16_vals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| bf16::from_f32(self.next() * scale).to_f32())
            .collect()
    }
}

pub fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect()
}

pub fn from_bf16_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

pub fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(16))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

/// 2026-10-15: `n` bytes at `p` after the backend's stream drained.
pub fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn par_rows(m: usize, width: usize, f: impl Fn(usize, usize) -> Vec<f64> + Sync) -> Vec<f64> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let block = m.div_ceil(threads).max(1);
    let parts: Vec<Vec<f64>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..m)
            .step_by(block)
            .map(|r0| {
                let f = &f;
                s.spawn(move || f(r0, (r0 + block).min(m)))
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("ref thread"))
            .collect()
    });
    let out: Vec<f64> = parts.into_iter().flatten().collect();
    assert_eq!(out.len(), m * width);
    out
}

/// 2026-10-15: `host::head_bmm_ref` over row blocks in parallel.
pub fn head_bmm_par(a: &[f64], b: &[f64], m: usize, heads: usize, kk: usize, n: usize) -> Vec<f64> {
    let row = heads * kk;
    par_rows(m, heads * n, |r0, r1| {
        head_bmm_ref(&a[r0 * row..r1 * row], b, r1 - r0, heads, kk, n)
    })
}

/// 2026-10-15: `host::dense_ref` over row blocks in parallel.
pub fn dense_par(a: &[f64], w: &[f64], m: usize, n: usize, kk: usize) -> Vec<f64> {
    par_rows(m, n, |r0, r1| {
        dense_ref(&a[r0 * kk..r1 * kk], w, r1 - r0, n, kk)
    })
}

pub fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

/// 2026-10-15: One BF16 unit in the last place at `|x|` (normal range).
pub fn ulp_bf16(x: f64) -> f64 {
    x.abs().max(1e-30).log2().floor().exp2() / 128.0
}

/// 2026-10-15: An output against its f64 reference: the RMS of the error relative to the
/// reference's RMS, the worst absolute error, the worst error in BF16 ulps of the reference
/// element, and where the worst ulp error sits.
#[derive(Clone, Copy, Debug)]
pub struct ErrStats {
    pub rel_rms: f64,
    pub max_abs: f64,
    pub max_ulp: f64,
    pub worst_at: usize,
}

pub fn err_stats(out: &[f32], reference: &[f64]) -> ErrStats {
    assert_eq!(out.len(), reference.len());
    let (mut se, mut sr, mut max_abs, mut max_ulp, mut worst_at) = (0f64, 0f64, 0f64, 0f64, 0);
    for (i, (o, r)) in out.iter().zip(reference).enumerate() {
        let d = (*o as f64 - r).abs();
        se += d * d;
        sr += r * r;
        max_abs = max_abs.max(d);
        let u = d / ulp_bf16(*r);
        if u > max_ulp {
            max_ulp = u;
            worst_at = i;
        }
    }
    ErrStats {
        rel_rms: (se / sr.max(1e-300)).sqrt(),
        max_abs,
        max_ulp,
        worst_at,
    }
}

impl std::fmt::Display for ErrStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rel-RMS {:.3e} max-abs {:.3e} max-ulp {:.1} (elem {})",
            self.rel_rms, self.max_abs, self.max_ulp, self.worst_at
        )
    }
}

/// 2026-10-15: Device memory in use now, in MB, as the driver reports it.
pub fn used_mb(g: &dyn GpuBackend) -> Result<f64> {
    Ok((g.total_memory()? - g.device_free_memory()?) as f64 / 1e6)
}
