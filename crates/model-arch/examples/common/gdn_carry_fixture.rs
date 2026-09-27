// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Inputs, uploads, comparisons and the parent-kernel launch shared by
//! `gdn_carry_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;

pub(crate) const KD: usize = 128;
pub(crate) const VD: usize = 128;
/// 2026-09-26: Pointer entries per table, equal to
/// `metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS`.
pub(crate) const SLAB_ENTRIES: usize = 32;

pub(crate) fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// 2026-09-26: Deterministic LCG in [-1, 1).
pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

pub(crate) fn upload_bytes(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub(crate) fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub(crate) fn read_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn ptr_table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let mut bytes = vec![0u8; SLAB_ENTRIES * 8];
    for (i, p) in ptrs.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&p.0.to_le_bytes());
    }
    upload_bytes(g, &bytes)
}

/// 2026-09-26: (mismatching elements, max ULP distance) between two f32 buffers.
pub(crate) fn diff_f32(a: &[f32], b: &[f32]) -> (usize, u32) {
    let mut n = 0usize;
    let mut ulp = 0u32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            let d = (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs();
            ulp = ulp.max(d.min(u32::MAX as u64) as u32);
        }
    }
    (n, ulp)
}

pub(crate) struct Dims {
    pub(crate) seqs: usize,
    pub(crate) nk: usize,
    pub(crate) nv: usize,
    pub(crate) conv_dim: usize,
    pub(crate) value_dim: usize,
    pub(crate) gb_stride: usize,
    pub(crate) h_numel: usize,
}

/// 2026-09-26: One round of verify inputs: bf16 q|k|v rows and f32 gate|beta rows.
pub(crate) struct Round {
    pub(crate) q: DevicePtr,
    pub(crate) k: DevicePtr,
    pub(crate) v: DevicePtr,
    pub(crate) gate: DevicePtr,
    pub(crate) beta: DevicePtr,
}

/// 2026-09-26: Host inputs of `rows` verify rows: BF16 q|k|v bytes and f32 gate|beta.
pub(crate) fn round_data(d: &Dims, rows: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f32>) {
    let qkv: Vec<u8> = (0..rows * d.conv_dim)
        .flat_map(|_| bf16::from_f32(rng.next_f32() * 0.5).to_bits().to_le_bytes())
        .collect();
    let mut gb = vec![0f32; rows * d.gb_stride];
    for r in 0..rows {
        for h in 0..d.nv {
            gb[r * d.gb_stride + h] = 0.5 + 0.49 * rng.next_f32();
            gb[r * d.gb_stride + d.nv + h] = 0.5 + 0.5 * rng.next_f32().abs();
        }
    }
    (qkv, gb)
}

pub(crate) fn make_round(g: &dyn GpuBackend, d: &Dims, kk: usize, rng: &mut Rng) -> Result<Round> {
    let (qkv, gb) = round_data(d, d.seqs * kk, rng);
    upload_round(g, d, &qkv, &gb)
}

pub(crate) fn upload_round(g: &dyn GpuBackend, d: &Dims, qkv: &[u8], gb: &[f32]) -> Result<Round> {
    let qkv_dev = upload_bytes(g, qkv)?;
    let gb_dev = upload_bytes(g, &f32_bytes(gb))?;
    let key_dim = d.nk * KD;
    Ok(Round {
        q: qkv_dev,
        k: qkv_dev.offset(key_dim * 2),
        v: qkv_dev.offset(key_dim * 2 * 2),
        gate: gb_dev,
        beta: gb_dev.offset(d.nv * 4),
    })
}

/// 2026-09-26: Parent verify of width `kk` in table form: `h` states, `hi[t]` intermediates.
pub(crate) fn run_parent(
    g: &dyn GpuBackend,
    k: KernelHandle,
    kk: usize,
    d: &Dims,
    r: &Round,
    h: DevicePtr,
    hi: &[DevicePtr],
    out: DevicePtr,
) -> Result<()> {
    let s = g.default_stream();
    let (n, nk, nv) = (d.seqs as u32, d.nk as u32, d.nv as u32);
    let (cd, gbs) = (d.conv_dim as u32, d.gb_stride as u32);
    match kk {
        2 => ops::gdn_decode_wy2(
            g, k, h, r.q, r.k, r.v, r.gate, r.beta, out, hi[0], n, nk, nv, KD as u32, VD as u32,
            cd, cd, gbs, true, s,
        )?,
        3 => ops::gdn_decode_wy3(
            g, k, h, r.q, r.k, r.v, r.gate, r.beta, out, hi[0], hi[1], n, nk, nv, KD as u32,
            VD as u32, cd, cd, gbs, true, s,
        )?,
        _ => ops::gdn_decode_wy4(
            g, k, h, r.q, r.k, r.v, r.gate, r.beta, out, hi[0], hi[1], hi[2], n, nk, nv, KD as u32,
            VD as u32, cd, cd, gbs, true, s,
        )?,
    }
    g.synchronize(s)
}
