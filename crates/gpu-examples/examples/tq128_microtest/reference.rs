// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The CPU side of `tq128_microtest`: the paged layout, the input rows, the
//! reference encode and dequant (`metrale-kv-tq128`), f64 reference attention, and the error
//! measure the decode gate uses.
//!
//! Owner: kv-tq128 examples.
//! Invariants:
//! - Caches and dequant outputs not written for a slot keep `SENTINEL`, as the GPU legs do.
//! - Row t of a token-major buffer starts at `t * Geometry::width()`; head h at `h * head_dim`.

use anyhow::Result;
use half::bf16;
use metrale_kv_tq128::{GROUP, KvSide, Tq128Codec, dequant_group, dequant_head_bf16, encode_head};

use super::fixture::SENTINEL;

pub const BLOCK_SIZE: usize = 16;
pub const KV_HEADS: usize = 2;
pub const Q_HEADS: usize = 4;
pub const SEQ_LENS: [usize; 3] = [1, 23, 61];
/// 2026-10-18: Physical blocks in the order the sequences take them; block 4 stays unwritten.
pub const PHYS: [i32; 7] = [5, 2, 7, 0, 3, 6, 1];
pub const NUM_BLOCKS: usize = 8;
pub const MAX_BLOCKS_PER_SEQ: usize = 4;
/// 2026-10-18: Decode tolerance: max |gpu - ref| over an output row <= TOL * max |ref|. Rounding
/// the output to bf16 alone costs up to 2^-8 (3.9e-3) of an element; the rest is slack for the
/// fp32 softmax. The tightest negative control (turbo3_tcq's 1.02 V factor dropped) misses by
/// about 2%, so it stays four times above this.
pub const TOL: f64 = 5e-3;

pub struct Rng(pub u64);
impl Rng {
    pub fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

pub fn bf16_at(b: &[u8], i: usize) -> f64 {
    f64::from(bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32())
}

/// 2026-10-18: Tokens of all sequences in order, then one with slot -1.
pub fn slots() -> Vec<i64> {
    let mut out = Vec::new();
    let mut next_block = 0;
    for &len in &SEQ_LENS {
        let blocks = &PHYS[next_block..next_block + len.div_ceil(BLOCK_SIZE)];
        for pos in 0..len {
            out.push(
                i64::from(blocks[pos / BLOCK_SIZE]) * BLOCK_SIZE as i64 + (pos % BLOCK_SIZE) as i64,
            );
        }
        next_block += blocks.len();
    }
    out.push(-1);
    out
}

pub fn rows(rng: &mut Rng, tokens: usize, width: usize) -> Vec<bf16> {
    let mut v: Vec<f64> = (0..tokens * width).map(|_| rng.normal()).collect();
    let row = |t: usize| t * width..(t + 1) * width;
    v[row(3)].fill(0.0);
    v[row(5)].fill(1e-20);
    v[7 * width + 17] *= 100.0;
    v[row(9)].iter_mut().for_each(|x| *x *= 1000.0);
    v[row(11)].fill(0.75);
    v.into_iter().map(bf16::from_f64).collect()
}

pub struct Geometry {
    pub codec: Tq128Codec,
    pub head_dim: usize,
}
impl Geometry {
    pub fn head_bytes(&self) -> usize {
        self.head_dim / GROUP * self.codec.block_bytes()
    }
    pub fn block_stride(&self) -> usize {
        BLOCK_SIZE * KV_HEADS * self.head_bytes()
    }
    pub fn offset(&self, slot: i64, head: usize) -> Result<usize> {
        let s = usize::try_from(slot)?;
        Ok(s / BLOCK_SIZE * self.block_stride()
            + (s % BLOCK_SIZE * KV_HEADS + head) * self.head_bytes())
    }
    pub fn width(&self) -> usize {
        KV_HEADS * self.head_dim
    }
}

pub fn cpu_encode(geo: &Geometry, side: KvSide, src: &[bf16], slots: &[i64]) -> Result<Vec<u8>> {
    let mut cache = vec![SENTINEL; NUM_BLOCKS * geo.block_stride()];
    for (t, &slot) in slots.iter().enumerate().filter(|(_, s)| **s >= 0) {
        for h in 0..KV_HEADS {
            let base = t * geo.width() + h * geo.head_dim;
            let row: Vec<f32> = src[base..base + geo.head_dim]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            let off = geo.offset(slot, h)?;
            encode_head(
                geo.codec,
                side,
                &row,
                &mut cache[off..off + geo.head_bytes()],
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }
    Ok(cache)
}

pub fn cpu_dequant(geo: &Geometry, side: KvSide, cache: &[u8], slots: &[i64]) -> Result<Vec<u8>> {
    let mut out = vec![SENTINEL; slots.len() * geo.width() * 2];
    for (r, &slot) in slots.iter().enumerate().filter(|(_, s)| **s >= 0) {
        for h in 0..KV_HEADS {
            let off = geo.offset(slot, h)?;
            let mut row = vec![0u16; geo.head_dim];
            dequant_head_bf16(
                geo.codec,
                side,
                &cache[off..off + geo.head_bytes()],
                &mut row,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let base = (r * geo.width() + h * geo.head_dim) * 2;
            for (i, v) in row.iter().enumerate() {
                out[base + 2 * i..base + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// 2026-10-18: The reference dequant before bf16 rounding, which is what the decode kernel
/// attends over: it never rounds K or V to bf16, only its output row.
pub fn cpu_dequant_f32(
    geo: &Geometry,
    side: KvSide,
    cache: &[u8],
    slots: &[i64],
) -> Result<Vec<f64>> {
    let bb = geo.codec.block_bytes();
    let mut out = vec![0f64; slots.len() * geo.width()];
    for (r, &slot) in slots.iter().enumerate().filter(|(_, s)| **s >= 0) {
        for h in 0..KV_HEADS {
            for g in 0..geo.head_dim / GROUP {
                let off = geo.offset(slot, h)? + g * bb;
                let y = dequant_group(geo.codec, side, &cache[off..off + bb])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let base = r * geo.width() + h * geo.head_dim + g * GROUP;
                out[base..base + GROUP]
                    .iter_mut()
                    .zip(y)
                    .for_each(|(o, v)| *o = f64::from(v));
            }
        }
    }
    Ok(out)
}

pub fn block_tables() -> Vec<i32> {
    let mut t = vec![0i32; SEQ_LENS.len() * MAX_BLOCKS_PER_SEQ];
    let mut next = 0;
    for (s, &len) in SEQ_LENS.iter().enumerate() {
        for b in 0..len.div_ceil(BLOCK_SIZE) {
            t[s * MAX_BLOCKS_PER_SEQ + b] = PHYS[next];
            next += 1;
        }
    }
    t
}

/// 2026-10-18: f64 attention over dequantized rows `kd` / `vd` (one value per element, laid out
/// as `cpu_dequant` lays out its bytes); `kv_of` maps a query head to its KV head and `v_mul`
/// scales every V value.
pub fn reference(
    geo: &Geometry,
    q: &[bf16],
    kd: &[f64],
    vd: &[f64],
    kv_of: fn(usize) -> usize,
    v_mul: f64,
) -> Vec<f64> {
    let d = geo.head_dim;
    let mut out = vec![0f64; SEQ_LENS.len() * Q_HEADS * d];
    let mut first = 0;
    for (s, &len) in SEQ_LENS.iter().enumerate() {
        for h in 0..Q_HEADS {
            let kvh = kv_of(h);
            let qr: Vec<f64> = (0..d)
                .map(|i| f64::from(q[(s * Q_HEADS + h) * d + i].to_f32()))
                .collect();
            let at = |t: usize, i: usize| (first + t) * geo.width() + kvh * d + i;
            let scores: Vec<f64> = (0..len)
                .map(|t| (0..d).map(|i| qr[i] * kd[at(t, i)]).sum::<f64>() / (d as f64).sqrt())
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for i in 0..d {
                let acc: f64 = (0..len).map(|t| w[t] * vd[at(t, i)]).sum();
                out[(s * Q_HEADS + h) * d + i] = acc / z * v_mul;
            }
        }
        first += len;
    }
    out
}

/// 2026-10-18: The worst row's max |gpu - ref| / max |ref|.
pub fn worst_row_error(geo: &Geometry, gpu: &[u8], r: &[f64]) -> f64 {
    let d = geo.head_dim;
    (0..SEQ_LENS.len() * Q_HEADS)
        .map(|row| {
            let scale = (0..d)
                .map(|i| r[row * d + i].abs())
                .fold(0.0, f64::max)
                .max(1e-30);
            (0..d)
                .map(|i| (bf16_at(gpu, row * d + i) - r[row * d + i]).abs())
                .fold(0.0, f64::max)
                / scale
        })
        .fold(0.0, f64::max)
}

pub fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}
