// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Inputs and the f64 reference for the head-batched DSA decode bench
//! (`glm5next_dsa_hb_bench`): an FP8 latent pool past L2, and per shape [`VARIANTS`] block
//! tables and selections laid out as `dsa_expand_selection` lays them.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub const KVL: usize = 512;
/// 2026-10-09: `out_width` for index_topk 2048, kpool 4, tail on.
pub const WIDTH: usize = 2051;
pub const BLOCK: usize = 64;
pub const SEQ: usize = 8192;
pub const BLOCKS_PER_SEQ: usize = SEQ / BLOCK;
pub const POOL_BLOCKS: usize = 4096;
pub const VARIANTS: usize = 64;
pub struct Lcg(pub u64);
impl Lcg {
    pub fn u(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    pub fn f(&mut self) -> f32 {
        (self.u() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

pub fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    s * if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

pub fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
pub fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
pub fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|x| bf16::from_le_bytes([x[0], x[1]]).to_f32())
        .collect()
}

/// 2026-10-09: The FP8 latent pool every block table points into: finite E4M3 values of
/// magnitude <= 4 (exponent field <= 8).
pub struct Pool {
    pub host: Vec<u8>,
    pub dev: DevicePtr,
}

pub fn make_pool(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Pool> {
    let host: Vec<u8> = (0..POOL_BLOCKS * BLOCK * KVL)
        .map(|_| {
            let r = rng.u();
            ((r & 0x80) as u8) | ((((r >> 8) % 9) as u8) << 3) | ((r >> 16) & 7) as u8
        })
        .collect();
    Ok(Pool {
        dev: up(g, &host)?,
        host,
    })
}

/// 2026-10-09: One shape: Q, seq_lens, and [`VARIANTS`] block tables and selections.
pub struct Case {
    pub heads: usize,
    pub q: Vec<f32>,
    pub q_d: DevicePtr,
    pub sl: Vec<i32>,
    pub sl_d: DevicePtr,
    pub bt: Vec<Vec<i32>>,
    pub bt_d: Vec<DevicePtr>,
    pub sel: Vec<Vec<i32>>,
    pub sel_d: Vec<DevicePtr>,
}

/// 2026-10-09: Device pointers of one launch: `rows` rows from row `r0` of one variant.
#[derive(Clone, Copy)]
pub struct View {
    pub q: DevicePtr,
    pub bt: DevicePtr,
    pub sel: DevicePtr,
    pub sl: DevicePtr,
    pub rows: usize,
    pub heads: usize,
}

impl Case {
    pub fn view(&self, v: usize, r0: usize, rows: usize) -> View {
        View {
            q: self.q_d.offset(r0 * self.heads * KVL * 2),
            bt: self.bt_d[v].offset(r0 * BLOCKS_PER_SEQ * 4),
            sel: self.sel_d[v].offset(r0 * WIDTH * 4),
            sl: self.sl_d.offset(r0 * 4),
            rows,
            heads: self.heads,
        }
    }
}

/// 2026-10-09: As `dsa_expand_selection` lays a row out: `valid` slots of pools of 4
/// consecutive tokens (every 9th pool a -1 hole), the 3-token tail, then -1. Q is scaled so
/// scores spread over several units.
pub fn make_case(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    heads: usize,
    rows: usize,
    valid: usize,
) -> Result<Case> {
    let q: Vec<f32> = (0..rows * heads * KVL)
        .map(|_| bf16::from_f32(rng.f() * 4.0).to_f32())
        .collect();
    let sl: Vec<i32> = (0..rows).map(|r| (SEQ - 7 * r) as i32).collect();
    let (mut bt, mut bt_d, mut sel, mut sel_d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..VARIANTS {
        let t: Vec<i32> = (0..rows * BLOCKS_PER_SEQ)
            .map(|_| (rng.u() as usize % POOL_BLOCKS) as i32)
            .collect();
        let mut s = vec![-1i32; rows * WIDTH];
        for r in 0..rows {
            let pools = (valid.min(WIDTH - 3) / 4).min(sl[r] as usize / 4);
            for p in 0..pools {
                let start = 4 * (rng.u() as usize % (sl[r] as usize / 4));
                for k in 0..4 {
                    s[r * WIDTH + 4 * p + k] = if p % 9 == 8 { -1 } else { (start + k) as i32 };
                }
            }
            for k in 0..3 {
                s[r * WIDTH + 4 * pools + k] = sl[r] - 3 + k as i32;
            }
        }
        bt_d.push(up_i32(g, &t)?);
        bt.push(t);
        sel_d.push(up_i32(g, &s)?);
        sel.push(s);
    }
    let qb: Vec<u8> = q
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    Ok(Case {
        heads,
        q_d: up(g, &qb)?,
        q,
        sl_d: up_i32(g, &sl)?,
        sl,
        bt,
        bt_d,
        sel,
        sel_d,
    })
}

/// 2026-10-09: f64 reference for (row, head) of variant 0: softmax over the valid entries.
pub fn reference(c: &Case, pool: &Pool, row: usize, h: usize) -> Vec<f64> {
    let tok = |t: usize| {
        let b = c.bt[0][row * BLOCKS_PER_SEQ + t / BLOCK] as usize;
        &pool.host[(b * BLOCK + t % BLOCK) * KVL..][..KVL]
    };
    let q = &c.q[(row * c.heads + h) * KVL..][..KVL];
    let mut sc = Vec::new();
    for j in 0..WIDTH {
        let t = c.sel[0][row * WIDTH + j];
        if t >= 0 && t < c.sl[row] {
            let d: f64 = tok(t as usize)
                .iter()
                .zip(q)
                .map(|(k, q)| *q as f64 * e4m3(*k) as f64)
                .sum();
            sc.push((t as usize, d / (KVL as f64).sqrt()));
        }
    }
    let mut o = vec![0f64; KVL];
    if sc.is_empty() {
        return o;
    }
    let m = sc.iter().fold(f64::MIN, |a, s| a.max(s.1));
    let mut l = 0f64;
    for (t, s) in &sc {
        let p = (s - m).exp();
        l += p;
        for (i, b) in tok(*t).iter().enumerate() {
            o[i] += p * e4m3(*b) as f64;
        }
    }
    o.iter_mut().for_each(|x| *x /= l);
    o
}
