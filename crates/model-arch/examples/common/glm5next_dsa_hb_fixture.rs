// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Inputs and the f64 reference for the head-batched DSA decode bench
//! (`glm5next_dsa_hb_bench`, `glm5next_dsa_prefill_pipe_gate`): an FP8 latent pool past L2, and per shape [`VARIANTS`] block
//! tables and selections laid out as `dsa_expand_selection` lays them.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

pub const KVL: usize = 512;
/// 2026-10-09: `out_width` for index_topk 2048, kpool 4, tail on.
pub const WIDTH: usize = 2051;
pub const BLOCK: usize = 64;
pub const SEQ: usize = 8192;
pub const BLOCKS_PER_SEQ: usize = SEQ / BLOCK;
pub const POOL_BLOCKS: usize = 4096;
pub const VARIANTS: usize = 64;
/// 2026-10-09: Heads per CTA of the head-batched entries.
pub const HB_HEADS: usize = 24;
/// 2026-10-09: Allowed distance from the f64 reference, in BF16 ulps (2^-8 relative) of the
/// head's largest |O|. The output itself rounds to half an ulp of each value; the rest is the
/// fp32 summation and the hi + lo P split. The per-head kernel measures the same (printed).
pub const TOL_ULP: f64 = 2.0;
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

/// 2026-10-09: The FP8 latent pool every block table points into: finite E4M3 values with an
/// exponent field <= `exp_max` (8: magnitude < 4; 15: the whole finite range, up to 448).
pub struct Pool {
    pub host: Vec<u8>,
    pub dev: DevicePtr,
}

pub fn make_pool(g: &dyn GpuBackend, rng: &mut Lcg, exp_max: u64) -> Result<Pool> {
    let host: Vec<u8> = (0..POOL_BLOCKS * BLOCK * KVL)
        .map(|_| {
            let r = rng.u();
            let b = ((r & 0x80) as u8)
                | ((((r >> 8) % (exp_max + 1)) as u8) << 3)
                | ((r >> 16) & 7) as u8;
            // 2026-10-09: 0x7f / 0xff are E4M3 NaN; use the largest finite code instead.
            if b & 0x7f == 0x7f { b - 1 } else { b }
        })
        .collect();
    Ok(Pool {
        dev: up(g, &host)?,
        host,
    })
}

/// 2026-10-09: One shape: Q, seq_lens, and [`VARIANTS`] block tables and selections.
/// 2026-10-14: `bps` is the block-table row stride (`max_blocks_per_seq`), [`BLOCKS_PER_SEQ`]
/// for [`make_case`].
pub struct Case {
    pub heads: usize,
    pub bps: usize,
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
    pub bps: usize,
}

impl Case {
    pub fn view(&self, v: usize, r0: usize, rows: usize) -> View {
        View {
            q: self.q_d.offset(r0 * self.heads * KVL * 2),
            bt: self.bt_d[v].offset(r0 * self.bps * 4),
            sel: self.sel_d[v].offset(r0 * WIDTH * 4),
            sl: self.sl_d.offset(r0 * 4),
            rows,
            heads: self.heads,
            bps: self.bps,
        }
    }
}

/// 2026-10-09: As `dsa_expand_selection` lays a row out: `valid` slots of pools of 4
/// consecutive tokens (every 9th pool a -1 hole), the 3-token tail, then -1. Q is uniform in
/// [-q_scale, q_scale].
pub fn make_case(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    heads: usize,
    rows: usize,
    valid: usize,
    q_scale: f32,
) -> Result<Case> {
    let q: Vec<f32> = (0..rows * heads * KVL)
        .map(|_| bf16::from_f32(rng.f() * q_scale).to_f32())
        .collect();
    let sl: Vec<i32> = (0..rows).map(|r| (SEQ - 7 * r) as i32).collect();
    let (mut bt, mut sel) = (Vec::new(), Vec::new());
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
        bt.push(t);
        sel.push(s);
    }
    upload_case(g, heads, BLOCKS_PER_SEQ, q, sl, bt, sel)
}

/// 2026-10-14: A [`Case`] from host arrays: `q` `[rows, heads, KVL]` (bf16-representable),
/// `sl` `[rows]`, per variant `bt` `[rows, bps]` and `sel` `[rows, WIDTH]`.
pub fn upload_case(
    g: &dyn GpuBackend,
    heads: usize,
    bps: usize,
    q: Vec<f32>,
    sl: Vec<i32>,
    bt: Vec<Vec<i32>>,
    sel: Vec<Vec<i32>>,
) -> Result<Case> {
    let qb: Vec<u8> = q
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    Ok(Case {
        heads,
        bps,
        q_d: up(g, &qb)?,
        q,
        sl_d: up_i32(g, &sl)?,
        sl,
        bt_d: bt.iter().map(|t| up_i32(g, t)).collect::<Result<_>>()?,
        bt,
        sel_d: sel.iter().map(|t| up_i32(g, t)).collect::<Result<_>>()?,
        sel,
    })
}

/// 2026-10-09: Row 0 all -1 (exact zeros), row 1 `seq_len` 0 (untouched), row 2 every entry
/// at or past `seq_len` (zeros), row 3 duplicates and out-of-range entries mixed in (vs the
/// reference, which attends a duplicate twice); 22 heads, variant 0 edited. 2026-10-14: moved
/// here from glm5next_dsa_hb_bench.rs, shared with glm5next_dsa_prefill_pipe_gate.rs.
pub fn make_edge_case(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Case> {
    let mut c = make_case(g, rng, 22, 4, 300, 4.0)?;
    let row = |r: usize| r * WIDTH..(r + 1) * WIDTH;
    c.sel[0][row(0)].fill(-1);
    c.sl[1] = 0;
    let sl2 = c.sl[2];
    c.sel[0][row(2)]
        .iter_mut()
        .filter(|x| **x >= 0)
        .for_each(|x| *x = sl2 + 3);
    for j in 0..200 {
        match j % 5 {
            0 => c.sel[0][3 * WIDTH + j] = 17,
            1 => c.sel[0][3 * WIDTH + j] = c.sl[3] + j as i32,
            _ => {}
        }
    }
    c.sel_d[0] = up_i32(g, &c.sel[0])?;
    c.sl_d = up_i32(g, &c.sl)?;
    Ok(c)
}

/// 2026-10-09: f64 reference for (row, head) of variant 0: softmax over the valid entries.
pub fn reference(c: &Case, pool: &Pool, row: usize, h: usize) -> Vec<f64> {
    let tok = |t: usize| {
        let b = c.bt[0][row * c.bps + t / BLOCK] as usize;
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

/// 2026-10-09: Worst distance from the reference over `rows` (variant 0), in BF16 ulps of each
/// head's largest |O|. 2026-10-14: moved here from glm5next_dsa_hb_bench.rs; takes the rows.
pub fn worst_ulp(c: &Case, pool: &Pool, out: &[f32], rows: impl IntoIterator<Item = usize>) -> f64 {
    let mut worst = 0f64;
    for r in rows {
        for h in 0..c.heads {
            let o = reference(c, pool, r, h);
            let ulp = o.iter().fold(0f64, |a, x| a.max(x.abs())).max(1e-30) / 256.0;
            for (i, x) in o.iter().enumerate() {
                worst = worst.max((out[(r * c.heads + h) * KVL + i] as f64 - x).abs() / ulp);
            }
        }
    }
    worst
}

/// 2026-10-12: One launch of a head-batched entry (`glm5next_dsa_mla_decode_hb_fp8`, its P-hi
/// twin, or 2026-10-14 `glm5next_dsa_mla_prefill_pipe_phi` with its `smem` dynamic bytes) over
/// `v`, then the merge from two splits up. 2026-10-14: moved here from glm5next_dsa_hb_bench.rs.
#[allow(clippy::too_many_arguments)]
pub fn launch_hb_entry(
    g: &dyn GpuBackend,
    entry: KernelHandle,
    smem: u32,
    merge: KernelHandle,
    cache: DevicePtr,
    v: View,
    out: DevicePtr,
    ws: (DevicePtr, DevicePtr),
    splits: usize,
) -> Result<()> {
    KernelLaunch::new(g, entry)
        .grid([
            splits as u32,
            v.rows as u32,
            v.heads.div_ceil(HB_HEADS) as u32,
        ])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(v.q)
        .arg_ptr(cache)
        .arg_ptr(out)
        .arg_ptr(v.bt)
        .arg_ptr(v.sl)
        .arg_ptr(v.sel)
        .arg_ptr(ws.0)
        .arg_ptr(ws.1)
        .arg_u32(WIDTH as u32)
        .arg_u32(v.bps as u32)
        .arg_u32(v.heads as u32)
        .arg_u32(BLOCK as u32)
        .arg_f32((KVL as f32).powf(-0.5))
        .arg_f32(1.0)
        .arg_f32(1.0)
        .arg_u64((BLOCK * KVL) as u64)
        .arg_u32(splits as u32)
        .launch(0)?;
    if splits > 1 {
        KernelLaunch::new(g, merge)
            .grid([v.heads as u32, v.rows as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(ws.0)
            .arg_ptr(ws.1)
            .arg_ptr(v.sl)
            .arg_ptr(out)
            .arg_u32(v.heads as u32)
            .arg_u32(splits as u32)
            .arg_f32(1.0)
            .launch(0)?;
    }
    Ok(())
}
