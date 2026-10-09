// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The paged decode attention reference: per sequence `r` and query head `h`, with
//! `g = h / (q_heads / kv_heads)` its KV head, scores `s_j = (q_rh . k_jg) * sm_scale` over the
//! sequence's `L_r` cached positions, read through its block table, and the output
//! `o_rh = sum_j softmax(s)_j v_jg`, following the family's declared pipeline: the cache format,
//! the score precision, the softmax precision (running max, exponentials, the normaliser) and
//! the accumulator of the PV sums.
//!
//! Tensors of a case: `q` `[rows, q_heads * head_dim]`, `k_cache` and `v_cache`
//! `[pages, PAGE, kv_heads, head_dim]` (the NHD paged layout), `block_table` `[rows, blocks]`
//! (i32 page ids), `seq_lens` `[rows]` (i32), scalar `sm_scale`. Output
//! `[rows, q_heads * head_dim]`.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The bound covers any online-softmax kernel that brackets the head-dim dot in a tree of the
//!   declared `d` depth and the context in a tree of the declared `ctx` depth, rescaling each
//!   operand of a combine at most once before it (the running-max form), computing `exp` as
//!   `exp2` of the argument times log2(e) with the declared `exp2` error, and dividing once.
//! - Page slots past a sequence's length hold data (stale KV), so reading one is a wrong value.

use std::collections::BTreeMap;

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::{Num, StepKind, Value};

use crate::bounded::{Bounded, gamma, sum};
use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, Elem, F32_FTZ, F64};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: Tokens per KV page: the serve default `--block-size 16` (server
/// `cli/serve_args.rs`), which the engine's block tables are built with.
pub const PAGE: usize = 16;

/// 2026-10-09: Context lengths, sequence `r` taking `CONTEXTS[r % 8]`: up to 4k tokens (CPU-cheap,
/// past the kernel's per-warp batching), most not a multiple of [`PAGE`] (a partial last page),
/// one a single token, two exact page multiples.
pub const CONTEXTS: [usize; 8] = [4093, 17, 2049, 1000, 1, 3584, 300, 64];

/// 2026-10-09: Pages a case's pool holds beyond the longest sequence's two copies: the pool is
/// capped there so a wide batch shares pages (as prefix caching does) instead of allocating
/// every sequence's own; within a sequence the pages are distinct.
const POOL_SPARE: usize = 8;

/// 2026-10-09: The sizes of one launch, read from its tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geom {
    /// 2026-10-09: Sequences (one query row each).
    pub rows: usize,
    /// 2026-10-09: Query heads.
    pub q_heads: usize,
    /// 2026-10-09: KV heads.
    pub kv_heads: usize,
    /// 2026-10-09: Head dim.
    pub head_dim: usize,
    /// 2026-10-09: Tokens per page.
    pub page: usize,
    /// 2026-10-09: Block-table entries per sequence.
    pub blocks: usize,
}

impl Geom {
    /// 2026-10-09: The geometry of `case`.
    pub fn of(case: &Case) -> Result<Geom, String> {
        let (q, k) = (case.tensor("q")?, case.tensor("k_cache")?);
        let bt = case.tensor("block_table")?;
        if k.dims.len() != 4 || q.dims.len() != 2 || bt.dims.len() != 2 {
            return Err(format!(
                "q {:?}, k_cache {:?}, block_table {:?}",
                q.dims, k.dims, bt.dims
            ));
        }
        let (page, kv_heads, head_dim) = (k.dims[1], k.dims[2], k.dims[3]);
        if head_dim == 0 || q.dims[1] % head_dim != 0 {
            return Err(format!("q width {} over head_dim {head_dim}", q.dims[1]));
        }
        let q_heads = q.dims[1] / head_dim;
        if kv_heads == 0 || q_heads % kv_heads != 0 || bt.dims[0] != q.dims[0] {
            return Err(format!("{q_heads} q heads over {kv_heads} kv heads"));
        }
        Ok(Geom {
            rows: q.dims[0],
            q_heads,
            kv_heads,
            head_dim,
            page,
            blocks: bt.dims[1],
        })
    }

    /// 2026-10-09: Output width.
    pub fn width(&self) -> usize {
        self.q_heads * self.head_dim
    }

    /// 2026-10-09: Flat index of element `i` of the cache row at position `j` of sequence `r`,
    /// KV head `g`, through the block table.
    pub fn kv_at(&self, bt: &Tensor, r: usize, j: usize, g: usize, i: usize) -> usize {
        let page = bt.get(r * self.blocks + j / self.page) as usize;
        ((page * self.page + j % self.page) * self.kv_heads + g) * self.head_dim + i
    }
}

/// 2026-10-09: The formats of a paged-attention plan.
#[derive(Debug, Clone, Copy)]
pub struct Formats {
    /// 2026-10-09: The query as read.
    pub q: Elem,
    /// 2026-10-09: The KV cache as stored.
    pub cache: Elem,
    /// 2026-10-09: The QK dot products and the scaled score.
    pub scores: Elem,
    /// 2026-10-09: Running max, exponentials, their arguments, the normaliser and its reciprocal.
    pub softmax: Elem,
    /// 2026-10-09: The PV sums, their rescales and the final product.
    pub acc: Elem,
    /// 2026-10-09: Output.
    pub out: Elem,
}

fn num(n: Num, ftz: bool) -> Elem {
    let e = elem::of_num(n);
    if ftz && n == Num::F32 { F32_FTZ } else { e }
}

fn layout(f: Format) -> Result<Elem, String> {
    match f {
        Format::Bf16 => Ok(elem::BF16),
        Format::F32 => Ok(elem::F32),
        other => Err(format!(
            "a {} attention operand (the reference reads bf16 or f32)",
            other.name()
        )),
    }
}

/// 2026-10-09: The formats `plan` declares.
pub fn formats(plan: &Plan) -> Result<Formats, String> {
    let p = &plan.pipeline;
    let step = |k: StepKind| match plan.step(k) {
        Some(Value::Num(n)) => Ok(num(*n, plan.ftz)),
        _ => Err(format!("the pipeline has no `{k:?}` precision")),
    };
    let cache = match plan.step(StepKind::Cache) {
        Some(Value::Kv(s)) if s == "bf16" => elem::BF16,
        Some(Value::Kv(s)) => {
            return Err(format!(
                "a `{s}` cache (the reference reads a bf16 cache, without scales)"
            ));
        }
        _ => return Err("the pipeline has no `cache` step".into()),
    };
    Ok(Formats {
        q: layout(*p.inputs.first().ok_or("the pipeline names no input")?)?,
        cache,
        scores: step(StepKind::Scores)?,
        softmax: step(StepKind::Softmax)?,
        acc: step(StepKind::Accumulate)?,
        out: layout(*p.outputs.first().ok_or("the pipeline names no output")?)?,
    })
}

fn runtime(shape: &Shape, name: &str) -> Result<usize, String> {
    shape
        .runtime
        .get(name)
        .and_then(|v| v.parse().ok())
        .filter(|v: &usize| *v > 0)
        .ok_or_else(|| format!("the shape states no runtime `{name}`"))
}

/// 2026-10-09: Head dim and the longest context of `shape`'s launch (the `d` and `ctx` lengths
/// of the contract's reductions). A shape without `q_heads` gets no `d` and one without rows no
/// `ctx`, so the plan names the missing length instead of assuming one.
pub fn lens(shape: &Shape) -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    if let Ok(q_heads) = runtime(shape, "q_heads") {
        m.insert("d".to_string(), shape.in_dim / q_heads as u64);
    }
    if let Some(ctx) = (0..shape.rows as usize)
        .map(|r| CONTEXTS[r % CONTEXTS.len()])
        .max()
    {
        m.insert("ctx".to_string(), ctx as u64);
    }
    m
}

fn shuffled(n: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.below(i as u64 + 1) as usize);
    }
    v
}

/// 2026-10-09: Fill `case` with one decode launch of `shape` for `class`. The class shapes the
/// queries and the values; the keys are drawn as a model holds them (unit-RMS gaussian, the
/// scale qk-norm leaves). Every page of the pool is drawn, including the slots past a
/// sequence's end, and each sequence's block table is a seeded shuffle of the pool.
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    shape: &Shape,
    class: InputClass,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<(), String> {
    let f = formats(plan)?;
    let (q_heads, kv_heads) = (runtime(shape, "q_heads")?, runtime(shape, "kv_heads")?);
    let width = shape.in_dim as usize;
    if shape.rows == 0
        || width % q_heads != 0
        || q_heads % kv_heads != 0
        || shape.out_dim != shape.in_dim
    {
        return Err(format!(
            "a {}-row [{width} -> {}] attention over {q_heads} q / {kv_heads} kv heads",
            shape.rows, shape.out_dim
        ));
    }
    let (rows, hd) = (shape.rows as usize, width / q_heads);
    let ctx: Vec<usize> = (0..rows).map(|r| CONTEXTS[r % CONTEXTS.len()]).collect();
    let need: Vec<usize> = ctx.iter().map(|l| l.div_ceil(PAGE)).collect();
    let longest = need.iter().copied().max().ok_or("no sequence")?;
    let pool = (need.iter().sum::<usize>() + POOL_SPARE).min(2 * longest + POOL_SPARE);
    let blocks = longest + 1;
    let mut rt = stream("block_table");
    let mut table = Vec::with_capacity(rows * blocks);
    for _ in 0..rows {
        table.extend(shuffled(pool, &mut rt)[..blocks].iter().map(|&p| p as f64));
    }
    let q = tensor(&mut stream("q"), class, rows, width, 1.0, f.q);
    let slots = pool * PAGE * kv_heads;
    let k = tensor(
        &mut stream("k"),
        InputClass::Gaussian,
        slots,
        hd,
        1.0,
        f.cache,
    );
    let v = tensor(&mut stream("v"), class, slots, hd, 1.0, f.cache);
    let enc = |e: Elem| if e == elem::BF16 { Enc::Bf16 } else { Enc::F32 };
    let cache_dims = vec![pool, PAGE, kv_heads, hd];
    let t = &mut case.tensors;
    t.insert("q".into(), Tensor::encode(enc(f.q), vec![rows, width], &q)?);
    t.insert(
        "k_cache".into(),
        Tensor::encode(enc(f.cache), cache_dims.clone(), &k)?,
    );
    t.insert(
        "v_cache".into(),
        Tensor::encode(enc(f.cache), cache_dims, &v)?,
    );
    t.insert(
        "block_table".into(),
        Tensor::encode(Enc::I32, vec![rows, blocks], &table)?,
    );
    let lens: Vec<f64> = ctx.iter().map(|&l| l as f64).collect();
    t.insert(
        "seq_lens".into(),
        Tensor::encode(Enc::I32, vec![rows], &lens)?,
    );
    let scale = elem::F32
        .round(1.0 / (hd as f64).sqrt())
        .ok_or("sm_scale")?;
    case.scalars.insert("sm_scale".into(), scale);
    case.out = (vec![rows, width], enc(f.out));
    Ok(())
}

/// 2026-10-09: `-ln(1 - x)`: the log-size of a factor in `[1 - x, 1 + x]`.
fn ln_factor(x: f64) -> f64 {
    if x >= 1.0 {
        f64::INFINITY
    } else {
        -(-x).ln_1p()
    }
}

/// 2026-10-09: The bounded reference of one (sequence, head) at head-dim columns `cols`.
///
/// Write `w_j = exp(s_j - max s)`, `p_j = w_j / sum w`. A conforming kernel computes
/// `O_i = C sum_j w_j c_j (1 + t_ij) v_ij` and `L = C sum_j w_j c_j (1 + f_j)` for a common
/// `C > 0` (the offset of its running max, which cancels): `c_j` collects the score error, the
/// approximate exponentials and their rounded arguments (shared by O and L, since each rescale
/// factor multiplies both), `t`, `f` the roundings of products, rescales and adds. With
/// `|c_j(1+t_ij) - 1| <= gO_j`, `|c_j(1+f_j) - 1| <= gL_j`:
/// `|O_i/L - o_i| <= sum_j p_j (|v_ij| gO_j + |o_i| gL_j) / (1 - sum_j p_j gL_j)`, since
/// `sum_j p_j (v_ij - o_i) = 0`. The final `fl(O_i * fl(1/L))` adds two roundings.
pub fn group(
    case: &Case,
    plan: &Plan,
    r: usize,
    h: usize,
    cols: &[usize],
) -> Result<Vec<Bounded>, String> {
    let f = formats(plan)?;
    let g = Geom::of(case)?;
    let (q, kc, vc) = (
        case.tensor("q")?,
        case.tensor("k_cache")?,
        case.tensor("v_cache")?,
    );
    let bt = case.tensor("block_table")?;
    let len = case.tensor("seq_lens")?.get(r) as usize;
    if len == 0 || len > g.blocks * g.page {
        return Err(format!("sequence {r} has length {len}"));
    }
    let (depth_d, depth_ctx) = (plan.depth_of("d")?, plan.depth_of("ctx")?);
    let eps_exp2 = plan.approx_of("exp2")?;
    let scale = Bounded::exact(case.scalar("sm_scale")?);
    let kvh = h / (g.q_heads / g.kv_heads);
    let qv = |i: usize| q.get(r * g.width() + h * g.head_dim + i);
    let product_rounds = f.q.precision + f.cache.precision > f.scores.precision;
    let scores: Vec<Bounded> = (0..len)
        .map(|j| {
            let terms: Vec<Bounded> = (0..g.head_dim)
                .map(|i| {
                    let p = Bounded::exact(qv(i))
                        .mul(Bounded::exact(kc.get(g.kv_at(bt, r, j, kvh, i))));
                    if product_rounds {
                        p.round(f.scores)
                    } else {
                        // 2026-10-09: Exact above the subnormal range; the floor covers below.
                        Bounded {
                            v: p.v,
                            e: p.e + f.scores.underflow_floor(),
                        }
                    }
                })
                .collect();
            sum(&terms, f.scores, depth_d).mul(scale).round(f.scores)
        })
        .collect();
    let top = scores.iter().map(|s| s.v).fold(f64::NEG_INFINITY, f64::max);
    let worst = scores.iter().map(|s| s.e).fold(0.0, f64::max);
    let w: Vec<f64> = scores.iter().map(|s| (s.v - top).exp()).collect();
    let total: f64 = w.iter().sum();
    let (us, ua) = (f.softmax.unit_roundoff(), f.acc.unit_roundoff());
    // 2026-10-09: An exponential's argument is a rounded difference times a rounded log2(e),
    // rounded: a relative error kappa of the argument, whose sizes along one term's rescale
    // chain telescope to (running max at the end - s_j) <= top + worst - (s_j - e_j).
    let kappa = (1.0 + us).powi(3) - 1.0;
    let n_exp = depth_ctx + 1;
    let (round_o, round_l) = (
        (2 * depth_ctx + 1) as f64 * ln_factor(ua),
        (2 * depth_ctx) as f64 * ln_factor(us),
    );
    // 2026-10-09: The f64 evaluation of w, p and the sums over `len` terms.
    let f64_rel = gamma(len as u64 + 8, F64.unit_roundoff());
    let (mut g_o, mut g_l) = (Vec::with_capacity(len), Vec::with_capacity(len));
    for s in &scores {
        let gap = top - s.v;
        let common = s.e
            + n_exp as f64 * ln_factor(eps_exp2)
            + kappa * (gap + s.e + worst)
            + f64_rel * (1.0 + gap);
        g_o.push((common + round_o).exp_m1());
        g_l.push((common + round_l).exp_m1());
    }
    let p: Vec<f64> = w.iter().map(|x| x / total).collect();
    let spread_l: f64 = p.iter().zip(&g_l).map(|(p, g)| p * g).sum();
    // 2026-10-09: Below the normal range the relative model fails: an exponential is off by
    // up to the smallest normal (`__expf` may lower to the flushing `ex2.approx.ftz`, which
    // also covers 2 ulp plus the rounding of a subnormal result), a product or rescale by the
    // format's floor, and a rescale's floor multiplies a partial of at most `len` terms. In
    // units of the kernel's normaliser (which is at least exp(-worst) of `total`).
    let floor_exp = elem::pow2(f.softmax.emin);
    let lf = len as f64;
    let vmax = (0..g.head_dim)
        .flat_map(|i| (0..len).map(move |j| (i, j)))
        .map(|(i, j)| vc.get(g.kv_at(bt, r, j, kvh, i)).abs())
        .fold(0.0, f64::max);
    let grow = 1.0 + g_l.iter().copied().fold(0.0, f64::max);
    let to_p = worst.exp() / total;
    let abs_l = lf
        * (n_exp as f64 * floor_exp * lf * grow
            + (2 * depth_ctx) as f64 * f.softmax.underflow_floor())
        * to_p;
    let abs_o = lf
        * (n_exp as f64 * floor_exp * lf * grow * vmax
            + (2 * depth_ctx + 1) as f64 * f.acc.underflow_floor())
        * to_p;
    let den = 1.0 - spread_l - abs_l;
    let last = ((1.0 + ua) * (1.0 + us) - 1.0, f.acc.underflow_floor());
    cols.iter()
        .map(|&i| {
            let vi: Vec<f64> = (0..len)
                .map(|j| vc.get(g.kv_at(bt, r, j, kvh, i)))
                .collect();
            let o: f64 = p.iter().zip(&vi).map(|(p, v)| p * v).sum();
            let mag: f64 = p.iter().zip(&vi).map(|(p, v)| p * v.abs()).sum();
            if den <= 0.0 {
                return Ok(Bounded {
                    v: o,
                    e: f64::INFINITY,
                });
            }
            let num: f64 = p
                .iter()
                .zip(&vi)
                .zip(g_o.iter().zip(&g_l))
                .map(|((p, v), (go, gl))| p * (v.abs() * go + o.abs() * gl))
                .sum();
            let b = (num + abs_o + o.abs() * abs_l) / den;
            let e = b + (o.abs() + b) * last.0 + last.1 + 4.0 * f64_rel * (mag + o.abs());
            Ok(Bounded {
                v: o,
                e: e * (1.0 + f64_rel),
            })
        })
        .collect()
}

/// 2026-10-09: The bounded reference at flat output indices `idx` (`r * width + h * hd + i`),
/// before the kernel's final rounding into the output format. Consecutive indices of one
/// (sequence, head) share its scores.
pub fn reference(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
    let g = Geom::of(case)?;
    let mut out = Vec::with_capacity(idx.len());
    let mut at = 0;
    while at < idx.len() {
        let key = idx[at] / g.head_dim;
        let end = idx[at..]
            .iter()
            .position(|&i| i / g.head_dim != key)
            .map_or(idx.len(), |n| at + n);
        let (r, h) = (key / g.q_heads, key % g.q_heads);
        if r >= g.rows {
            return Err(format!("index {} beyond {} rows", idx[at], g.rows));
        }
        let cols: Vec<usize> = idx[at..end].iter().map(|i| i % g.head_dim).collect();
        out.extend(group(case, plan, r, h, &cols)?);
        at = end;
    }
    Ok(out)
}
