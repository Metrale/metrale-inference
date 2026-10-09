// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The paged decode attention reference as a [`RefImpl`]: [`super::attention`] (case,
//! bound) plus a conforming f32 emulation of the online softmax and the `kv_page_swap`
//! mutation.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Every emulation bracketing stays within the declared depths: the dot through
//!   [`crate::emulate::reduce`], the context in at most `ctx` combines per position.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use super::RefImpl;
use super::attention::{self, Formats, Geom};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: `paged_attention`: decode attention over a paged KV cache.
pub struct PagedAttention;

/// 2026-10-09: Warps of one CTA, each a contiguous chunk of the context, merged pairwise
/// (paged_decode_attn.cu:27, :104-112, :275-298): the bracketing of variants 0 and 1.
const CHUNKS: usize = 8;

impl RefImpl for PagedAttention {
    fn name(&self) -> &'static str {
        "paged_attention"
    }

    fn serves(&self, op: &str) -> bool {
        op == "paged_attention"
    }

    fn lens(&self, shape: &Shape, _pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        attention::lens(shape)
    }

    fn fill(
        &self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        stream: &dyn Fn(&str) -> SplitMix64,
    ) -> Result<(), String> {
        attention::fill(case, plan, shape, class, stream)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        attention::reference(case, plan, idx)
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        let mut f = attention::formats(plan)?;
        if let Some(a) = acc {
            f.acc = a;
        }
        let g = Geom::of(case)?;
        let mut out = Vec::with_capacity(idx.len());
        let mut cached: Option<(usize, Vec<f64>)> = None;
        for &i in idx {
            let key = i / g.head_dim;
            if cached.as_ref().is_none_or(|(k, _)| *k != key) {
                let (r, h) = (key / g.q_heads, key % g.q_heads);
                if r >= g.rows {
                    return Err(format!("index {i} beyond {} rows", g.rows));
                }
                cached = Some((key, head(case, plan, &f, &g, r, h, variant)?));
            }
            out.push(cached.as_ref().map_or(f64::NAN, |(_, o)| o[i % g.head_dim]));
        }
        Ok(out)
    }

    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        match m {
            Mutation::KvPageSwap => page_swap(case, rng),
            other => Err(format!(
                "`{}` does not apply to paged attention",
                other.name()
            )),
        }
    }

    fn strides(&self, case: &Case) -> Vec<usize> {
        Geom::of(case).map(|g| vec![g.head_dim]).unwrap_or_default()
    }
}

/// 2026-10-09: Swap the last (partial) page of the shortest multi-page sequence whose length is
/// not a page multiple with one of its full pages: the kernel then attends the slots past the
/// end of one page and drops attended positions of the other. The reference keeps the true
/// table. Reaches every output of that sequence.
fn page_swap(case: &mut Case, rng: &mut SplitMix64) -> Result<Vec<usize>, String> {
    let g = Geom::of(case)?;
    let lens = case.tensor("seq_lens")?.values();
    let r = (0..g.rows)
        .filter(|&r| {
            let l = lens[r] as usize;
            l > g.page && l % g.page != 0
        })
        .min_by_key(|&r| lens[r] as usize)
        .ok_or("no sequence spans two pages with a partial last page")?;
    let last = (lens[r] as usize).div_ceil(g.page) - 1;
    let full = rng.below(last as u64) as usize;
    let t = case
        .tensors
        .get_mut("block_table")
        .ok_or("the case has no block table")?;
    let bytes = std::sync::Arc::make_mut(&mut t.bytes);
    let (a, b) = (4 * (r * g.blocks + full), 4 * (r * g.blocks + last));
    for k in 0..4 {
        bytes.swap(a + k, b + k);
    }
    Ok((0..g.width()).map(|c| r * g.width() + c).collect())
}

/// 2026-10-09: An online-softmax state: running max, normaliser, unnormalised output.
struct State {
    m: f64,
    l: f64,
    o: Vec<f64>,
}

/// 2026-10-09: `log2(e)` as the f32 constant `__expf` multiplies by.
const LOG2E_F32: f64 = 1.442_695_021_629_333_496_093_75;

struct Arith {
    f: Formats,
}

impl Arith {
    fn rnd(e: Elem, x: f64) -> f64 {
        e.round_saturating(x).unwrap_or(f64::NAN)
    }

    /// 2026-10-09: `exp(a - b)` as the kernel's `__expf` computes it: the rounded difference,
    /// times the f32 log2(e), rounded, then `exp2` correctly rounded (within the declared
    /// error of `ex2.approx`).
    fn exp_diff(&self, a: f64, b: f64) -> f64 {
        let s = self.f.softmax;
        let t = Self::rnd(s, Self::rnd(s, a - b) * LOG2E_F32);
        Self::rnd(s, t.exp2())
    }

    /// 2026-10-09: Merge two states: each rescaled to the common max, then added.
    fn merge(&self, a: State, b: State) -> State {
        let (s, acc) = (self.f.softmax, self.f.acc);
        let m = a.m.max(b.m);
        let (fa, fb) = (self.exp_diff(a.m, m), self.exp_diff(b.m, m));
        let l = Self::rnd(s, Self::rnd(s, a.l * fa) + Self::rnd(s, b.l * fb));
        let o =
            a.o.iter()
                .zip(&b.o)
                .map(|(x, y)| Self::rnd(acc, Self::rnd(acc, x * fa) + Self::rnd(acc, y * fb)))
                .collect();
        State { m, l, o }
    }

    fn fold(&self, states: impl Iterator<Item = State>) -> Option<State> {
        states.fold(None, |acc, s| {
            Some(match acc {
                Some(a) => self.merge(a, s),
                None => s,
            })
        })
    }

    /// 2026-10-09: The kernel's warp merge: slot `w` takes slot `w + stride` for strides 4, 2, 1;
    /// an empty slot contributes nothing (its `l == 0` skip).
    fn strided(&self, mut slots: Vec<Option<State>>) -> Option<State> {
        let mut stride = slots.len() / 2;
        while stride > 0 {
            for w in 0..stride {
                if let Some(b) = slots[w + stride].take() {
                    slots[w] = Some(match slots[w].take() {
                        Some(a) => self.merge(a, b),
                        None => b,
                    });
                }
            }
            stride /= 2;
        }
        slots.into_iter().next().flatten()
    }

    fn tree(&self, mut states: Vec<State>) -> Option<State> {
        while states.len() > 1 {
            let mut next = Vec::with_capacity(states.len().div_ceil(2));
            let mut it = states.into_iter();
            while let Some(a) = it.next() {
                next.push(match it.next() {
                    Some(b) => self.merge(a, b),
                    None => a,
                });
            }
            states = next;
        }
        states.pop()
    }
}

/// 2026-10-09: One (sequence, head)'s output row in the emulation, rounded into the output
/// format. Variant 0: the kernel's eight contiguous chunks, each sequential, merged pairwise;
/// 1: the same over the reversed positions; 2: a balanced tree over single positions. The dot
/// products are bracketed per [`crate::emulate::reduce`] in the same variant.
fn head(
    case: &Case,
    plan: &Plan,
    f: &Formats,
    g: &Geom,
    r: usize,
    h: usize,
    variant: u32,
) -> Result<Vec<f64>, String> {
    let (q, kc, vc) = (
        case.tensor("q")?,
        case.tensor("k_cache")?,
        case.tensor("v_cache")?,
    );
    let bt = case.tensor("block_table")?;
    let len = case.tensor("seq_lens")?.get(r) as usize;
    let depth_d = plan.depth_of("d")?;
    let scale = case.scalar("sm_scale")?;
    let kvh = h / (g.q_heads / g.kv_heads);
    let product_rounds = f.q.precision + f.cache.precision > f.scores.precision;
    let ar = Arith { f: *f };
    let single = |j: usize| {
        let terms: Vec<f64> = (0..g.head_dim)
            .map(|i| {
                let p =
                    q.get(r * g.width() + h * g.head_dim + i) * kc.get(g.kv_at(bt, r, j, kvh, i));
                if product_rounds {
                    Arith::rnd(f.scores, p)
                } else {
                    p
                }
            })
            .collect();
        let dot = crate::emulate::reduce(&terms, f.scores, depth_d, variant);
        State {
            m: Arith::rnd(f.scores, dot * scale),
            l: 1.0,
            o: (0..g.head_dim)
                .map(|i| vc.get(g.kv_at(bt, r, j, kvh, i)))
                .collect(),
        }
    };
    let state = match variant % crate::emulate::VARIANTS {
        2 => ar.tree((0..len).map(single).collect()),
        v => {
            let order: Vec<usize> = if v == 1 {
                (0..len).rev().collect()
            } else {
                (0..len).collect()
            };
            let chunk = len.div_ceil(CHUNKS).max(1);
            let mut warps: Vec<Option<State>> = order
                .chunks(chunk)
                .map(|c| ar.fold(c.iter().map(|&j| single(j))))
                .collect();
            warps.resize_with(CHUNKS, || None);
            ar.strided(warps)
        }
    }
    .ok_or_else(|| format!("sequence {r} attends no position"))?;
    let inv = Arith::rnd(f.softmax, 1.0 / state.l);
    Ok(state
        .o
        .iter()
        .map(|o| Arith::rnd(f.out, Arith::rnd(f.acc, o * inv)))
        .collect())
}
