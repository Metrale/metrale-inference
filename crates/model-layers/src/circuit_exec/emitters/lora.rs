// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: LoRA adapters under the circuit (LIFECYCLE-DESIGN.md 15.10): the binding types
//! a model hands the executor ([`LoraBoot`], [`LoraFixed`], [`LoraLayer`]) and the two emitters
//! of FUSIONS.toml's LoRA section. `lora_bgmv` mirrors legacy's routed attention fold
//! (`ops::lora_delta::apply_lora_bgmv`, each row's adapter slot); `lora_pair` mirrors its
//! active-adapter fold of the FFN and the GatedDeltaNet out_proj
//! (`ops::lora_delta::apply_lora_delta`).
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants:
//! - The adapted output is the projection's output, folded in place (an alias), as legacy folds
//!   into `base_out`.
//! - Every launch reads a fixed address: the pool's tables and pairs, the arena's `lora_xa` /
//!   `lora_delta` scratch, the mode's per-row slot upload. A rotation of the active adapter
//!   rebuilds the executor, as it drains legacy's graphs.
//! - An emitter refuses a projection its layer binds no adapter for: a plan never claims a fold
//!   it does not launch.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::planner::Layout;
use metrale_circuit::{LinearRole, Mode};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::compile::{Cx, GroupRef, OpEmitter};
use crate::layers::ops;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair, LoraRoute};

/// 2026-10-03: What a build with adapters adapts: the spec `metrale_circuit::lora::adapt`
/// rewrites the circuit with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraBoot {
    pub spec: metrale_circuit::lora::LoraSpec,
}

/// 2026-10-03: The fixed device addresses the LoRA launches read.
#[derive(Debug, Clone, Copy)]
pub struct LoraFixed {
    /// 2026-10-03: `BufferArena::lora_xa`: the shrink's output, `rows x max_rank` BF16.
    pub xa: DevicePtr,
    /// 2026-10-03: `BufferArena::lora_delta`: the pair's expand output, one row of `n_out`.
    pub delta: DevicePtr,
    /// 2026-10-03: Where each mode's step uploads its rows' adapter slots (`i32`, `-1` resolved
    /// to the active adapter): one-row decode, multi-sequence decode, the MTP verify.
    pub slots_decode: DevicePtr,
    pub slots_multi_seq: DevicePtr,
    pub slots_verify: DevicePtr,
}

/// 2026-10-03: One layer's adapters: the routing tables of the projections the pool adapts per
/// row, and the active adapter's pairs (two for `gate_up`: gate, then up).
#[derive(Clone)]
pub struct LoraLayer {
    pub kernels: LoraKernels,
    pub routes: BTreeMap<LinearRole, LoraRoute>,
    pub pairs: BTreeMap<LinearRole, Vec<LoraPair>>,
}

impl std::fmt::Debug for LoraLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let routes: Vec<&str> = self.routes.keys().map(|r| r.name()).collect();
        let pairs: Vec<(&str, usize)> = self
            .pairs
            .iter()
            .map(|(r, p)| (r.name(), p.len()))
            .collect();
        write!(f, "LoraLayer {{ routes: {routes:?}, pairs: {pairs:?} }}")
    }
}

/// 2026-10-03: The widest step LoRA phase 1 serves under the circuit. Above 4 rows legacy's
/// adapted FFN takes its non-MMQ wide arm (`dense_ffn_nvfp4_plan.rs`), which no rule models yet
/// (the MMQ arms are gated off under `lora_active = "on"`, so such a plan would not fuse).
pub const PHASE1_MAX_ROWS: u64 = 4;

/// 2026-10-03: The overlay spec of a pool of `rank` over `layers`' bindings: per layer, the
/// projections its adapters fold into.
pub fn spec_of(rank: u64, layers: &[Option<super::super::bindings::CircuitLayer>]) -> LoraBoot {
    let layers = layers
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let a = l.as_ref()?.lora.as_ref()?;
            Some((i, a.routes.keys().chain(a.pairs.keys()).copied().collect()))
        })
        .collect();
    LoraBoot {
        spec: metrale_circuit::lora::LoraSpec { rank, layers },
    }
}

/// 2026-10-03: Every emitter of this module.
pub(super) static ALL: &[&dyn OpEmitter] = &[&LoraBgmv, &LoraPairFold];

/// 2026-10-03: The role an adapter node adapts (its `role` param, set by `adapt`).
fn role(cx: &Cx<'_>) -> Result<LinearRole> {
    let n = cx.g.node(0);
    n.params
        .get("role")
        .and_then(|r| LinearRole::parse(r))
        .with_context(|| format!("`{}` names no adapted role", n.id))
}

/// 2026-10-03: The layer's adapters; an error for a layer bound without them.
fn adapters<'a>(cx: &Cx<'a>) -> Result<&'a LoraLayer> {
    cx.layer(0)?
        .lora
        .as_ref()
        .with_context(|| format!("`{}`: its layer binds no adapters", cx.g.node(0).id))
}

fn fixed(cx: &Cx<'_>) -> Result<LoraFixed> {
    cx.fixed
        .lora
        .context("a LoRA plan without the adapters' fixed buffers")
}

/// 2026-10-03: The adapted output folds in place over the projection's output.
fn alias(g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
    g.expect_ops("lora", &["lora_shrink", "lora_expand"])?;
    layout.aliases.push((g.output(1, 0)?, g.input(1, 1)?));
    Ok(())
}

/// 2026-10-03: `lora_bgmv`: the routed fold, one adapter slot per row (`lora_bgmv_shrink`, then
/// `lora_bgmv_expand_fold`).
pub(crate) struct LoraBgmv;

impl OpEmitter for LoraBgmv {
    fn id(&self) -> &'static str {
        "lora_bgmv"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        alias(g, layout)
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lora_shrink", "lora_expand"])?;
        super::expect_kernel(cx, 0, "lora_bgmv_shrink")?;
        super::expect_kernel(cx, 1, "lora_bgmv_expand_fold")?;
        let r = role(cx)?;
        let ad = adapters(cx)?;
        let route = *ad
            .routes
            .get(&r)
            .with_context(|| format!("no routing table for `{}`", r.name()))?;
        let f = fixed(cx)?;
        let slots = match cx.mode {
            Mode::Decode => f.slots_decode,
            Mode::MultiSeq => f.slots_multi_seq,
            Mode::Verify => f.slots_verify,
            other => bail!("no adapter slots are uploaded for {other:?}"),
        };
        let (x, xs) = cx.strided(cx.g.input(0, 0)?)?;
        let (out, os) = cx.strided(cx.g.input(1, 1)?)?;
        ensure!(
            xs >= route.k_in && os >= route.n_out,
            "`{}`: rows narrower than the adapter",
            r.name()
        );
        let n = super::rows(cx)?;
        let mut k = ad.kernels;
        k.bgmv_shrink_k = cx.handle(0)?;
        k.bgmv_expand_fold_k = cx.handle(1)?;
        cx.push(
            0,
            Box::new(move |e| {
                ops::lora_delta::lora_bgmv_shrink(
                    e.gpu,
                    &k,
                    &route,
                    (x, xs),
                    slots,
                    n,
                    f.xa,
                    e.stream,
                )
            }),
        )?;
        cx.push(
            1,
            Box::new(move |e| {
                ops::lora_delta::lora_bgmv_expand_fold(
                    e.gpu,
                    &k,
                    &route,
                    (out, os),
                    slots,
                    n,
                    f.xa,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: `lora_pair`: the active adapter's fold. Per row, the shrink GEMV, the expand GEMV
/// and the scaled add (`apply_lora_delta` at one row, which legacy loops up to 48 rows). Pairs in
/// order (gate, then up), rows inside each, as legacy.
pub(crate) struct LoraPairFold;

impl OpEmitter for LoraPairFold {
    fn id(&self) -> &'static str {
        "lora_pair"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        alias(g, layout)
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lora_shrink", "lora_expand"])?;
        let r = role(cx)?;
        let pairs = adapters(cx)?
            .pairs
            .get(&r)
            .with_context(|| format!("no active pair for `{}`", r.name()))?
            .clone();
        let halves = if r == LinearRole::GateUp { 2 } else { 1 };
        ensure!(
            pairs.len() == halves,
            "`{}` binds {} pairs; the projection takes {halves}",
            r.name(),
            pairs.len()
        );
        let f = fixed(cx)?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let y = cx.ptr(cx.g.input(1, 1)?)?;
        let rows = cx.rows as usize;
        ensure!(
            cx.g.group.kernels.len() == 3 * halves,
            "rule `{}` lists {} kernels; the per-row pair of {halves} adapters launches {}",
            cx.g.group.rule,
            cx.g.group.kernels.len(),
            3 * halves
        );
        let mut half_at = 0usize;
        for (p, pair) in pairs.into_iter().enumerate() {
            let base = y.offset(half_at);
            half_at += rows * pair.n_out as usize * 2;
            let (gemv, add) = (cx.handle(3 * p)?, cx.handle(3 * p + 2)?);
            ensure!(
                cx.handle(3 * p + 1)?.0 == gemv.0,
                "the pair's two GEMVs are one kernel"
            );
            for row in 0..rows {
                let xr = x.offset(row * pair.k_in as usize * 2);
                let or = base.offset(row * pair.n_out as usize * 2);
                cx.push(
                    3 * p,
                    Box::new(move |e| {
                        let (a, k_in) = (&pair.a, pair.k_in);
                        ops::dense_gemv(e.gpu, gemv, xr, a, f.xa, pair.max_rank, k_in, e.stream)
                    }),
                )?;
                cx.push(
                    3 * p + 1,
                    Box::new(move |e| {
                        let (b, n) = (&pair.b, pair.n_out);
                        ops::dense_gemv(e.gpu, gemv, f.xa, b, f.delta, n, pair.max_rank, e.stream)
                    }),
                )?;
                cx.push(
                    3 * p + 2,
                    Box::new(move |e| {
                        ops::scaled_add(e.gpu, add, or, f.delta, pair.scale, pair.n_out, e.stream)
                    }),
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod lora_tests;
