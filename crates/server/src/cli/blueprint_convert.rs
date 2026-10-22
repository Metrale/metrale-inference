// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The adapter from the engine's circuit types to the blueprint wire types:
//! `Circuit` → `CircuitSpec`, `FusionPlan` → `PlanSpec`. Indices go `usize` → `u32` through
//! `try_from` (BU64); the mirrored enums convert by exhaustive match, so a new engine variant
//! fails to compile here (TIDE).
//!
//! Owner: server CLI (blueprint).
//! Invariants: nothing here decides anything; it renames.

use anyhow::{Context, Result};
use metrale_blueprint::Sha256;
use metrale_blueprint::types as w;
use metrale_circuit::fuser::EdgeState;
use metrale_circuit::rules::Repeat;
use metrale_circuit::state::{StateAccess, StateFormat};
use metrale_circuit::{Circuit, FusionPlan, KernelId, LayerKind, Mode, Numerics, Section};

pub(crate) fn idx(i: usize) -> Result<u32> {
    u32::try_from(i).context("an index exceeds u32")
}

fn idxs(v: &[usize]) -> Result<Vec<u32>> {
    v.iter().map(|&i| idx(i)).collect()
}

fn layer_kind(k: LayerKind) -> w::LayerKind {
    match k {
        LayerKind::LinearAttention => w::LayerKind::LinearAttention,
        LayerKind::FullAttention => w::LayerKind::FullAttention,
        LayerKind::Mamba => w::LayerKind::Mamba,
        LayerKind::Moe => w::LayerKind::Moe,
        LayerKind::SparseAttention => w::LayerKind::SparseAttention,
        LayerKind::SlidingAttention => w::LayerKind::SlidingAttention,
    }
}

fn section(s: Section) -> w::Section {
    match s {
        Section::Main => w::Section::Main,
        Section::Draft => w::Section::Draft,
    }
}

fn access(a: StateAccess) -> w::StateAccess {
    match a {
        StateAccess::Read => w::StateAccess::Read,
        StateAccess::Write => w::StateAccess::Write,
        StateAccess::Update => w::StateAccess::Update,
        StateAccess::Snapshot => w::StateAccess::Snapshot,
    }
}

pub(crate) fn mode(m: Mode) -> w::Mode {
    match m {
        Mode::Decode => w::Mode::Decode,
        Mode::MultiSeq => w::Mode::MultiSeq,
        Mode::Verify => w::Mode::Verify,
        Mode::Draft => w::Mode::Draft,
        Mode::VerifyBatch => w::Mode::VerifyBatch,
    }
}

fn repeat(r: Repeat) -> w::Repeat {
    match r {
        Repeat::Once => w::Repeat::Once,
        Repeat::PerRow => w::Repeat::PerRow,
        Repeat::Chunk(n) => w::Repeat::Chunk(n),
        Repeat::PerRowButLast => w::Repeat::PerRowButLast,
        Repeat::PerRun => w::Repeat::PerRun,
    }
}

fn numerics(n: &Numerics) -> w::Numerics {
    match n {
        Numerics::BitIdentical { microtest } => w::Numerics::BitIdentical {
            microtest: microtest.clone(),
        },
        Numerics::Reference => w::Numerics::Reference,
        Numerics::Differs { lever } => w::Numerics::Differs {
            lever: lever.clone(),
        },
    }
}

pub(crate) fn kernel_id(k: &KernelId) -> w::KernelId {
    w::KernelId {
        module: k.module.clone(),
        func: k.func.clone(),
    }
}

fn opt_idx(v: Option<usize>) -> Result<Option<u32>> {
    v.map(idx).transpose()
}

/// 2026-10-22: The circuit as wire records.
pub(crate) fn circuit(c: &Circuit) -> Result<w::CircuitSpec> {
    let nodes = c
        .nodes
        .iter()
        .map(|n| {
            Ok(w::NodeSpec {
                id: n.id.clone(),
                local: n.local.clone(),
                op: n.op.name(),
                inputs: idxs(&n.inputs)?,
                outputs: idxs(&n.outputs)?,
                weight: n.weight.as_ref().map(|f| f.name()),
                binding: n.binding.clone(),
                params: n.params.clone(),
                layer: opt_idx(n.layer)?,
                block: n.block.clone(),
                state: n
                    .state
                    .iter()
                    .map(|(s, a)| Ok((idx(*s)?, access(*a))))
                    .collect::<Result<_>>()?,
            })
        })
        .collect::<Result<_>>()?;
    let edges = c
        .edges
        .iter()
        .map(|e| {
            Ok(w::EdgeSpec {
                id: e.id.clone(),
                format: e.format.name(),
                rows: e.rows.text().to_string(),
                dim: e.dim.text().to_string(),
                dim_value: e.dim_value,
                producer: opt_idx(e.producer)?,
                consumers: idxs(&e.consumers)?,
                is_output: e.is_output,
                binds: e.binds.map(|b| b.name().to_string()),
            })
        })
        .collect::<Result<_>>()?;
    let blocks = c
        .blocks
        .iter()
        .map(|b| {
            Ok(w::BlockSpec {
                template: b.template.clone(),
                layer: opt_idx(b.layer)?,
                section: section(b.section),
                first: idx(b.first)?,
                end: idx(b.end)?,
            })
        })
        .collect::<Result<_>>()?;
    let states = c
        .states
        .iter()
        .map(|s| {
            Ok(w::StateSpec {
                id: s.id.clone(),
                local: s.local.clone(),
                block: s.block.clone(),
                layer: opt_idx(s.layer)?,
                section: section(s.section),
                kind: s.kind.name().to_string(),
                format: match &s.format {
                    StateFormat::Fixed(d) => d.name().to_string(),
                    StateFormat::Keyed(k) => format!("{{{k}}}"),
                },
                elements: s.elements,
                lifetime: s.lifetime.name().to_string(),
            })
        })
        .collect::<Result<_>>()?;
    Ok(w::CircuitSpec {
        arch: c.arch.clone(),
        dims: c.dims.clone(),
        layer_kinds: c.layer_kinds.iter().copied().map(layer_kind).collect(),
        nodes,
        edges,
        blocks,
        states,
    })
}

/// 2026-10-22: One fused arm as a wire record.
pub(crate) fn plan(p: &FusionPlan) -> Result<w::PlanSpec> {
    let groups = p
        .groups
        .iter()
        .map(|g| {
            Ok(w::GroupSpec {
                rule: g.rule.clone(),
                kernels: g.kernels.iter().map(kernel_id).collect(),
                repeat: repeat(g.repeat),
                copies: g.copies.map(repeat),
                emitter: g.emitter.clone(),
                numerics: numerics(&g.numerics),
                nodes: idxs(&g.nodes)?,
            })
        })
        .collect::<Result<_>>()?;
    let edge_states = p
        .edge_states
        .iter()
        .map(|s| {
            Ok(match s {
                None => w::EdgeStateSpec::Absent,
                Some(EdgeState::Materialized) => w::EdgeStateSpec::Materialized,
                Some(EdgeState::Fused(g)) => w::EdgeStateSpec::Fused(idx(*g)?),
            })
        })
        .collect::<Result<_>>()?;
    Ok(w::PlanSpec {
        mode: mode(p.mode),
        rows: p.rows,
        groups,
        edge_states,
        edge_formats: p.edge_formats.iter().map(|f| f.name()).collect(),
        digest: Sha256::parse_hex(&p.digest)
            .with_context(|| format!("plan digest `{}` is not sha256 hex", p.digest))?,
    })
}
