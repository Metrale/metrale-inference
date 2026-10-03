// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `--profile` under the circuit (LIFECYCLE-DESIGN.md 15.10): the decode program run
//! eagerly with a timing event at every group boundary, each group's time attributed to the
//! buckets legacy's profile reports (`decode_profiled`, `me/model/impl_b1_decode.rs`): per layer
//! the GatedDeltaNet `qkvz` and `ba_gates` projections and the FFN block (`SSM-MoE` /
//! `Attn-MoE`), per layer kind the layer totals, and the head.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants:
//! - A profiled run issues exactly the program's launches, in its order: it times the plan the
//!   unprofiled step runs, never another arm.
//! - Group `g`'s time is the span between the events recorded before its first launch and
//!   after its last; a group without launches spans nothing.
//! - Attribution is pure ([`attribute`]); every group lands in one bucket, and a group fused
//!   across a layer boundary counts in the layer of its first node.
//! - Never captured: the run synchronizes at the end (and wherever the caller's layer hook
//!   does), so the caller runs it with graphs off, as legacy's profile does.

use anyhow::{Context, Result, ensure};
use metrale_circuit::{Circuit, FusionPlan, LayerKind, Section};
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::program::{Program, StepEnv};

/// 2026-10-03: Where a group's time is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    /// 2026-10-03: The prologue (the embedding), which legacy's profile does not time.
    Prologue,
    /// 2026-10-03: Part of layer `layer`'s step.
    Layer { layer: usize, part: Part },
    /// 2026-10-03: The final norm, lm_head and what follows them.
    Head,
}

/// 2026-10-03: The parts of a layer legacy's profile names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// 2026-10-03: The GatedDeltaNet `qkvz` projection (`SSM qkvz`).
    Qkvz,
    /// 2026-10-03: The GatedDeltaNet `b|a` projection and gates (`SSM ba_gates`).
    BaGates,
    /// 2026-10-03: The FFN block from the mixer's residual add and post-norm through the down
    /// projection (`SSM-MoE` / `Attn-MoE`): every FFN-block node but the closing residual add.
    Ffn,
    /// 2026-10-03: Anything else in the layer (the mixer, the closing residual add).
    Other,
}

/// 2026-10-03: The bucket of every group of `plan`, a plan of the target section.
pub fn attribute(circuit: &Circuit, plan: &FusionPlan) -> Result<Vec<Bucket>> {
    let block_of = |n: usize| {
        circuit
            .blocks
            .iter()
            .position(|b| (b.first..b.end).contains(&n))
            .with_context(|| format!("node {n} is in no block"))
    };
    plan.groups
        .iter()
        .enumerate()
        .map(|(g, group)| {
            let first = *group
                .nodes
                .first()
                .with_context(|| format!("group {g} has no node"))?;
            let b = &circuit.blocks[block_of(first)?];
            ensure!(
                b.section == Section::Main,
                "group {g} is outside the target's forward"
            );
            let Some(layer) = b.layer else {
                let before_layers = circuit
                    .blocks
                    .iter()
                    .take_while(|x| x.first <= first)
                    .all(|x| x.layer.is_none());
                return Ok(if before_layers {
                    Bucket::Prologue
                } else {
                    Bucket::Head
                });
            };
            Ok(Bucket::Layer {
                layer,
                part: part_of(circuit, &group.nodes, layer, &block_of)?,
            })
        })
        .collect()
}

/// 2026-10-03: A layer's first block is its mixer, the rest its FFN.
fn part_of(
    circuit: &Circuit,
    nodes: &[usize],
    layer: usize,
    block_of: &dyn Fn(usize) -> Result<usize>,
) -> Result<Part> {
    let mixer = circuit
        .blocks
        .iter()
        .position(|b| b.layer == Some(layer))
        .with_context(|| format!("layer {layer} has no block"))?;
    let mut ffn = false;
    let mut mixer_locals = Vec::new();
    for &n in nodes {
        let b = block_of(n)?;
        let node = &circuit.nodes[n];
        if circuit.blocks[b].layer == Some(layer) && b != mixer {
            let closing = circuit.blocks[b]
                .stream_out
                .is_some_and(|e| node.outputs.contains(&e));
            ffn |= !closing;
        } else if b == mixer {
            mixer_locals.push(node.local.as_str());
        }
    }
    let gdn = circuit.layer_kinds.get(layer) == Some(&LayerKind::LinearAttention);
    Ok(if ffn {
        Part::Ffn
    } else if gdn && mixer_locals.contains(&"qkvz") {
        Part::Qkvz
    } else if gdn && mixer_locals.iter().any(|l| *l == "ba" || *l == "gates") {
        Part::BaGates
    } else {
        Part::Other
    })
}

/// 2026-10-03: One profiled step: each group's milliseconds, with its bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct StepTimes {
    /// 2026-10-03: Per group, in plan order.
    pub group_ms: Vec<f32>,
    /// 2026-10-03: Per group, its bucket.
    pub buckets: Vec<Bucket>,
}

impl StepTimes {
    /// 2026-10-03: Milliseconds of the groups whose bucket satisfies `pick`.
    pub fn sum(&self, pick: impl Fn(&Bucket) -> bool) -> f64 {
        self.group_ms
            .iter()
            .zip(&self.buckets)
            .filter(|(_, b)| pick(b))
            .map(|(&ms, _)| f64::from(ms))
            .sum()
    }

    /// 2026-10-03: Milliseconds of `part` in `layer`.
    pub fn part_ms(&self, layer: usize, part: Part) -> f64 {
        self.sum(|b| *b == Bucket::Layer { layer, part })
    }

    /// 2026-10-03: Milliseconds of every part of `layer`.
    pub fn layer_ms(&self, layer: usize) -> f64 {
        self.sum(|b| matches!(b, Bucket::Layer { layer: l, .. } if *l == layer))
    }
}

/// 2026-10-03: Legacy's per-step line (`decode_profiled`): times in ms, the counts are layers.
pub fn step_line(tok: usize, attn: (f64, usize), ssm: (f64, usize), head_ms: f64) -> String {
    format!(
        "PROFILE tok={tok}: total={:.1}ms attn={:.1}ms({}) ssm={:.1}ms({}) head={head_ms:.1}ms",
        attn.0 + ssm.0 + head_ms,
        attn.0,
        attn.1,
        ssm.0,
        ssm.1,
    )
}

/// 2026-10-03: Legacy's GatedDeltaNet op line (`ssm_forward.rs`'s `prof!`), in whole µs.
pub fn ssm_op_line(label: &str, us: u128) -> String {
    format!("    SSM {label}: {us}μs")
}

/// 2026-10-03: Legacy's FFN-block line (`trait_decode.rs`, `decode_inner.rs`): `kind` is `SSM`
/// for a GatedDeltaNet layer, `Attn` for an attention layer; µs shown as ms.
pub fn ffn_line(kind: &str, us: u128) -> String {
    format!("  {kind}-MoE: {:.1}ms", us as f64 / 1000.0)
}

/// 2026-10-03: The report of one profiled step in legacy's shape: per layer, its lines in the
/// order legacy logs them, then the step line. Times are truncated to whole µs, as legacy's
/// `as_micros` does.
pub fn report(times: &StepTimes, kinds: &[LayerKind], tok: usize) -> (Vec<Vec<String>>, String) {
    let us = |ms: f64| (ms * 1000.0) as u128;
    let per_layer = kinds
        .iter()
        .enumerate()
        .map(|(l, k)| match k {
            LayerKind::LinearAttention => vec![
                ssm_op_line("qkvz", us(times.part_ms(l, Part::Qkvz))),
                ssm_op_line("ba_gates", us(times.part_ms(l, Part::BaGates))),
                ffn_line("SSM", us(times.part_ms(l, Part::Ffn))),
            ],
            _ => vec![ffn_line("Attn", us(times.part_ms(l, Part::Ffn)))],
        })
        .collect();
    let attn_layers: Vec<usize> = (0..kinds.len())
        .filter(|&l| kinds[l] == LayerKind::FullAttention)
        .collect();
    let attn = attn_layers.iter().map(|&l| times.layer_ms(l)).sum::<f64>();
    let all = times.sum(|b| matches!(b, Bucket::Layer { .. }));
    let line = step_line(
        tok,
        (attn, attn_layers.len()),
        (all - attn, kinds.len() - attn_layers.len()),
        times.sum(|b| *b == Bucket::Head),
    );
    (per_layer, line)
}

/// 2026-10-03: The decode program's timing events, one per group boundary, made at build when
/// the serve profiles.
pub struct DecodeProfile {
    events: Vec<u64>,
    buckets: Vec<Bucket>,
    kinds: Vec<LayerKind>,
}

impl DecodeProfile {
    /// 2026-10-03: The events for `program`, compiled from `plan` over `circuit`; refuses a
    /// program whose launches are not grouped in plan order.
    pub fn new(
        gpu: &dyn GpuBackend,
        circuit: &Circuit,
        plan: &FusionPlan,
        program: &Program,
    ) -> Result<Self> {
        ensure!(
            program.plan_digest == plan.digest,
            "the profiled program was not compiled from this plan"
        );
        ensure!(
            program
                .launches
                .windows(2)
                .all(|w| w[0].group <= w[1].group)
                && program
                    .launches
                    .last()
                    .is_none_or(|l| l.group < plan.groups.len()),
            "the program's launches are not in plan group order"
        );
        let buckets = attribute(circuit, plan)?;
        let mut events = Vec::with_capacity(plan.groups.len() + 1);
        for _ in 0..=plan.groups.len() {
            match gpu.create_timing_event() {
                Ok(e) => events.push(e),
                Err(e) => {
                    for &ev in &events {
                        gpu.destroy_event(ev).ok();
                    }
                    return Err(e.context("creating the profile's timing events"));
                }
            }
        }
        Ok(Self {
            events,
            buckets,
            kinds: circuit.layer_kinds.clone(),
        })
    }

    /// 2026-10-03: The buckets, per group.
    pub fn buckets(&self) -> &[Bucket] {
        &self.buckets
    }

    /// 2026-10-03: [`report`] of `times`, a run of this profile, at token `tok`.
    pub fn report(&self, times: &StepTimes, tok: usize) -> (Vec<Vec<String>>, String) {
        report(times, &self.kinds, tok)
    }

    /// 2026-10-03: Run `program` on `env.stream` with an event at every group boundary, call
    /// `after_layer(l)` once the last group attributed to layer `l` is queued, then wait for the
    /// step and read every group's time.
    pub fn run(
        &self,
        program: &Program,
        env: &StepEnv<'_>,
        after_layer: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<StepTimes> {
        let gpu = env.gpu;
        let groups = self.buckets.len();
        let mut launches = program.launches.iter().peekable();
        for g in 0..groups {
            gpu.record_event(self.events[g], env.stream)?;
            while let Some(l) = launches.next_if(|l| l.group == g) {
                (l.run)(env).with_context(|| format!("circuit group {g} ({})", l.kernel))?;
            }
            if let Bucket::Layer { layer, .. } = self.buckets[g] {
                let next = self.buckets.get(g + 1);
                if !matches!(next, Some(Bucket::Layer { layer: n, .. }) if *n == layer) {
                    after_layer(layer)?;
                }
            }
        }
        ensure!(
            launches.next().is_none(),
            "launches outside the plan's groups"
        );
        gpu.record_event(self.events[groups], env.stream)?;
        gpu.event_synchronize(self.events[groups])?;
        let group_ms = (0..groups)
            .map(|g| gpu.event_elapsed_ms(self.events[g], self.events[g + 1]))
            .collect::<Result<Vec<_>>>()?;
        Ok(StepTimes {
            group_ms,
            buckets: self.buckets.clone(),
        })
    }

    /// 2026-10-03: Destroy the events.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for e in self.events {
            gpu.destroy_event(e)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod profile_tests;
