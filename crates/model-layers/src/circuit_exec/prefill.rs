// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The prefill programs (M6, LIFECYCLE-DESIGN.md 15.4): one program per (prefill
//! mode, row bucket, GDN arm), compiled at boot. The bucket ladder comes from the rule set
//! (`metrale_circuit::buckets`), so a program fused at its bucket's top is exact for every row
//! count in the bucket; the pass's row count reaches the emitters at run time
//! ([`super::program::PrefillStep`]).
//!
//! A prefill plan's edges live in the legacy forward's buffer arena, where the legacy layers
//! keep them ([`placement`]): the arena is sized for the widest pass already, and a second copy
//! of it in the circuit workspace would cost the KV pool its size. Prefill programs run eagerly,
//! as legacy prefill does.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every row count `1..=max_tokens` of each prefill mode selects exactly one program per GDN
//!   arm.
//! - Every edge a prefill plan materialises is placed in the arena or refused at build; no
//!   prefill program reads the workspace.

use anyhow::{Context, Result, bail};
use metrale_circuit::buckets::{Bucket, bucket_ladder, bucket_of};
use metrale_circuit::{AvailableKernels, Circuit, FusionPlan, Mode, Policy, fuse};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::compile::{self, Inputs, Placement};
use super::program::Program;

/// 2026-10-03: The policy setting that picks the GatedDeltaNet prefill arm: `on` after a
/// prefix-cache restore (the exact-replay arm, legacy `gdn_exact_replay`), `off` otherwise (the
/// chunked FLA arm).
pub const EXACT_REPLAY: &str = "gdn_exact_replay";

/// 2026-10-03: One compiled prefill program.
pub struct PrefillProgram {
    pub mode: Mode,
    pub bucket: Bucket,
    /// 2026-10-03: Compiled under `gdn_exact_replay = on`.
    pub exact_replay: bool,
    pub program: Program,
    pub plan: FusionPlan,
}

/// 2026-10-03: Every prefill program of a build.
pub struct PrefillPrograms {
    pub programs: Vec<PrefillProgram>,
    /// 2026-10-03: The widest pass compiled (the arena's rows).
    pub max_tokens: u64,
}

/// 2026-10-03: What a prefill build reads besides the compile inputs.
pub struct PrefillBoot<'a> {
    pub circuit: &'a Circuit,
    pub rules: &'a [metrale_circuit::Rule],
    pub runtime: &'a [metrale_circuit::RuntimeRoute],
    pub available: &'a AvailableKernels,
    pub policy: &'a Policy,
    pub arena: &'a BufferArena,
    /// 2026-10-03: The widest pass (`BufferArena::max_batch_tokens`).
    pub max_tokens: u64,
}

impl PrefillPrograms {
    /// 2026-10-03: Compile every (mode, bucket, arm).
    pub fn build(b: &PrefillBoot<'_>, inputs: &Inputs<'_>) -> Result<Self> {
        let mut programs = Vec::new();
        for mode in Mode::PREFILL {
            for bucket in bucket_ladder(b.rules, b.runtime, mode, b.max_tokens) {
                for exact_replay in [false, true] {
                    let mut policy = b.policy.clone();
                    policy.settings.insert(
                        EXACT_REPLAY.to_string(),
                        if exact_replay { "on" } else { "off" }.to_string(),
                    );
                    let plan = fuse(b.circuit, b.rules, b.available, &policy, mode, bucket.hi)
                        .with_context(|| {
                            format!(
                                "fusing {} at rows {}..={}",
                                mode.name(),
                                bucket.lo,
                                bucket.hi
                            )
                        })?;
                    let placement = placement(b.circuit, &plan, b.arena, inputs.fixed)?;
                    let program = compile::compile_placed(b.circuit, &plan, &placement, inputs)
                        .with_context(|| {
                            format!(
                                "compiling {} at rows {}..={}",
                                mode.name(),
                                bucket.lo,
                                bucket.hi
                            )
                        })?;
                    programs.push(PrefillProgram {
                        mode,
                        bucket,
                        exact_replay,
                        program,
                        plan,
                    });
                }
            }
        }
        heads_agree(&programs)?;
        Ok(Self {
            programs,
            max_tokens: b.max_tokens,
        })
    }

    /// 2026-10-03: The program of a `tokens`-row pass in `mode` on the GDN arm `exact_replay`.
    pub fn select(&self, mode: Mode, tokens: u64, exact_replay: bool) -> Result<&PrefillProgram> {
        let ladder: Vec<Bucket> = self
            .programs
            .iter()
            .filter(|p| p.mode == mode && p.exact_replay == exact_replay)
            .map(|p| p.bucket)
            .collect();
        let bucket = bucket_of(&ladder, tokens).with_context(|| {
            format!(
                "no {} program holds {tokens} rows (compiled up to {})",
                mode.name(),
                self.max_tokens
            )
        })?;
        self.programs
            .iter()
            .find(|p| p.mode == mode && p.exact_replay == exact_replay && p.bucket == bucket)
            .context("a bucket of the ladder has no program")
    }

    /// 2026-10-03: The program whose head a `tokens`-row last pass runs. The head reads only the
    /// pass's last row, and its rules are the same in both prefill modes and both GDN arms, so
    /// the offset-0, non-replay program of the bucket is the one (`heads_agree` checks it at
    /// build).
    pub fn head(&self, tokens: u64) -> Result<&PrefillProgram> {
        self.select(Mode::Prefill, tokens, false)
    }
}

/// 2026-10-03: Refuse a build whose prefill programs' head launches differ between programs of
/// one bucket: a driver runs the head of [`PrefillPrograms::head`] after any pass of the bucket.
fn heads_agree(programs: &[PrefillProgram]) -> Result<()> {
    let head = |p: &PrefillProgram| -> Vec<String> {
        p.program
            .segments
            .iter()
            .filter(|s| matches!(s.of, super::program::SegmentOf::Head(_)))
            .flat_map(|s| {
                p.program.launches[s.launches.clone()]
                    .iter()
                    .map(|l| l.kernel.clone())
            })
            .collect()
    };
    for a in programs {
        for b in programs.iter().filter(|b| b.bucket == a.bucket) {
            anyhow::ensure!(
                head(a) == head(b),
                "the prefill heads of {} and {} at rows {}..={} differ",
                a.mode.name(),
                b.mode.name(),
                a.bucket.lo,
                a.bucket.hi
            );
        }
    }
    Ok(())
}

/// 2026-10-03: The arena buffer of a prefill edge, by its block and template-local id: the
/// buffer the legacy layer keeps it in. The list grows with the prefill rules; an edge a plan
/// materialises that is not listed refuses the build.
fn arena_buffer(block: &str, local: &str, arena: &BufferArena) -> Option<DevicePtr> {
    Some(match (block, local) {
        ("gdn" | "attn", "xn") => arena.norm_output(),
        // 2026-10-03: GatedDeltaNet edges (emitters/prefill_gdn.rs): the qkvz projection's
        // `[Q | K | V | Z]` rows (`trait_prefill_block.rs:88`), the gated norm's output over the
        // conv rows (`:318`), and out_proj's (`:365`).
        ("gdn", "qkvz") => arena.ssm_deinterleaved(),
        ("gdn", "gated") => arena.ssm_qkvz(),
        ("gdn", "o") => arena.moe_output(),
        // 2026-10-03: Attention edges (emitters/prefill_attn.rs). The O projection writes
        // `norm_output` (`qwen3_attention/prefill/paged_oproj.rs:31`), and the post-attention
        // norm writes the FFN's normed input there (`trait_impl/prefill_inner.rs:302-314`).
        ("attn", "o") => arena.norm_output(),
        // 2026-10-03: Dense FFN, embedding and head edges (emitters/prefill_ffn.rs).
        // The FFN reads the post-mixer norm from `norm_output`; gate|up lands in
        // `expert_gate_out` (its up half in `expert_up_out`, `dense_ffn_prefill.rs:141-142`); down
        // writes `moe_output`; the head's final norm writes `norm_output`.
        ("dense_ffn", "xn") | ("head", "xn") => arena.norm_output(),
        ("dense_ffn", "gu") => arena.expert_gate_out(),
        ("dense_ffn", "d") => arena.moe_output(),
        _ => return None,
    })
}

/// 2026-10-03: The placement of `plan`'s edges: the stream edges and declared outputs as in
/// every plan (`compile::external_buffer`), every other materialised edge in its arena buffer.
pub fn placement(
    circuit: &Circuit,
    plan: &FusionPlan,
    arena: &BufferArena,
    fixed: &super::fixed::Fixed,
) -> Result<Placement> {
    let layout = compile::layout(circuit, plan)?;
    let mut ptrs = vec![None; circuit.edges.len()];
    for (e, edge) in circuit.edges.iter().enumerate() {
        if plan.edge_states[e] != Some(metrale_circuit::EdgeState::Materialized) {
            continue;
        }
        if layout.external.contains(&e) {
            ptrs[e] = Some(compile::external_buffer(circuit, e, fixed, plan.mode)?);
            continue;
        }
        let local = edge.id.rsplit('.').next().unwrap_or(&edge.id);
        let block = edge_block(circuit, e).unwrap_or("");
        match arena_buffer(block, local, arena) {
            Some(p) => ptrs[e] = Some(p),
            None => bail!(
                "prefill edge `{}` (block `{block}`) has no arena buffer; a prefill plan's edges \
                 live where the legacy layer keeps them",
                edge.id
            ),
        }
    }
    Ok(Placement {
        strides: vec![None; ptrs.len()],
        ptrs,
    })
}

/// 2026-10-03: The block of the node that produces edge `e`.
fn edge_block(circuit: &Circuit, e: usize) -> Option<&str> {
    circuit
        .nodes
        .iter()
        .find(|n| n.outputs.contains(&e))
        .map(|n| n.block.as_str())
}
