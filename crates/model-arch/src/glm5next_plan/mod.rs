// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: GLM-5.3's plan emitters: the family's [`GroupEmitter`]s, one per FUSIONS emitter
//! id it serves, registered here and nowhere in the generic executor (FACE), and the compile of
//! one block instance of a rank's plan into the typed steps the layer runs.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`DONE`] lists exactly the plan sites whose groups compile here, per block; every other
//!   site is a declared legacy site of its block ([`CompiledSite::legacy`]), never dropped.
//! - Compiling reads only the plan and the rank circuit: no lever, no environment.

use anyhow::{Context, Result, ensure};
use metrale_circuit::{Circuit, FusionPlan};
use metrale_model_layers::circuit_exec::family::{EmitterRegistry, GroupEmitter, PlanGroup};

use crate::glm5next_mlp::forward::plan_moe::MoeSteps;

#[cfg(test)]
pub(crate) mod fixture;
pub mod moe;

/// 2026-10-23: The compile target the GLM emitters fill: one block instance's steps.
#[derive(Debug, Default)]
pub struct GlmBlock {
    pub moe: moe::MoeSlots,
}

static MOE_ROUTER: moe::MoeRouter = moe::MoeRouter;
static GLM_MOE_W4A4: moe::GlmMoeW4a4 = moe::GlmMoeW4a4;
static GLM_SWIGLU: moe::GlmSwiglu = moe::GlmSwiglu;
static MOE_BLEND: moe::MoeBlend = moe::MoeBlend { split: false };
static MOE_BLEND_SPLIT: moe::MoeBlend = moe::MoeBlend { split: true };

/// 2026-10-23: Every GLM emitter.
static EMITTERS: [&dyn GroupEmitter<GlmBlock>; 5] = [
    &MOE_ROUTER,
    &GLM_MOE_W4A4,
    &GLM_SWIGLU,
    &MOE_BLEND,
    &MOE_BLEND_SPLIT,
];

/// 2026-10-23: The plan-executed sites, `(block, template-local ids of a group)`, as the plan
/// reports name them: the groups the emitters above compile. A group of a listed block whose
/// locals are not here is a legacy site of that block.
pub const DONE: [(&str, &[&str]); 7] = [
    ("moe", &["gate"]),
    ("moe", &["top_k"]),
    ("moe", &["gate", "top_k"]),
    ("moe", &["xn_quant", "experts_gate_up"]),
    ("moe", &["experts_clamp", "experts_act"]),
    ("moe", &["eact_quant", "experts_down"]),
    ("moe", &["blend"]),
];

/// 2026-10-23: The GLM emitter registry.
pub fn registry() -> Result<EmitterRegistry<GlmBlock>> {
    EmitterRegistry::new(&EMITTERS)
}

/// 2026-10-23: One block instance compiled from a plan: its steps and the sites of the block the
/// plan names but the layer still runs on its own code, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledSite<S> {
    pub steps: S,
    pub legacy: Vec<String>,
}

/// 2026-10-23: The template-local ids of a routed MoE site's own nodes (`moe` block of
/// kernels/circuits/blocks/glm5_next_ffn.toml, with the quantizers the precision inserts); the
/// block's mHC pre-mix, post-mix and derived collectives are other sites.
const MOE_SITE: [&str; 13] = [
    "gate",
    "top_k",
    "xn_quant",
    "experts_gate_up",
    "experts_clamp",
    "experts_act",
    "eact_quant",
    "experts_down",
    "shared_gate_up",
    "shared_clamp",
    "shared_act",
    "shared_down",
    "blend",
];

/// 2026-10-23: Whether a group is one of the routed MoE site's.
fn moe_site_group(g: &PlanGroup<'_>) -> bool {
    g.block() == "moe" && g.locals().iter().all(|l| MOE_SITE.contains(l))
}

/// 2026-10-23: The routed MoE site of layer `layer` in `plan` (over the rank circuit `circuit`):
/// its groups through the GLM emitters, the shared expert's group a declared legacy site.
pub fn compile_moe(
    circuit: &Circuit,
    plan: &FusionPlan,
    layer: usize,
) -> Result<CompiledSite<MoeSteps<metrale_circuit::KernelId>>> {
    let reg = registry()?;
    let mut block = GlmBlock::default();
    let mut legacy = Vec::new();
    let mut last = None;
    for g in PlanGroup::all(circuit, plan).filter(|g| g.layer() == Some(layer) && moe_site_group(g))
    {
        if g.locals().iter().all(|l| l.starts_with("shared_")) {
            legacy.push(g.site());
        } else {
            reg.emit(&g, &mut block)?;
        }
        last = Some(g);
    }
    let g = last.with_context(|| format!("layer {layer} has no routed MoE site in this plan"))?;
    ensure!(
        legacy.len() <= 1,
        "layer {layer}: the MoE site has {} shared-expert groups",
        legacy.len()
    );
    Ok(CompiledSite {
        steps: block.moe.finish(&g)?,
        legacy,
    })
}

#[cfg(test)]
mod tests;
