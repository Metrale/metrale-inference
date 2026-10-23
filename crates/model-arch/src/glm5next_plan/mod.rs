// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: GLM-5.3's plan emitters: the family's [`GroupEmitter`]s, one per FUSIONS emitter
//! id it serves, registered here and nowhere in the generic executor (FACE), and the compile of
//! one block instance of a rank's plan into the typed steps the layer runs.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`DONE`] lists exactly the plan sites whose groups compile here, per block; every other
//!   site is a declared legacy site of its block ([`CompiledSite::legacy`]), never dropped.
//! - Compiling reads only the plan, the rank circuit and (for a setting a rule's `when` names,
//!   [`compile_dsa`]) the policy the plan was fused under: no lever, no environment.

use anyhow::{Context, Result, ensure};
use metrale_circuit::{Circuit, FusionPlan};
use metrale_model_layers::circuit_exec::family::{EmitterRegistry, GroupEmitter, PlanGroup};

use crate::glm5next_mlp::forward::plan_moe::MoeSteps;

pub mod dsa;
#[cfg(test)]
pub(crate) mod dsa_fixture;
#[cfg(test)]
mod dsa_mtp_tests;
#[cfg(test)]
pub(crate) mod dsa_sabotage;
#[cfg(test)]
pub(crate) mod fixture;
#[cfg(test)]
pub(crate) mod fixture_site;
pub mod install;
pub mod kda;
pub mod mhc;
pub mod moe;
#[cfg(test)]
mod world_tests;

/// 2026-10-23: The compile target the GLM emitters fill: one block instance's steps.
#[derive(Debug, Default)]
pub struct GlmBlock {
    pub moe: moe::MoeSlots,
    pub mhc: mhc::MhcSlots,
    pub kda: kda::KdaSlots,
    pub dsa: dsa::DsaSlots,
}

static MOE_ROUTER: moe::MoeRouter = moe::MoeRouter;
static GLM_MOE_W4A4: moe::GlmMoeW4a4 = moe::GlmMoeW4a4;
static GLM_SWIGLU: moe::GlmSwiglu = moe::GlmSwiglu;
static MOE_BLEND: moe::MoeBlend = moe::MoeBlend { split: false };
static MOE_BLEND_SPLIT: moe::MoeBlend = moe::MoeBlend { split: true };
static GLM_HC: mhc::GlmHc = mhc::GlmHc;
static KDA: kda::Kda = kda::Kda;
static GLM_W4A16: kda::GlmW4a16 = kda::GlmW4a16;
static DSA: dsa::Dsa = dsa::Dsa;

/// 2026-10-23: Every GLM emitter.
static EMITTERS: [&dyn GroupEmitter<GlmBlock>; 9] = [
    &GLM_HC,
    &KDA,
    &GLM_W4A16,
    &DSA,
    &MOE_ROUTER,
    &GLM_MOE_W4A4,
    &GLM_SWIGLU,
    &MOE_BLEND,
    &MOE_BLEND_SPLIT,
];

/// 2026-10-23: The plan-executed sites, `(block, template-local ids of a group)`, as the plan
/// reports name them: the groups the emitters above compile. A group of a listed block whose
/// locals are not here is a legacy site of that block.
pub const DONE: [(&str, &[&str]); 33] = [
    ("dsa", &["mix", "pre", "input_norm"]),
    ("kda", &["mix", "pre", "input_norm"]),
    ("dsa", &["o@all_reduce", "post"]),
    ("kda", &["o@all_reduce", "post"]),
    ("moe", &["mix", "pre", "post_norm"]),
    ("mlp", &["mix", "pre", "post_norm"]),
    ("moe", &["f@all_reduce", "post"]),
    ("mlp", &["f@all_reduce", "post"]),
    ("kda", &["qkv"]),
    ("kda", &["short_conv", "conv_ckpt", "l2"]),
    ("kda", &["fa_quant", "f_b", "gates", "ga_quant", "g_b"]),
    ("kda", &["recur"]),
    ("kda", &["out_norm"]),
    ("kda", &["q", "k", "v", "f_a", "b", "g_a"]),
    ("kda", &["o"]),
    ("kda", &["out_norm", "o"]),
    ("dsa", &["kv_norm", "kv_write"]),
    ("dsa", &["idx_k_norm"]),
    ("dsa", &["pool"]),
    ("dsa", &["select"]),
    ("dsa", &["attend"]),
    ("mtp_dsa", &["kv_norm", "kv_write"]),
    ("mtp_dsa", &["idx_k_norm"]),
    ("mtp_dsa", &["pool"]),
    ("mtp_dsa", &["select"]),
    ("mtp_dsa", &["attend"]),
    ("moe", &["gate"]),
    ("moe", &["top_k"]),
    ("moe", &["gate", "top_k"]),
    ("moe", &["xn_quant", "experts_gate_up"]),
    ("moe", &["experts_clamp", "experts_act"]),
    ("moe", &["eact_quant", "experts_down"]),
    ("moe", &["blend"]),
];

/// 2026-10-23: The plans installed on one GLM text layer, by site: each empty until
/// `--forward circuit` installs it, and an empty site runs the layer's own code.
#[derive(Debug, Default)]
pub struct LayerPlans {
    pub moe: install::InstalledMoe,
    pub mhc: crate::glm5next_mhc_plan::InstalledMhc,
    pub kda: crate::glm5next_kda::plan::InstalledKda,
    pub dsa: crate::glm5next_dsa::plan::InstalledDsa,
}

/// 2026-10-23: The GLM emitter registry.
pub fn registry() -> Result<EmitterRegistry<GlmBlock>> {
    EmitterRegistry::new(&EMITTERS)
}

/// 2026-10-23: One block instance compiled from a plan: its steps, the plan's sites they run
/// (each `block [locals]: kernels`), and the sites of the block the plan names but the layer
/// still runs on its own code, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledSite<S> {
    pub steps: S,
    pub executed: Vec<String>,
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
    let (mut executed, mut legacy) = (Vec::new(), Vec::new());
    let mut last = None;
    for g in PlanGroup::all(circuit, plan).filter(|g| g.layer() == Some(layer) && moe_site_group(g))
    {
        if g.locals().iter().all(|l| l.starts_with("shared_")) {
            legacy.push(g.site());
        } else {
            reg.emit(&g, &mut block)?;
            let kernels: Vec<String> = g.group.kernels.iter().map(|k| k.to_string()).collect();
            executed.push(format!("{}: {}", g.site(), kernels.join(",")));
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
        executed,
        legacy,
    })
}

/// 2026-10-23: Layer `layer`'s mHC sites in `plan` (over the rank circuit `circuit`): the pre-norm
/// and the exchanged post-mix of its mixer and FFN sites, one step for both.
pub fn compile_mhc(
    circuit: &Circuit,
    plan: &FusionPlan,
    layer: usize,
) -> Result<CompiledSite<crate::glm5next_mhc_plan::MhcSteps<metrale_circuit::KernelId>>> {
    let reg = registry()?;
    let mut block = GlmBlock::default();
    let mut executed = Vec::new();
    let mut last = None;
    for g in
        PlanGroup::all(circuit, plan).filter(|g| g.layer() == Some(layer) && mhc::is_site_group(g))
    {
        reg.emit(&g, &mut block)?;
        let kernels: Vec<String> = g.group.kernels.iter().map(|k| k.to_string()).collect();
        executed.push(format!("{}: {}", g.site(), kernels.join(",")));
        last = Some(g);
    }
    let g = last.with_context(|| format!("layer {layer} has no mHC site in this plan"))?;
    Ok(CompiledSite {
        steps: block.mhc.finish(&g)?,
        executed,
        legacy: Vec::new(),
    })
}

/// 2026-10-23: Layer `layer`'s KDA mixer in `plan` (over the rank circuit `circuit`): its own
/// groups through the `kda` emitter, its W4A16 projection groups through `glm_w4a16`; a
/// projection group of another emitter would be a declared legacy site.
pub fn compile_kda(
    circuit: &Circuit,
    plan: &FusionPlan,
    layer: usize,
) -> Result<CompiledSite<crate::glm5next_kda::plan::KdaSteps<metrale_circuit::KernelId>>> {
    let reg = registry()?;
    let mut block = GlmBlock::default();
    let (mut executed, mut legacy) = (Vec::new(), Vec::new());
    let mut last = None;
    let groups = PlanGroup::all(circuit, plan)
        .filter(|g| g.layer() == Some(layer) && g.block() == "kda" && !mhc::is_site_group(g));
    for g in groups {
        if kda::is_site_group(&g) {
            reg.emit(&g, &mut block)?;
            let kernels: Vec<String> = g.group.kernels.iter().map(|k| k.to_string()).collect();
            executed.push(format!("{}: {}", g.site(), kernels.join(",")));
            last = Some(g);
        } else {
            ensure!(
                kda::is_legacy_group(&g),
                "layer {layer}: KDA group {} (rule `{}`, emitter `{}`) is neither the mixer's own \
                 nor one of its projections",
                g.site(),
                g.group.rule,
                g.group.emitter
            );
            legacy.push(g.site());
        }
    }
    let g = last.with_context(|| format!("layer {layer} has no KDA mixer in this plan"))?;
    Ok(CompiledSite {
        steps: block.kda.finish(&g)?,
        executed,
        legacy,
    })
}

/// 2026-10-23: The DSA mixer of `site` (a text layer's or the MTP block's) in `plan` (over the
/// rank circuit `circuit`, fused under `policy`): its own groups through the `dsa` emitter, its
/// projection groups declared legacy sites. The attend's split count is the policy's
/// `glm_dsa_decode_hb` (`dsa::hb_splits`).
pub fn compile_dsa(
    circuit: &Circuit,
    plan: &FusionPlan,
    site: dsa::DsaSite,
    policy: &metrale_circuit::Policy,
) -> Result<CompiledSite<crate::glm5next_dsa::plan::DsaSteps<metrale_circuit::KernelId>>> {
    let reg = registry()?;
    let mut block = GlmBlock::default();
    block.dsa.hb_splits = dsa::hb_splits(policy)?;
    let (mut executed, mut legacy) = (Vec::new(), Vec::new());
    let mut last = None;
    let layer = site.layer();
    for g in PlanGroup::all(circuit, plan).filter(|g| site.holds(g)) {
        if dsa::is_site_group(&g) {
            reg.emit(&g, &mut block)?;
            let kernels: Vec<String> = g.group.kernels.iter().map(|k| k.to_string()).collect();
            executed.push(format!("{}: {}", g.site(), kernels.join(",")));
            last = Some(g);
        } else {
            ensure!(
                dsa::is_legacy_group(&g),
                "layer {layer}: DSA group {} (rule `{}`, emitter `{}`) is neither the mixer's own \
                 nor one of its projections",
                g.site(),
                g.group.rule,
                g.group.emitter
            );
            legacy.push(g.site());
        }
    }
    let g = last.with_context(|| format!("layer {layer} has no DSA mixer in this plan"))?;
    Ok(CompiledSite {
        steps: block.dsa.finish(&g)?,
        executed,
        legacy,
    })
}

#[cfg(test)]
mod tests;
