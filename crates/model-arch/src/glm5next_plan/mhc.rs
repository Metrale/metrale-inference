// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The emitter of a layer's mHC sites (`glm_hc`): the fused pre-mix + norm of the
//! mixer site (`[mix,pre,input_norm]`) and of the FFN site (`[mix,pre,post_norm]`), and each
//! site's all-reduce lowered to the one-shot exchange with the post-mix summing the partials
//! (`[<out>@all_reduce,post]`, `glm_ar_fused_post`). Both sites of a layer run one step each.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Recognizes the entries it was written for (`glm5next_hc_pre_norm_c1`, the exchange +
//!   `glm5next_hc_post_rank_sum`) by the plan's `(module, func)` and comm launches; refuses any
//!   other, never guessing a grid.
//! - The highway's expand (layer 0) and contract (the last layer) stay the layer's own code.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::rules::CommApi;
use metrale_circuit::{KernelId, Mode};
use metrale_model_layers::circuit_exec::family::{GroupEmitter, PlanGroup};

use super::GlmBlock;
use crate::glm5next_mhc::GLM5NEXT_HC_POST_SUM_MODULE;
use crate::glm5next_mhc_c1 as c1;
use crate::glm5next_mhc_plan::{MhcSteps, PostStep, PreStep, c1_grid_y};

/// 2026-10-23: The blocks whose mHC site is the mixer's; the FFN site's are `moe` and `mlp`.
const MIXER_BLOCKS: [&str; 2] = ["kda", "dsa"];
const FFN_BLOCKS: [&str; 2] = ["moe", "mlp"];

/// 2026-10-23: A layer's mHC steps as its groups fill them: `[mixer, ffn]` of each.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MhcSlots {
    pub pre: [Option<PreStep<KernelId>>; 2],
    pub post: [Option<PostStep<KernelId>>; 2],
}

/// 2026-10-23: Whether a group is one of a layer's mHC site groups this emitter compiles.
pub fn is_site_group(g: &PlanGroup<'_>) -> bool {
    match g.locals().as_slice() {
        ["mix", "pre", "input_norm"] | ["mix", "pre", "post_norm"] => true,
        [out, "post"] => out.ends_with("@all_reduce"),
        _ => false,
    }
}

/// 2026-10-23: `0` for the mixer site's block, `1` for the FFN site's.
fn site_index(g: &PlanGroup<'_>) -> Result<usize> {
    let b = g.block();
    if MIXER_BLOCKS.contains(&b) {
        Ok(0)
    } else if FFN_BLOCKS.contains(&b) {
        Ok(1)
    } else {
        bail!(
            "`glm_hc` serves the text layers' mixer and FFN blocks; group {} is {}",
            g.index,
            g.site()
        )
    }
}

fn set<T>(slot: &mut Option<T>, v: T, what: &str) -> Result<()> {
    ensure!(slot.is_none(), "the mHC site has two {what} groups");
    *slot = Some(v);
    Ok(())
}

/// 2026-10-23: `glm_hc` over a layer's mHC sites.
pub struct GlmHc;

impl GroupEmitter<GlmBlock> for GlmHc {
    fn id(&self) -> &'static str {
        "glm_hc"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        ensure!(
            matches!(g.mode, Mode::Decode | Mode::Verify | Mode::MultiSeq),
            "`glm_hc` serves decode, verify and multi-sequence decode; the plan is {:?}",
            g.mode
        );
        let i = site_index(g)?;
        g.expect_kernels(self.id(), 1)?;
        let k = g.kernel(0)?;
        let s = &mut site.mhc;
        match g.locals().as_slice() {
            ["mix", "pre", norm] => {
                let want = if i == 0 { "input_norm" } else { "post_norm" };
                ensure!(
                    *norm == want,
                    "the {} block's pre-norm reads `{norm}`",
                    g.block()
                );
                ensure!(
                    k.module == c1::HC_PRE_C1_MODULE && k.func == c1::HC_PRE_C1_KERNEL,
                    "the plan names `{k}`; this emitter launches {}::{}",
                    c1::HC_PRE_C1_MODULE,
                    c1::HC_PRE_C1_KERNEL
                );
                ensure!(
                    (1..=u64::from(c1::HC_PRE_C1_MAX_TOKENS)).contains(&g.rows),
                    "`{k}` serves 1..={} tokens; the plan has {}",
                    c1::HC_PRE_C1_MAX_TOKENS,
                    g.rows
                );
                let hc = usize::try_from(g.dim("hc")?)?;
                let step = PreStep {
                    kernel: k.clone(),
                    grid_y: c1_grid_y(hc)?,
                };
                set(&mut s.pre[i], step, "pre-norm")
            }
            [_, "post"] => {
                let comm = &g.group.comm;
                ensure!(
                    comm.len() == 1 && comm[0].api == CommApi::Exchange && comm[0].at == 0,
                    "the post-mix group lowers its all-reduce to one leading exchange; the plan \
                     has {comm:?}"
                );
                ensure!(
                    k.module == GLM5NEXT_HC_POST_SUM_MODULE
                        && k.func == "glm5next_hc_post_rank_sum",
                    "the plan names `{k}`; this emitter launches \
                     {GLM5NEXT_HC_POST_SUM_MODULE}::glm5next_hc_post_rank_sum"
                );
                set(&mut s.post[i], PostStep { kernel: k.clone() }, "post-mix")
            }
            other => bail!("`glm_hc` takes an mHC pre-norm or post-mix group; got {other:?}"),
        }
    }
}

impl MhcSlots {
    /// 2026-10-23: The layer's one mHC step; an error naming a missing site group, or two sites
    /// whose groups differ.
    pub fn finish(self, g: &PlanGroup<'_>) -> Result<MhcSteps<KernelId>> {
        let need = |what: &str, i: usize| {
            format!(
                "the layer's plan has no {} {what} group",
                ["mixer", "FFN"][i]
            )
        };
        let [pa, pf] = self.pre;
        let [qa, qf] = self.post;
        let (pa, pf) = (
            pa.with_context(|| need("pre-norm", 0))?,
            pf.with_context(|| need("pre-norm", 1))?,
        );
        let (qa, qf) = (
            qa.with_context(|| need("post-mix", 0))?,
            qf.with_context(|| need("post-mix", 1))?,
        );
        ensure!(
            pa == pf,
            "the mixer and FFN pre-norms differ: {pa:?} / {pf:?}"
        );
        ensure!(
            qa == qf,
            "the mixer and FFN post-mixes differ: {qa:?} / {qf:?}"
        );
        Ok(MhcSteps {
            rows: usize::try_from(g.rows)?,
            hidden: usize::try_from(g.dim("hidden")?)?,
            hc_mult: usize::try_from(g.dim("hc")?)?,
            pre: pa,
            post: qa,
        })
    }
}

/// 2026-10-23: The mHC levers the rules do not model, at a value other than the one the rules
/// name: installation refuses them.
pub fn unmodelled(levers: &crate::glm5next_layer::GlmLevers) -> Vec<&'static str> {
    let m = &levers.mhc;
    [
        (
            !m.pre_c1,
            "METRALE_GLM_MHC_PRE_C1=0 (the rules name glm5next_hc_pre_norm_c1)",
        ),
        (
            m.pre_splitk,
            "METRALE_GLM_MHC_PRE_SPLITK (it precedes c1; the rules name c1)",
        ),
        (
            levers.mhc_post_fold,
            "METRALE_GLM_MHC_POST_FOLD (the rules run the mixer site's post-mix alone)",
        ),
        (
            m.post_sum_ctas.is_some(),
            "METRALE_GLM_MHC_POST_SUM_CTAS (the plan runs the uncapped post-mix grid)",
        ),
    ]
    .into_iter()
    .filter_map(|(set, what)| set.then_some(what))
    .collect()
}

/// 2026-10-23: Compile and bind layer `layer`'s mHC sites from each gated plan of `plans` into
/// `into` (`None` clears); returns the executed sites, each `layer L n=R: <site>`. `levers`,
/// `mhc` and `hidden` are the layer's.
pub fn install(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    layer: usize,
    plans: Option<metrale_model_layers::circuit_exec::family::RankPlans<'_>>,
    levers: &crate::glm5next_layer::GlmLevers,
    mhc: &crate::glm5next_layer::Glm5NextMhc,
    hidden: usize,
    into: &crate::glm5next_mhc_plan::InstalledMhc,
) -> Result<Vec<String>> {
    let Some(rp) = plans else {
        into.set(Vec::new());
        return Ok(Vec::new());
    };
    let refused = unmodelled(levers);
    ensure!(
        refused.is_empty(),
        "GLM layer {layer}: the plan's mHC rules do not model {}",
        refused.join("; ")
    );
    for (site, w) in [("mixer", &mhc.attn), ("FFN", &mhc.ffn)] {
        ensure!(
            crate::glm5next_mhc_levers::fused_runs(&mhc.kernels, w, hidden, mhc.hc_mult),
            "GLM layer {layer}: the {site} site cannot run the fused pre-norm the plan names"
        );
    }
    let mut next = Vec::new();
    for plan in rp.plans {
        if !super::install::GATED.contains(&(plan.mode, plan.rows)) {
            continue;
        }
        let site = super::compile_mhc(rp.circuit, plan, layer)?;
        let s = site.steps;
        let rows = s.rows;
        ensure!(
            (s.hidden, s.hc_mult) == (hidden, mhc.hc_mult),
            "GLM layer {layer}: the plan's mHC dims ({}, {}) are not the layer's ({hidden}, {})",
            s.hidden,
            s.hc_mult,
            mhc.hc_mult
        );
        let bound = s.map(|k| metrale_model_layers::circuit_exec::family::plan_kernel(gpu, &k))?;
        let sites = site
            .executed
            .iter()
            .map(|e| format!("layer {layer} n={rows}: {e}"))
            .collect();
        next.push((bound, sites));
    }
    let sites = next
        .iter()
        .flat_map(|(_, s): &(_, Vec<String>)| s.clone())
        .collect();
    into.set(next);
    Ok(sites)
}

#[cfg(test)]
#[path = "mhc_tests.rs"]
mod tests;
