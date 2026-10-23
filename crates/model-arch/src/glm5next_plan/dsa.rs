// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The emitter of a DSA mixer's own groups (`dsa`): the latent write
//! (`[kv_norm,kv_write]`), the indexer key norm (`[idx_k_norm]`: per row, or once over a verify's
//! rows), the indexer row's placement (`[pool]`: store, pool key, geometry), the selection over
//! the persisted pool keys (`[select]`) and the attend (`[attend]`). The projections, their norm
//! and activation quantizers and the FP32-out selector GEMVs are declared legacy sites of the
//! block: the layer's code.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Recognizes the entries `glm5next_dsa::plan` launches (`glm5next_dsa::plan::entry`) by the
//!   plan's `(module, func)`; refuses any other, never guessing a launch form.
//! - The head-batched attend's split count is the `glm_dsa_decode_hb` the plan was fused under
//!   ([`DsaSlots::hb_splits`], set by `compile_dsa` from the policy), the setting its rule's
//!   `when` names.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::rules::Repeat;
use metrale_circuit::{KernelId, Mode};
use metrale_model_layers::circuit_exec::family::{GroupEmitter, PlanGroup};

use super::GlmBlock;
use crate::glm5next_dsa::plan::entry::{self, Entry};
use crate::glm5next_dsa::plan::{
    AttendStep, DsaSteps, KNormStep, PoolKeys, ScoresStep, SelectStep,
};

/// 2026-10-23: The DSA block's own groups this emitter compiles.
const LATENT: [&str; 2] = ["kv_norm", "kv_write"];
const K_NORM: [&str; 1] = ["idx_k_norm"];
const POOL: [&str; 1] = ["pool"];
const SELECT: [&str; 1] = ["select"];
const ATTEND: [&str; 1] = ["attend"];

/// 2026-10-23: The block's projections, their norm and quantizers, and the selector GEMVs: the
/// sites the layer runs.
const LEGACY: [&str; 12] = [
    "xn_quant",
    "q_a",
    "kv_a",
    "q_a_norm",
    "qan_quant",
    "q_b",
    "idx_k",
    "idx_gate",
    "idx_w",
    "idx_q",
    "a_quant",
    "o",
];

/// 2026-10-23: The policy setting that carries the head-batched attend's split count.
pub const HB_SETTING: &str = "glm_dsa_decode_hb";

/// 2026-10-23: A DSA mixer's steps as its groups fill them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DsaSlots {
    /// 2026-10-23: `glm_dsa_decode_hb` of the plan's policy: `None` when `off`.
    pub hb_splits: Option<usize>,
    pub latent_write: Option<KernelId>,
    pub k_norm: Option<KNormStep<KernelId>>,
    /// 2026-10-23: `(store, pool key, geometry)`.
    pub pool: Option<(KernelId, KernelId, KernelId)>,
    pub select: Option<SelectStep<KernelId>>,
    pub attend: Option<AttendStep<KernelId>>,
}

/// 2026-10-23: Whether a group is one of the DSA block's own groups this emitter compiles.
pub fn is_site_group(g: &PlanGroup<'_>) -> bool {
    g.block() == "dsa" && g.group.emitter == "dsa"
}

/// 2026-10-23: Whether a group is one of the block's projection sites the layer runs.
pub fn is_legacy_group(g: &PlanGroup<'_>) -> bool {
    g.block() == "dsa" && g.locals().iter().all(|l| LEGACY.contains(l))
}

/// 2026-10-23: `glm_dsa_decode_hb` of `policy` as a split count, `None` for `off`.
pub fn hb_splits(policy: &metrale_circuit::Policy) -> Result<Option<usize>> {
    let v = policy
        .settings
        .get(HB_SETTING)
        .with_context(|| format!("the plan's policy does not declare `{HB_SETTING}`"))?;
    match v.as_str() {
        "off" => Ok(None),
        n => Ok(Some(n.parse().with_context(|| {
            format!("`{HB_SETTING} = {n}` is neither `off` nor a split count")
        })?)),
    }
}

fn is(k: &KernelId, (module, func): Entry) -> bool {
    k.module == module && k.func == func
}

fn want(k: &KernelId, e: Entry) -> Result<KernelId> {
    ensure!(
        is(k, e),
        "the plan names `{k}`; this emitter launches {}::{}",
        e.0,
        e.1
    );
    Ok(k.clone())
}

fn set<T>(slot: &mut Option<T>, v: T, what: &str) -> Result<()> {
    ensure!(slot.is_none(), "the DSA site has two {what} groups");
    *slot = Some(v);
    Ok(())
}

/// 2026-10-23: `dsa` over a DSA mixer's own groups.
pub struct Dsa;

impl GroupEmitter<GlmBlock> for Dsa {
    fn id(&self) -> &'static str {
        "dsa"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        ensure!(
            matches!(g.mode, Mode::Decode | Mode::Verify),
            "`dsa` runs one sequence's decode or verify here; the plan is {:?}",
            g.mode
        );
        ensure!(
            g.block() == "dsa",
            "`dsa` serves the `dsa` block here; group {} is {}",
            g.index,
            g.site()
        );
        let s = &mut site.dsa;
        let locals = g.locals();
        let ks = &g.group.kernels;
        if locals == LATENT {
            g.expect_kernels(self.id(), 1)?;
            set(
                &mut s.latent_write,
                want(&ks[0], entry::LATENT_WRITE)?,
                "latent-write",
            )
        } else if locals == K_NORM {
            g.expect_kernels(self.id(), 1)?;
            let k = want(&ks[0], entry::K_NORM)?;
            let step = match (&g.group.repeat, g.rows) {
                (Repeat::PerRow, 1) => KNormStep::Row(k),
                (Repeat::Once, r) if r > 1 => KNormStep::Rows(k),
                (rep, r) => bail!("a {r}-row indexer key norm repeated {rep:?}"),
            };
            set(&mut s.k_norm, step, "key-norm")
        } else if locals == POOL {
            g.expect_kernels(self.id(), 3)?;
            let pool = (
                want(&ks[0], entry::STORE)?,
                want(&ks[1], entry::POOL_KEYS)?,
                want(&ks[2], entry::WRITE_GEOM)?,
            );
            set(&mut s.pool, pool, "indexer-placement")
        } else if locals == SELECT {
            set(&mut s.select, select_step(g)?, "selection")
        } else if locals == ATTEND {
            let step = match ks.as_slice() {
                [k] => AttendStep::PerHead(want(k, entry::PER_HEAD)?),
                [d, m] => AttendStep::HeadBatched {
                    decode: want(d, entry::DECODE_HB)?,
                    merge: want(m, entry::HB_MERGE)?,
                    splits: s.hb_splits.filter(|&n| n >= 2).with_context(|| {
                        format!(
                            "the attend names the head-batched merge; `{HB_SETTING}` is {:?}",
                            s.hb_splits
                        )
                    })?,
                },
                _ => bail!("the attend names {} kernels", ks.len()),
            };
            set(&mut s.attend, step, "attend")
        } else {
            bail!(
                "`dsa` takes a DSA latent-write, key-norm, placement, selection or attend group; \
                 got {locals:?}"
            )
        }
    }
}

/// 2026-10-23: The selection group's step: the scores (lane-per-head; the window walk for a
/// verify's rows), `select::TOPK_MC_PASSES` top-k passes, the collect and the expansion, once
/// per row at one row or once over a verify's rows.
fn select_step(g: &PlanGroup<'_>) -> Result<SelectStep<KernelId>> {
    let passes = crate::glm5next_dsa::select::TOPK_MC_PASSES as usize;
    let ks = &g.group.kernels;
    ensure!(
        ks.len() == passes + 3,
        "the selection names {} kernels; it launches the scores, {passes} top-k passes, the \
         collect and the expansion",
        ks.len()
    );
    ensure!(
        g.rows == 1 || matches!(g.group.repeat, Repeat::Once),
        "a {}-row selection repeated {:?}: this executor selects a verify's rows in one pass",
        g.rows,
        g.group.repeat
    );
    let scores = if is(&ks[0], entry::SCORES_HEADS_PAGED) {
        ScoresStep::HeadsPaged(ks[0].clone())
    } else {
        ScoresStep::HeadsWindow(want(&ks[0], entry::SCORES_HEADS_WINDOW)?)
    };
    let topk_pass = want(&ks[1], entry::TOPK_MC_PASS)?;
    for k in &ks[2..=passes] {
        ensure!(
            *k == topk_pass,
            "the top-k passes name `{k}` and `{topk_pass}`"
        );
    }
    Ok(SelectStep {
        pool_keys: PoolKeys::F32,
        scores,
        topk_pass,
        topk_collect: want(&ks[passes + 1], entry::TOPK_MC_COLLECT)?,
        expand: want(&ks[passes + 2], entry::EXPAND_ROWS)?,
    })
}

impl DsaSlots {
    /// 2026-10-23: The mixer's steps; an error naming a missing group.
    pub fn finish(self, g: &PlanGroup<'_>) -> Result<DsaSteps<KernelId>> {
        let need = |what: &str| format!("the layer's DSA plan has no {what} group");
        let (store, pool_keys, write_geom) = self.pool.with_context(|| need("placement"))?;
        Ok(DsaSteps {
            rows: usize::try_from(g.rows)?,
            latent_write: self.latent_write.with_context(|| need("latent-write"))?,
            k_norm: self.k_norm.with_context(|| need("key-norm"))?,
            store,
            pool_keys,
            write_geom,
            select: self.select.with_context(|| need("selection"))?,
            attend: self.attend.with_context(|| need("attend"))?,
        })
    }
}

/// 2026-10-23: The DSA levers and diagnostics the rules do not model, at a value other than the
/// one the rules name: installation refuses them. `glm_dsa_decode_hb`, `glm_dsa_verify_rows` and
/// `glm_dsa_indexer_rows` are policy settings (`install::plan_settings`).
pub fn unmodelled(layer: &crate::glm5next_dsa::layer::Glm5NextDsaLayer) -> Vec<&'static str> {
    let (l, s) = (&layer.levers, &layer.levers.dsa.select);
    [
        (
            !l.dsa_verify_select_once,
            "METRALE_GLM_DSA_VERIFY_SELECT_ONCE=0 (the rules select a verify's rows in one pass)",
        ),
        (
            !l.dsa_proj_in_merged,
            "METRALE_GLM_DSA_PROJ_IN_MERGED=0 (the rules name the merged split4 projections)",
        ),
        (
            !l.dsa_selector_pair,
            "METRALE_GLM_DSA_SELECTOR_PAIR=0 (the rules name the FP32-out pair)",
        ),
        (
            !s.scores_heads,
            "METRALE_GLM_DSA_SCORES_HEADS=0 (the rules name the lane-per-head scores)",
        ),
        (
            !s.scores_heads_window,
            "METRALE_GLM_DSA_SCORES_HEADS_WINDOW=0 (the rules name the window walk at 2..=8 rows)",
        ),
        (
            !s.topk_mc || !s.decode_topk_radix,
            "METRALE_GLM_DSA_TOPK_MC=0 or _DECODE_TOPK_RADIX=0 (the rules name the multi-CTA \
             radix top-k)",
        ),
        (
            l.dsa.serve_ref || layer.dsa_owned.hb_check.is_some(),
            "METRALE_GLM_DSA_DECODE_HB_SERVE_REF or _CHECK (head-batched diagnostics)",
        ),
        (
            layer.indexer_format != crate::glm5next_dsa::fp8_rows::IndexerRowFormat::Bf16Pooled,
            "an indexer cache without persisted BF16 pool keys (METRALE_GLM_DSA_POOL_KEYS=0, a \
             flat cache or --indexer-cache-dtype fp8; the rules score persisted keys)",
        ),
        (
            layer.select_kernels.rows.pool_keys_bf16,
            "METRALE_GLM_DSA_POOL_KEYS_BF16 (and _DECODE_SCORES_MMA, which needs it): the rules \
             model FP32 persisted pool keys",
        ),
    ]
    .into_iter()
    .filter_map(|(set, what)| set.then_some(what))
    .collect()
}

/// 2026-10-23: Compile and bind layer `layer`'s DSA mixer `dsa` from each gated plan of `plans`
/// into `into` (`None` clears); returns the executed sites, each `layer L n=R: <site>`.
pub fn install(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    layer: usize,
    plans: Option<metrale_model_layers::circuit_exec::family::RankPlans<'_>>,
    dsa: &crate::glm5next_dsa::layer::Glm5NextDsaLayer,
    into: &crate::glm5next_dsa::plan::InstalledDsa,
) -> Result<Vec<String>> {
    use crate::glm5next_dsa::plan::DsaPlan;
    use crate::glm5next_dsa::select::{contiguous_pool_count, topk_mc_ctas};
    let Some(rp) = plans else {
        into.set(Vec::new());
        return Ok(Vec::new());
    };
    let refused = unmodelled(dsa);
    ensure!(
        refused.is_empty(),
        "GLM layer {layer}: the plan's DSA rules do not model {}",
        refused.join("; ")
    );
    let max_pools = contiguous_pool_count(
        dsa.cfg.index_kpool,
        crate::glm5next_dsa::state::max_dsa_context(&dsa.cfg),
    );
    let sm_count = dsa.select_kernels.rows.sm_count;
    ensure!(
        sm_count > 0,
        "GLM layer {layer}: the DSA plan's scores run one block per SM; the SM count did not resolve"
    );
    let mut next = Vec::new();
    for plan in rp.plans {
        if !super::install::GATED.contains(&(plan.mode, plan.rows)) {
            continue;
        }
        let site = super::compile_dsa(rp.circuit, plan, layer, rp.policy)?;
        let rows = site.steps.rows;
        let steps = site
            .steps
            .map(|k| metrale_model_layers::circuit_exec::family::plan_kernel(gpu, &k))?;
        let plan_hb = match steps.attend {
            AttendStep::HeadBatched { splits, .. } => Some(splits),
            AttendStep::PerHead(_) => None,
        };
        ensure!(
            plan_hb == dsa.decode_kernel.hb_splits(),
            "GLM layer {layer}: the plan's attend runs {plan_hb:?} head-batched splits, the \
             layer's decode kernel {:?}",
            dsa.decode_kernel.hb_splits()
        );
        let bound = DsaPlan {
            steps,
            max_pools,
            topk_ctas: topk_mc_ctas(max_pools),
            sm_count,
        };
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
#[path = "dsa_tests.rs"]
mod tests;
