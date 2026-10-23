// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan execution installed on one GLM text layer: the routed MoE site's plans by
//! row count, compiled from the rank's fused plans at the gated shapes ([`GATED`]) and bound to
//! the layer's own buffers, and the live policy the engine fuses them under.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A plan is installed only at a shape the CPU launch gate proves equal to the hand-written
//!   site (`plan_moe_tests`): decode n=1, verify n=2. `forward_moe` does not branch on the mode,
//!   so a site at those row counts in any mode (multi-sequence, a short prefill piece) runs the
//!   same launches either way.
//! - Under a lever the MoE rules do not model, installation is refused naming it, rather than
//!   running a plan that launches other kernels than the serve's levers pick.
//! - Resolved at installation, read-only per forward (ROTI): the hot path takes one read lock.

use anyhow::{Result, bail, ensure};
use metrale_circuit::{Mode, Policy};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::circuit_exec::family::RankPlans;
use parking_lot::RwLock;

use crate::glm5next_mlp::ExpertShard;
use crate::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::forward::plan_moe::{self, MoePlan};
use crate::glm5next_mlp::weights::Glm5NextMoeWeights;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-23: The shapes the plan-vs-legacy launch gate covers.
pub const GATED: [(Mode, u64); 2] = [(Mode::Decode, 1), (Mode::Verify, 2)];

/// 2026-10-23: The shape the MTP block's launch gate covers (`dsa_mtp_tests`): one draft row.
pub const DRAFT_GATED: [(Mode, u64); 1] = [(Mode::Draft, 1)];

/// 2026-10-23: The policy settings the MoE, mHC, KDA and DSA rules read that come from this
/// layer's levers and layout, as `(key, value)`. `glm_dsa_decode_hb` is the head-batched split
/// count, `off` for the per-head kernel.
pub fn plan_settings(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
) -> [(&'static str, String); 12] {
    let on = |b: bool| if b { "on" } else { "off" }.to_string();
    let l = &k.levers;
    let layout = match cfg.expert_shard {
        ExpertShard::Whole => "ep",
        ExpertShard::Sliced(_) => "tp",
    };
    [
        ("glm_router_rows", on(l.router_rows)),
        ("glm_router_topk_rank", on(l.router_topk_rank)),
        ("glm_moe_combine_split", on(l.moe_combine_split)),
        ("glm_moe_w4a4_grouped", on(l.mlp.w4a4_grouped)),
        ("glm_moe_expert_layout", layout.to_string()),
        ("glm_mhc_fused", on(l.mhc_fused)),
        ("glm_ar_fused_post", on(l.ar_fused_post)),
        ("glm_kda_row_reg", on(l.kda.row_reg)),
        ("glm_kda_verify_tokens", on(l.kda.verify_tokens)),
        (
            "glm_dsa_decode_hb",
            l.dsa.decode_hb.map_or("off".to_string(), |n| n.to_string()),
        ),
        ("glm_dsa_verify_rows", on(l.dsa_verify_rows)),
        ("glm_dsa_indexer_rows", on(l.dsa_indexer_rows)),
    ]
}

/// 2026-10-23: `base` with [`plan_settings`] set live. Errors on a key `base` does not declare:
/// the instance states every setting a rule reads (PCND).
pub fn live_policy(
    base: &Policy,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
) -> Result<Policy> {
    let mut p = base.clone();
    for (key, value) in plan_settings(k, cfg) {
        let Some(v) = p.settings.get_mut(key) else {
            bail!("GLM plan: the instance's policy does not declare `{key}`");
        };
        *v = value;
    }
    Ok(p)
}

/// 2026-10-23: The MoE levers the rules do not model, at a value other than the one the rules
/// name (the `_sl` gate/up sweep, the zeroing SwiGLU); the vec8 combine takes 1024 rows or more
/// only, so it does not reach a gated site: installation refuses them.
pub fn unmodelled(k: &Glm5NextMlpKernels) -> Vec<&'static str> {
    let m = &k.levers.mlp;
    [
        (
            !m.sweep_serial,
            "METRALE_GLM_MOE_SWEEP_SERIAL=0 (the rules name the `_sl` sweeps)",
        ),
        (
            !m.zero_in_swiglu,
            "METRALE_GLM_MOE_ZERO_IN_SWIGLU=0 (the rules zero expert_out in the SwiGLU)",
        ),
    ]
    .into_iter()
    .filter_map(|(set, what)| set.then_some(what))
    .collect()
}

/// 2026-10-23: The routed MoE plans one text layer runs, by row count.
#[derive(Default)]
pub struct InstalledMoe {
    by_rows: RwLock<Vec<(MoePlan, Vec<String>)>>,
}

impl std::fmt::Debug for InstalledMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstalledMoe")
            .field("sites", &self.sites().len())
            .finish()
    }
}

impl InstalledMoe {
    /// 2026-10-23: Compile and bind layer `layer`'s routed MoE site from each gated plan of
    /// `plans` (`None` clears); returns the executed sites, each `layer L n=R: <site>`.
    #[allow(clippy::too_many_arguments)]
    pub fn install(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        plans: Option<RankPlans<'_>>,
        k: &Glm5NextMlpKernels,
        cfg: &Glm5NextMlpConfig,
        w: &Glm5NextMoeWeights,
        ws: &Glm5NextMlpWorkspace,
    ) -> Result<Vec<String>> {
        let Some(rp) = plans else {
            self.by_rows.write().clear();
            return Ok(Vec::new());
        };
        let refused = unmodelled(k);
        ensure!(
            refused.is_empty(),
            "GLM layer {layer}: the plan's MoE rules do not model {}",
            refused.join("; ")
        );
        let mut next: Vec<(MoePlan, Vec<String>)> = Vec::new();
        for plan in rp.plans {
            if !GATED.contains(&(plan.mode, plan.rows)) {
                continue;
            }
            let site = super::compile_moe(rp.circuit, plan, layer)?;
            let rows = site.steps.rows;
            ensure!(
                rows <= k.levers.mlp.row_batch_max,
                "GLM layer {layer}: METRALE_GLM_MOE_ROW_BATCH_MAX {} splits {rows} rows the plan \
                 runs as one site",
                k.levers.mlp.row_batch_max
            );
            ensure!(
                next.iter().all(|(p, _)| p.0.rows != rows),
                "GLM layer {layer}: two gated plans at {rows} rows"
            );
            let bound = plan_moe::bind(gpu, site.steps, cfg, w, ws)?;
            let sites = site
                .executed
                .iter()
                .map(|s| format!("layer {layer} n={rows}: {s}"))
                .collect();
            next.push((bound, sites));
        }
        let sites = next.iter().flat_map(|(_, s)| s.clone()).collect();
        *self.by_rows.write() = next;
        Ok(sites)
    }

    /// 2026-10-23: The installed sites.
    pub fn sites(&self) -> Vec<String> {
        self.by_rows
            .read()
            .iter()
            .flat_map(|(_, s)| s.clone())
            .collect()
    }

    /// 2026-10-23: Run the site at `rows` rows from its plan; `Ok(false)` when no plan is
    /// installed at `rows` (the caller runs its own code).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &dyn GpuBackend,
        k: &Glm5NextMlpKernels,
        cfg: &Glm5NextMlpConfig,
        w: &Glm5NextMoeWeights,
        ws: &Glm5NextMlpWorkspace,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let by_rows = self.by_rows.read();
        let Some((plan, _)) = by_rows.iter().find(|(p, _)| p.0.rows == rows) else {
            return Ok(false);
        };
        plan_moe::run_with_legacy_shared(plan, gpu, k, cfg, w, ws, x, out, rows, stream)?;
        Ok(true)
    }
}

#[cfg(test)]
#[path = "install_tests.rs"]
mod tests;
