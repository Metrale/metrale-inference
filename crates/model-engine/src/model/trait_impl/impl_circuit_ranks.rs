// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `--forward circuit` under tensor parallelism: the circuit executor does not run a
//! TP rank (`circuit_unmodelled`), so each layer instead executes from this rank's fused plans
//! the sites its family's emitters compile (`CircuitBindings::install_plans`) and runs every
//! other site on its own code. The executed sites are reported at installation and disclosed.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - The plans are fused over this rank's circuit of the instance, under the one policy every
//!   layer derives from its live levers (`CircuitBindings::plan_policy`); layers that disagree,
//!   an instance of another world size, or a model with no plan-executed site refuse the
//!   selection rather than run legacy under the name `circuit`.
//! - Only `circuit` (every rule) is accepted: `circuit-reference` is refused under TP.
//! - A drafter with a circuit block (the GLM MTP block) installs its sites from the rank's draft
//!   plan (draft n=1), fused under the same policy, which its block must derive too.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::Mode;
use metrale_model_layers::circuit_exec::family::RankPlans;
use metrale_model_layers::circuit_exec::{Fusions, TargetModules, kernels, load_served};

use super::super::types::TransformerModel;
use crate::traits::ForwardDisclosure;

impl TransformerModel {
    /// 2026-10-23: Whether `--forward circuit` takes the per-layer path: a TP rank.
    pub(super) fn circuit_by_rank(&self) -> bool {
        self.config.tp_world_size > 1
    }

    /// 2026-10-23: Fuse this rank's plans of `instance` and install them on every layer;
    /// returns the executed sites.
    pub(super) fn install_rank_plans(
        &self,
        instance: &metrale_circuit::Instance,
        fusions: Fusions,
        modules: &TargetModules,
        config_json: &str,
    ) -> Result<Vec<String>> {
        ensure!(
            fusions == Fusions::All,
            "--forward circuit-reference is not supported on a tensor-parallel rank; use \
             --forward circuit"
        );
        let (rank, world) = (self.config.tp_rank, self.config.tp_world_size);
        let loaded = load_served(instance, &self.config, config_json)?;
        ensure!(
            loaded.ranks.len() == world,
            "instance `{}` plans {} rank(s); this serve runs tensor parallelism over {world}",
            instance.recipe,
            loaded.ranks.len()
        );
        let circuit = &loaded.ranks[rank];
        let mut policies = self.layers.iter().map(|l| l.plan_policy(&instance.policy));
        let policy = policies.next().context("the model has no layers")??;
        for (i, p) in policies.enumerate() {
            ensure!(
                p? == policy,
                "layer {} derives another plan policy than layer 0",
                i + 1
            );
        }
        let drafter = self.proposer.as_deref().and_then(|p| p.circuit_block());
        if let Some(b) = drafter {
            ensure!(
                b.plan_policy(&instance.policy)? == policy,
                "the draft block derives another plan policy than layer 0"
            );
        }
        let available = kernels::available_in(&loaded.rules, &modules.0)?;
        let shapes = std::iter::once((Mode::Decode, 1))
            .chain(
                self.proposer
                    .is_some()
                    .then_some([2, 3, 4].map(|r| (Mode::Verify, r)))
                    .into_iter()
                    .flatten(),
            )
            .chain(drafter.map(|_| (Mode::Draft, 1)));
        let mut plans = Vec::new();
        for (mode, rows) in shapes {
            let p = metrale_circuit::fuse(circuit, &loaded.rules, &available, &policy, mode, rows)
                .with_context(|| format!("fusing rank {rank} {mode:?} at {rows} rows"))?;
            plans.push(p);
        }
        let rp = RankPlans {
            circuit,
            plans: &plans,
            policy: &policy,
        };
        let mut sites = Vec::new();
        for l in &self.layers {
            sites.extend(l.install_plans(self.gpu.as_ref(), Some(rp))?);
        }
        if let Some(b) = drafter {
            sites.extend(b.install_plans(self.gpu.as_ref(), Some(rp))?);
        }
        if sites.is_empty() {
            bail!(
                "--forward circuit on TP rank {rank}/{world}: no layer of this model executes a \
                 site from the plan"
            );
        }
        Ok(sites)
    }

    /// 2026-10-23: Return every layer to its own code.
    pub(super) fn clear_rank_plans(&self) -> Result<()> {
        for l in &self.layers {
            l.install_plans(self.gpu.as_ref(), None)?;
        }
        if let Some(b) = self.proposer.as_deref().and_then(|p| p.circuit_block()) {
            b.install_plans(self.gpu.as_ref(), None)?;
        }
        Ok(())
    }

    /// 2026-10-23: The disclosure of installed rank plans, `None` when no layer has any: the
    /// forward `circuit-sites`, the digest over the executed sites (each names its layer, rows,
    /// group and kernels), no whole-step launch count (the other sites are legacy).
    pub(super) fn rank_plans_disclosure(&self) -> Option<ForwardDisclosure> {
        let drafter = self.proposer.as_deref().and_then(|p| p.circuit_block());
        let sites: Vec<String> = self
            .layers
            .iter()
            .flat_map(|l| l.plan_sites())
            .chain(drafter.into_iter().flat_map(|b| b.plan_sites()))
            .collect();
        (!sites.is_empty()).then(|| ForwardDisclosure {
            forward: "circuit-sites",
            plan_digest: Some(metrale_circuit::digest::plans_digest(
                sites.iter().map(String::as_str),
            )),
            launches_per_step: None,
        })
    }
}

/// 2026-10-23: The startup report of installed sites: their count, the site kinds (a site with
/// its layer stripped) and how many layers run each.
pub(super) fn report_sites(rank: usize, sites: &[String]) -> String {
    let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
    for s in sites {
        let kind = s
            .split_once(' ')
            .and_then(|(_, r)| r.split_once(' '))
            .map_or(s.as_str(), |(_, k)| k);
        *kinds.entry(kind).or_default() += 1;
    }
    let lines: Vec<String> = kinds.iter().map(|(k, n)| format!("  {n} x {k}")).collect();
    format!(
        "--forward circuit (TP rank {rank}): {} sites execute from the plan; every other site \
         runs the layer's own code:\n{}",
        sites.len(),
        lines.join("\n")
    )
}
