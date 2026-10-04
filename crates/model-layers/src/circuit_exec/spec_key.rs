// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The plan digests that key a measured speculative-cost table
//! (`--spec-cost-model measured`): per measured mode (`verify`, `verify_batch`, `draft`), one
//! digest over every plan the recipe's circuit instance declares for that mode, each
//! `verify`/`draft` plan followed by its runtime-route arms, fused offline as `met circuit show` fuses them. The golden plans
//! specify the kernel routing of the legacy forward too, so the key describes the kernels
//! either forward runs; it needs no executor. `met benchmark spec-cost-table` writes it, and the
//! serve recomputes it at boot and refuses a table whose key differs.
//!
//! Owner: model-layers (circuit_exec).
//! Invariants:
//! - A recipe without a circuit instance has no key: refused, never approximated.
//! - A measured mode for which the instance declares no plan is refused: its cost would be
//!   keyed by nothing that changes with its kernels.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use metrale_circuit::digest::plans_digest;
use metrale_circuit::fuser::{fuse, fuse_table};
use metrale_circuit::runtime::route_arms;
use metrale_circuit::{AvailableKernels, Instance, Loaded, Mode};

use super::sources::{instance, sources};

/// 2026-10-04: The modes a speculative step runs, in key order.
pub const KEY_MODES: [Mode; 3] = [Mode::Verify, Mode::VerifyBatch, Mode::Draft];

/// 2026-10-04: Mode name → digest over that mode's declared plans, for `recipe`.
pub fn spec_cost_plan_digests(recipe: &str) -> Result<BTreeMap<String, String>> {
    let inst = instance(recipe).context("the speculative cost model needs a circuit instance")?;
    let loaded = metrale_circuit::load(&inst, sources(&inst)?)?;
    let mut key = BTreeMap::new();
    for mode in KEY_MODES {
        let digests = mode_plan_digests(&inst, &loaded, mode)?;
        if digests.is_empty() {
            bail!(
                "recipe `{recipe}`: its circuit instance declares no {} plan, so the speculative \
                 cost of that mode would be unkeyed (kernels/circuits/INSTANCES.toml)",
                mode.name()
            );
        }
        key.insert(
            mode.name().to_string(),
            plans_digest(digests.iter().map(String::as_str)),
        );
    }
    Ok(key)
}

/// 2026-10-04: The digests of every plan `inst` declares for `mode`, in declared order, each
/// non-batched plan followed by its runtime-route arms: the `digest:` lines of the golden files,
/// in file order.
fn mode_plan_digests(inst: &Instance, loaded: &Loaded, mode: Mode) -> Result<Vec<String>> {
    // 2026-10-04: Offline, as `met circuit show`: every kernel a rule names counts as built.
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    let (circuit, rules, policy) = (&loaded.circuit, loaded.rules.as_slice(), &inst.policy);
    let mut out = Vec::new();
    if mode == Mode::VerifyBatch {
        // 2026-10-04: A batched-verify plan has no runtime-route arms (`render_table_plan`).
        for table in &inst.verify_batch {
            out.push(fuse_table(circuit, rules, &avail, policy, table)?.digest);
        }
        return Ok(out);
    }
    for &rows in inst.plans.get(&mode).into_iter().flatten() {
        let plan = fuse(circuit, rules, &avail, policy, mode, rows)?;
        let arms = route_arms(circuit, (rules, &loaded.runtime), &avail, policy, &plan)?;
        out.push(plan.digest);
        out.extend(arms.into_iter().map(|(_, arm)| arm.digest));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "spec_key_tests.rs"]
mod tests;
