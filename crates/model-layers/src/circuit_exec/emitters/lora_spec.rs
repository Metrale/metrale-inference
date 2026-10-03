// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The overlay spec of a LoRA pool from what its adapters target, pure: the one
//! derivation both the serve (from the installed slots) and `met circuit memory` (from the
//! adapters' tensor names) use.
//!
//! Owner: model-layers (FEATURES workstream).
//! Invariants:
//! - Attention q/k/v/o are the union over the pool's adapters: legacy folds them per row through
//!   the routing tables, which exist for every (layer, projection) any adapter targets.
//! - The dense FFN's gate|up and down and the GatedDeltaNet out_proj are the ACTIVE adapter's
//!   only: legacy folds the installed pair (`ops::lora_delta::apply_lora_delta`).
//! - What the circuit cannot run is refused, as the binding refuses it (`layers/circuit_lora.rs`):
//!   router and routed-expert targets, and a gate|up fold with one of its two pairs.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use metrale_circuit::LinearRole;
use metrale_circuit::lora::LoraSpec;
use metrale_config::ModelConfig;

use crate::lora::{AdapterAb, AdapterSlot, LoraModule, LoraTarget, classify_key};

/// 2026-10-03: One adapter's targets: the (layer, projection) pairs it holds.
pub type AdapterTargets = BTreeSet<(usize, LoraModule)>;

/// 2026-10-03: The targets of an adapter from its tensor names (`classify_key`); a router or
/// routed-expert target is refused (the MoE LoRA port is not done).
pub fn targets_from_keys<'a>(
    keys: impl IntoIterator<Item = &'a str>,
    cfg: &ModelConfig,
) -> Result<AdapterTargets> {
    let mut out = AdapterTargets::new();
    for key in keys {
        let (layer, target, ab) = classify_key(key, cfg)?;
        match target {
            LoraTarget::Attn(m) => {
                if ab == AdapterAb::A {
                    out.insert((layer, m));
                }
            }
            LoraTarget::Router | LoraTarget::Expert { .. } => {
                bail!("'{key}': a MoE LoRA target (the circuit's MoE LoRA port is not done)")
            }
        }
    }
    Ok(out)
}

/// 2026-10-03: The targets of an installed pool slot.
pub fn targets_of_slot(slot: &AdapterSlot) -> AdapterTargets {
    let mut out = AdapterTargets::new();
    for (layer, lw) in slot.layers.iter().enumerate() {
        let Some(lw) = lw else { continue };
        for (m, pair) in [
            (LoraModule::QProj, lw.q_proj.is_some()),
            (LoraModule::KProj, lw.k_proj.is_some()),
            (LoraModule::VProj, lw.v_proj.is_some()),
            (LoraModule::OProj, lw.o_proj.is_some()),
            (LoraModule::GateProj, lw.gate_proj.is_some()),
            (LoraModule::UpProj, lw.up_proj.is_some()),
            (LoraModule::DownProj, lw.down_proj.is_some()),
            (LoraModule::OutProj, lw.out_proj.is_some()),
        ] {
            if pair {
                out.insert((layer, m));
            }
        }
    }
    out
}

/// 2026-10-03: The spec of a pool of padded `rank` holding `adapters`, of which `active` is the
/// active one.
pub fn spec_from_targets(
    rank: u64,
    adapters: &[AdapterTargets],
    active: usize,
) -> Result<LoraSpec> {
    let Some(act) = adapters.get(active) else {
        bail!(
            "the active adapter {active} is not in the pool of {}",
            adapters.len()
        );
    };
    let mut layers: BTreeMap<usize, BTreeSet<LinearRole>> = BTreeMap::new();
    for &(layer, m) in adapters.iter().flatten() {
        let role = match m {
            LoraModule::QProj => LinearRole::Q,
            LoraModule::KProj => LinearRole::K,
            LoraModule::VProj => LinearRole::V,
            LoraModule::OProj => LinearRole::O,
            _ => continue,
        };
        layers.entry(layer).or_default().insert(role);
    }
    let gate_up: BTreeSet<usize> = act
        .iter()
        .filter(|(_, m)| matches!(m, LoraModule::GateProj | LoraModule::UpProj))
        .map(|&(l, _)| l)
        .collect();
    for layer in gate_up {
        let both = act.contains(&(layer, LoraModule::GateProj))
            && act.contains(&(layer, LoraModule::UpProj));
        if !both {
            bail!(
                "layer {layer}: an FFN adapter on one of gate and up (the gate|up fold takes both)"
            );
        }
        layers.entry(layer).or_default().insert(LinearRole::GateUp);
    }
    for &(layer, m) in act {
        let role = match m {
            LoraModule::DownProj => LinearRole::Down,
            LoraModule::OutProj => LinearRole::GdnOut,
            _ => continue,
        };
        layers.entry(layer).or_default().insert(role);
    }
    Ok(LoraSpec { rank, layers })
}

#[cfg(test)]
#[path = "lora_spec_tests.rs"]
mod lora_spec_tests;
