// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Check a target instance against the checkpoint it names: the layer kinds its
//! `config.json` lists, and the precision every bound module declares (ModelOpt
//! `hf_quant_config.json` or `config.json` `quantization_config`), resolved through the engine's
//! own policy at the `declared` tier with every W8A8 kernel capability present, so the answer
//! is the declared weight and activation formats.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: pure (the caller reads the files); every disagreement is listed in one error.

use metrale_config::{DeclaredPrecisionPlan, WeightQuantPolicy};

use super::VennError;
use super::repo::config_layer_kinds;
use crate::Loaded;
use crate::instances::Instance;
use crate::precision::EdgePrecision;
use crate::precision_policy::{PolicyPrecision, caps_named, tier_named};

/// 2026-09-29: Every KernelCaps bit: the declared formats, whatever kernels are built today.
const ALL_CAPS: [&str; 4] = [
    "w8a8_decode",
    "w8a8_moe_decode",
    "w8a8_block_scaled_decode",
    "fp8_lm_head_batched",
];

/// 2026-09-29: `Ok` when `config` (and `hf_quant`, if the checkpoint ships one) agree with the
/// instance and its loaded circuit.
pub fn check(
    inst: &Instance,
    loaded: &Loaded,
    config: &str,
    hf_quant: Option<&str>,
) -> Result<(), VennError> {
    let bad = |d: String| VennError::Checkpoint(d);
    let cfg: serde_json::Value =
        serde_json::from_str(config).map_err(|e| bad(format!("config.json: {e}")))?;
    let mut problems = Vec::new();
    let kinds = config_layer_kinds(&cfg)?;
    if kinds != inst.shape.layer_kinds {
        problems.push(format!(
            "layer kinds: config.json has {} layers, INSTANCES.toml {} (or the kinds differ)",
            kinds.len(),
            inst.shape.layer_kinds.len()
        ));
    }
    let qc = match hf_quant {
        Some(text) => serde_json::from_str::<serde_json::Value>(text)
            .map_err(|e| bad(format!("hf_quant_config.json: {e}")))?,
        None => cfg
            .get("quantization_config")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    };
    let plan = if qc.is_null() {
        DeclaredPrecisionPlan::UNDECLARED
    } else {
        DeclaredPrecisionPlan::from_quantization_config(&qc)
            .map_err(|e| bad(format!("quantization: {e:#}")))?
    };
    let caps: Vec<String> = ALL_CAPS.iter().map(|c| c.to_string()).collect();
    let policy = WeightQuantPolicy::new(
        tier_named("declared").map_err(|e| bad(e.to_string()))?,
        &plan,
        caps_named(&caps).map_err(|e| bad(e.to_string()))?,
    );
    let declared = PolicyPrecision::new(policy, &[]);
    let c = &loaded.circuit;
    for n in c.nodes.iter().filter(|n| n.weight.is_some()) {
        let Some(module) = n.binding.first() else {
            continue;
        };
        let want = declared.linear(&module.replace('*', "0"));
        let act = n.inputs.first().map(|&e| c.edges[e].format);
        if n.weight != Some(want.weight) || act != Some(want.activation) {
            problems.push(format!(
                "{module}: the circuit runs {}/{}, the checkpoint declares {}/{}",
                n.weight.map(|w| w.name()).unwrap_or_default(),
                act.map(|a| a.name()).unwrap_or_default(),
                want.weight,
                want.activation
            ));
        }
    }
    problems.dedup();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(bad(format!(
            "{} disagrees with the checkpoint:\n  {}",
            inst.recipe,
            problems.join("\n  ")
        )))
    }
}
