// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A layer's LoRA adapters as the circuit executor binds them
//! (`circuit_exec::lora::LoraLayer`), from the weights `set_lora_weights` installs on the layer:
//! the attention projections' routing tables, the active adapter's dense-FFN pairs and the
//! GatedDeltaNet out_proj pair. A layer whose adapters the circuit cannot run lists the reason as
//! unmodelled, so the executor refuses the model rather than run a plan without them.
//!
//! Owner: model-layers (FEATURES workstream).
//! Invariants:
//! - An adapted attention projection binds its routing table: the circuit folds attention per row
//!   (`lora_bgmv`), so a pair without a table is refused, never applied as a pair.
//! - The gate|up fold takes the gate and the up pair together; an adapter on one of them is
//!   refused.

use std::collections::BTreeMap;

use metrale_circuit::LinearRole;

use super::FfnComponent;
use super::ops::lora_delta::{LoraAttnWeights, LoraFfnWeights, LoraKernels, LoraPair};
use crate::circuit_exec::lora::LoraLayer;

impl FfnComponent {
    /// 2026-10-03: The dense FFN's installed adapter pairs; a MoE FFN's adapters are refused
    /// (`unmodelled`) until the MoE family's LoRA port.
    pub(crate) fn circuit_lora(&self, unmodelled: &mut Vec<String>) -> Option<&LoraFfnWeights> {
        match self {
            Self::Dense(d) => d.lora_weights(),
            Self::Moe(m) => {
                if m.lora.is_some() {
                    unmodelled.push("a MoE LoRA adapter (the MoE LoRA port is not done)".into());
                }
                None
            }
            Self::None => None,
        }
    }
}

/// 2026-10-03: What one layer holds: an attention layer's adapter, its FFN's, a GatedDeltaNet
/// layer's out_proj pair.
pub(crate) struct Installed<'a> {
    pub attn: Option<&'a LoraAttnWeights>,
    pub ffn: Option<&'a LoraFfnWeights>,
    pub gdn_out: Option<&'a (LoraPair, LoraKernels)>,
}

/// 2026-10-03: The layer's [`LoraLayer`], or why the circuit cannot run its adapters; `Ok(None)`
/// for a layer without adapters.
pub(crate) fn bind(i: Installed<'_>) -> Result<Option<LoraLayer>, String> {
    let mut routes = BTreeMap::new();
    let mut pairs = BTreeMap::new();
    let mut kernels = None;
    if let Some(a) = i.attn {
        kernels = Some(a.kernels);
        for (role, pair, route) in [
            (LinearRole::Q, a.q, a.q_route),
            (LinearRole::K, a.k, a.k_route),
            (LinearRole::V, a.v, a.v_route),
            (LinearRole::O, a.o, a.o_route),
        ] {
            match (pair, route) {
                (_, Some(r)) => {
                    routes.insert(role, r);
                }
                (Some(_), None) => {
                    return Err(format!(
                        "an attention `{}` adapter without its routing table",
                        role.name()
                    ));
                }
                (None, None) => {}
            }
        }
    }
    if let Some(f) = i.ffn {
        kernels = Some(f.kernels);
        match (f.gate, f.up) {
            (Some(g), Some(u)) => {
                pairs.insert(LinearRole::GateUp, vec![g, u]);
            }
            (None, None) => {}
            _ => {
                return Err(
                    "an FFN adapter on one of gate and up (the gate|up fold takes both)".into(),
                );
            }
        }
        if let Some(d) = f.down {
            pairs.insert(LinearRole::Down, vec![d]);
        }
    }
    if let Some((p, k)) = i.gdn_out {
        kernels = Some(*k);
        pairs.insert(LinearRole::GdnOut, vec![*p]);
    }
    Ok(kernels
        .filter(|_| !routes.is_empty() || !pairs.is_empty())
        .map(|kernels| LoraLayer {
            kernels,
            routes,
            pairs,
        }))
}

/// 2026-10-03: [`bind`], a refusal recorded in `unmodelled`.
pub(crate) fn bind_or_refuse(i: Installed<'_>, unmodelled: &mut Vec<String>) -> Option<LoraLayer> {
    bind(i).unwrap_or_else(|why| {
        unmodelled.push(why);
        None
    })
}

#[cfg(test)]
#[path = "circuit_lora_tests.rs"]
mod circuit_lora_tests;
