// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The recurrent state an embedded circuit declares, for a model the engine has
//! already configured: the circuit is chosen by `model_type` and the declarations are
//! evaluated under dims the caller reads from its own (tensor-parallel local) config, so no
//! precision and no config.json is involved.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - `Ok(None)` only when no embedded circuit serves `model_type`; a circuit that serves it
//!   but cannot be read, or whose layers hold recurrent state in more than one block, is an
//!   error.

use std::collections::BTreeMap;

use super::{ARCHES, BLOCKS, config_maps};
use crate::circuit_toml::{CircuitError, parse_file};
use crate::instantiate::state_decl;
use crate::ir::Section;
use crate::state::{StateDecl, StateKind};

/// 2026-09-30: The recurrent states of one layer of the circuit that serves the engine-configured
/// `model_type` (`ConfigMap::serves_engine_model_type`), under
/// `dims`: the declarations of the one layer block that keeps recurrent state (GatedDeltaNet
/// `gdn`, Mamba2 `mamba`), ids `<block>.<state>`. `Ok(Some(empty))` for a circuit with no
/// recurrent layer.
pub fn recurrent_states(
    model_type: &str,
    dims: &BTreeMap<String, u64>,
) -> Result<Option<Vec<StateDecl>>, CircuitError> {
    let maps = config_maps().map_err(|e| CircuitError::Parse(e.to_string()))?;
    let Some(i) = maps
        .iter()
        .position(|m| m.serves_engine_model_type(model_type))
    else {
        return Ok(None);
    };
    let file = parse_file(ARCHES[i].circuit, &BLOCKS)?;
    let mut templates: Vec<&String> = file
        .layout
        .blocks
        .values()
        .flatten()
        .filter(|t| {
            file.block
                .get(*t)
                .is_some_and(|b| b.state.iter().any(|s| s.kind == "recurrent"))
        })
        .collect();
    templates.sort();
    templates.dedup();
    let template = match templates.as_slice() {
        [] => return Ok(Some(Vec::new())),
        [one] => *one,
        many => {
            return Err(CircuitError::Layout(format!(
                "circuit `{}`: recurrent state in more than one layer block ({many:?})",
                file.arch
            )));
        }
    };
    let mut out = Vec::new();
    let block = &file.block[template].state;
    for sf in block {
        let d = state_decl(template, template, None, Section::Main, sf, block, dims)?;
        if d.kind == StateKind::Recurrent {
            out.push(d);
        }
    }
    Ok(Some(out))
}

/// 2026-10-03: The caches the circuit serving `model_type` declares, under `dims`: those of its
/// recurrent layer block (`layer`, one instance per recurrent layer: the prefix and ring
/// snapshots, the carry stash and tables) and those of its prologue and epilogue blocks
/// (`model`, one instance: the last-hidden prefix row, the verify tables). The reserve and the
/// allocators size the snapshot pool and the carry from these (LIFECYCLE-DESIGN.md 15.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStates {
    /// 2026-10-03: One recurrent layer's caches.
    pub layer: Vec<StateDecl>,
    /// 2026-10-03: The model-level caches.
    pub model: Vec<StateDecl>,
}

/// 2026-10-03: [`CacheStates`] of the circuit serving `model_type`; `Ok(None)` as
/// [`recurrent_states`].
pub fn cache_states(
    model_type: &str,
    dims: &BTreeMap<String, u64>,
) -> Result<Option<CacheStates>, CircuitError> {
    let maps = config_maps().map_err(|e| CircuitError::Parse(e.to_string()))?;
    let Some(i) = maps
        .iter()
        .position(|m| m.serves_engine_model_type(model_type))
    else {
        return Ok(None);
    };
    let file = parse_file(ARCHES[i].circuit, &BLOCKS)?;
    let caches_of = |template: &str| -> Result<Vec<StateDecl>, CircuitError> {
        let Some(b) = file.block.get(template) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for sf in &b.state {
            let d = state_decl(template, template, None, Section::Main, sf, &b.state, dims)?;
            if d.kind.is_cache() {
                out.push(d);
            }
        }
        Ok(out)
    };
    let mut layer_templates: Vec<&String> = file
        .layout
        .blocks
        .values()
        .flatten()
        .filter(|t| {
            file.block
                .get(*t)
                .is_some_and(|b| b.state.iter().any(|s| s.kind == "recurrent"))
        })
        .collect();
    layer_templates.sort();
    layer_templates.dedup();
    let layer = match layer_templates.as_slice() {
        [] => Vec::new(),
        [one] => caches_of(one)?,
        many => {
            return Err(CircuitError::Layout(format!(
                "circuit `{}`: recurrent state in more than one layer block ({many:?})",
                file.arch
            )));
        }
    };
    let mut model = Vec::new();
    for t in file.prologue.iter().chain(&file.epilogue) {
        model.extend(caches_of(t)?);
    }
    Ok(Some(CacheStates { layer, model }))
}
