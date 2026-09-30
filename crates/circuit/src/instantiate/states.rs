// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The state half of the instantiation builder: one `[[block.<name>.state]]` of
//! one block instance, its element count evaluated under the arch dims.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`crate::state`]; a recurrent state names its verify intermediates, a KV
//! side names none.

use std::collections::BTreeMap;

use crate::circuit_toml::{CircuitError, StateFile};
use crate::dims::DimExpr;
use crate::ir::Section;
use crate::state::{StateDecl, StateFormat, StateKind, VerifySteps};

/// 2026-09-30: The declaration `sf` of the block `template` instantiated at `prefix`.
pub(super) fn state_decl(
    template: &str,
    prefix: &str,
    layer: Option<usize>,
    section: Section,
    sf: &StateFile,
    dims: &BTreeMap<String, u64>,
) -> Result<StateDecl, CircuitError> {
    let err = |detail: String| CircuitError::State {
        block: template.to_string(),
        state: sf.id.clone(),
        detail,
    };
    let kind = StateKind::parse(&sf.kind)
        .ok_or_else(|| err(format!("kind `{}` is not recurrent or paged_kv", sf.kind)))?;
    let format = StateFormat::parse(&sf.format)
        .ok_or_else(|| err(format!("format `{}` is no dtype or {{key}}", sf.format)))?;
    let verify = match (kind, sf.verify.as_deref()) {
        (StateKind::Recurrent, Some(v)) => Some(
            VerifySteps::parse(v)
                .ok_or_else(|| err(format!("verify `{v}` is not h_steps or conv_steps")))?,
        ),
        (StateKind::Recurrent, None) => {
            return Err(err(
                "a recurrent state names its verify intermediates".into()
            ));
        }
        (StateKind::PagedKv, Some(_)) => {
            return Err(err("a KV side keeps no verify intermediates".into()));
        }
        (StateKind::PagedKv, None) => None,
    };
    let mut elements: u64 = 1;
    for axis in sf.shape.split(" x ") {
        let v = DimExpr::parse(axis)
            .and_then(|e| e.eval(dims))
            .map_err(|e| err(e.to_string()))?;
        elements = elements
            .checked_mul(v)
            .ok_or_else(|| err(format!("shape `{}` overflows", sf.shape)))?;
    }
    Ok(StateDecl {
        id: format!("{prefix}.{}", sf.id),
        local: sf.id.clone(),
        block: template.to_string(),
        layer,
        section,
        kind,
        format,
        elements,
        verify,
    })
}
