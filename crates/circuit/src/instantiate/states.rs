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
use crate::state::{Lifetime, StateAccess, StateDecl, StateFormat, StateKind, VerifySteps};

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
    let lifetime = Lifetime::parse(&sf.lifetime).ok_or_else(|| {
        err(format!(
            "lifetime `{}` is not model, sequence or verify",
            sf.lifetime
        ))
    })?;
    if lifetime != Lifetime::of_kind(kind) {
        return Err(err(format!(
            "a {} state lives `{}`, not `{}`",
            sf.kind,
            Lifetime::of_kind(kind).name(),
            sf.lifetime
        )));
    }
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
        lifetime,
    })
}

impl super::Builder<'_> {
    /// 2026-09-30: The states node `nf` of the block instance at `prefix` touches, resolved to
    /// indices into the circuit's states: each must be a state of that block instance.
    pub(super) fn state_refs(
        &self,
        template: &str,
        prefix: &str,
        nf: &crate::circuit_toml::NodeFile,
    ) -> Result<Vec<(usize, StateAccess)>, CircuitError> {
        let mut out = Vec::with_capacity(nf.state.len());
        for (name, access) in &nf.state {
            let err = |detail: String| CircuitError::State {
                block: template.to_string(),
                state: name.clone(),
                detail: format!("node `{}`: {detail}", nf.id),
            };
            let id = format!("{prefix}.{name}");
            let i = self
                .circuit
                .states
                .iter()
                .position(|s| s.id == id)
                .ok_or_else(|| err("the block declares no such state".into()))?;
            let a = StateAccess::parse(access).ok_or_else(|| {
                err(format!(
                    "access `{access}` is not read, write, update or snapshot"
                ))
            })?;
            out.push((i, a));
        }
        Ok(out)
    }
}

/// 2026-09-30: Every state of a block instance (`states[first_state..]`) is touched as its
/// kind requires by the instance's nodes (`nodes[first_node..]`): a recurrent state updated by
/// exactly one node and snapshotted by at most one, and only when it keeps verify
/// intermediates; a KV side written by exactly one node and read by at least one.
pub(super) fn check_state_access(
    template: &str,
    circuit: &crate::ir::Circuit,
    first_state: usize,
    first_node: usize,
) -> Result<(), CircuitError> {
    for (i, s) in circuit.states.iter().enumerate().skip(first_state) {
        let count = |a: StateAccess| {
            circuit.nodes[first_node..]
                .iter()
                .flat_map(|n| &n.state)
                .filter(|&&(si, sa)| si == i && sa == a)
                .count()
        };
        let err = |detail: String| CircuitError::State {
            block: template.to_string(),
            state: s.local.clone(),
            detail,
        };
        let (update, snapshot, write, read) = (
            count(StateAccess::Update),
            count(StateAccess::Snapshot),
            count(StateAccess::Write),
            count(StateAccess::Read),
        );
        match s.kind {
            StateKind::Recurrent => {
                if update != 1 || write + read > 0 {
                    return Err(err(format!(
                        "a recurrent state is updated by exactly one node and not read or \
                         written otherwise (update {update}, read {read}, write {write})"
                    )));
                }
                if snapshot > 1 || (snapshot == 1 && s.verify.is_none()) {
                    return Err(err(format!(
                        "snapshotted by {snapshot} nodes (at most one, and only with verify \
                         intermediates)"
                    )));
                }
            }
            StateKind::PagedKv => {
                if write != 1 || read == 0 || update + snapshot > 0 {
                    return Err(err(format!(
                        "a KV side is written by exactly one node and read by at least one \
                         (write {write}, read {read}, update {update}, snapshot {snapshot})"
                    )));
                }
            }
        }
    }
    Ok(())
}
