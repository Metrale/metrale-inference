// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The `[[family.op]]` table of KERNEL_FAMILIES.toml and its validation into an
//! [`OpSpec`]. Split from `families_file.rs` (the 500-line cap) unchanged.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: see [`super`].

use std::collections::BTreeSet;

use serde::Deserialize;

use super::{FamilyError, OpSpec, Values};
use crate::format::Format;
use crate::ir::{LinearRole, OpKind};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OpFile {
    pub(super) op: String,
    #[serde(default)]
    pub(super) roles: Vec<String>,
    #[serde(default)]
    pub(super) weight: Vec<String>,
    #[serde(default)]
    pub(super) activation: Vec<String>,
    #[serde(default)]
    pub(super) params: Values,
    #[serde(default)]
    pub(super) feeds: Vec<String>,
    #[serde(default)]
    pub(super) after: Vec<String>,
    #[serde(default)]
    pub(super) beside: Vec<String>,
}

/// 2026-09-29: True when `name` is an op base name of the circuit vocabulary.
pub(crate) fn known_op(name: &str) -> bool {
    matches!(name, "linear" | "act_quant") || OpKind::parse(name, None, None).is_ok()
}

pub(super) fn op_spec(family: &str, o: &OpFile) -> Result<OpSpec, FamilyError> {
    let unknown = |op: &str| FamilyError::UnknownOp {
        family: family.to_string(),
        op: op.to_string(),
    };
    // 2026-10-02: An `act_quant` may name its output format (`act_quant:nvfp4/g16`), so a
    // family implements only the quantizer it has.
    let quantizer =
        o.op.strip_prefix("act_quant:")
            .is_some_and(|f| Format::parse(f).is_ok());
    if !known_op(&o.op) && !quantizer {
        return Err(unknown(&o.op));
    }
    let qualified = |f: &String| match f.split_once(':') {
        Some(("linear", r)) => LinearRole::parse(r).is_some(),
        Some(("act_quant", fmt)) => Format::parse(fmt).is_ok(),
        Some(_) => false,
        None => known_op(f),
    };
    if let Some(f) = o
        .feeds
        .iter()
        .chain(&o.after)
        .chain(&o.beside)
        .find(|f| !qualified(f))
    {
        return Err(unknown(f));
    }
    let field = |detail: String| FamilyError::Field {
        family: family.to_string(),
        detail,
    };
    if !o.roles.is_empty() && o.op != "linear" {
        return Err(field(format!("op `{}` takes no roles", o.op)));
    }
    let roles = o
        .roles
        .iter()
        .map(|r| LinearRole::parse(r).ok_or_else(|| unknown(&format!("linear:{r}"))))
        .collect::<Result<_, _>>()?;
    let formats = |list: &[String]| {
        list.iter()
            .map(|s| Format::parse(s).map_err(|e| field(e.to_string())))
            .collect::<Result<BTreeSet<_>, _>>()
    };
    Ok(OpSpec {
        op: o.op.clone(),
        roles,
        weight: formats(&o.weight)?,
        activation: formats(&o.activation)?,
        params: o.params.clone(),
        feeds: o.feeds.iter().cloned().collect(),
        after: o.after.iter().cloned().collect(),
        beside: o.beside.iter().cloned().collect(),
    })
}
