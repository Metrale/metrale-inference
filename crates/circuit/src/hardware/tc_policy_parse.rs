// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `[tensor_core_policy]` of a class's `HARDWARE.toml` into [`super::TcPolicy`].
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: see [`super`]; an unknown key, op, role, mode, format or kind is an error.

use std::collections::BTreeSet;

use serde::Deserialize;

use super::{Exempt, ExemptKind, OpMatch, Require, TcPolicy, Weights};
use crate::format::Format;
use crate::rules::{KernelId, Mode};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFile {
    require: Vec<RequireFile>,
    #[serde(default)]
    exempt: Vec<ExemptFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequireFile {
    ops: Vec<String>,
    modes: Vec<String>,
    min_rows: u64,
    weights: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExemptFile {
    ops: Vec<String>,
    modes: Vec<String>,
    rows: [u64; 2],
    kernels: Vec<String>,
    kind: String,
    reason: String,
    evidence: Option<String>,
}

fn ops(list: &[String]) -> Result<Vec<OpMatch>, String> {
    if list.is_empty() {
        return Err("names no op".into());
    }
    list.iter().map(|o| OpMatch::parse(o)).collect()
}

fn modes(list: &[String]) -> Result<BTreeSet<Mode>, String> {
    if list.is_empty() {
        return Err("names no mode".into());
    }
    list.iter()
        .map(|m| Mode::parse(m).ok_or_else(|| format!("unknown mode `{m}`")))
        .collect()
}

fn weights(list: &[String]) -> Result<Weights, String> {
    match list {
        [one] if one == "any" => Ok(Weights::Any),
        [] => Err("names no weight format (`[\"any\"]` covers every one)".into()),
        _ => list
            .iter()
            .map(|f| Format::parse(f).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()
            .map(Weights::Only),
    }
}

/// 2026-10-02: The class's policy from its parsed `HARDWARE.toml`: `None` when it declares
/// none.
pub fn parse_policy(hardware: &toml::Table) -> Result<Option<TcPolicy>, String> {
    let Some(v) = hardware.get("tensor_core_policy") else {
        return Ok(None);
    };
    let f: PolicyFile = v
        .clone()
        .try_into()
        .map_err(|e: toml::de::Error| format!("[tensor_core_policy]: {e}"))?;
    if f.require.is_empty() {
        return Err("[tensor_core_policy] states no requirement".into());
    }
    let require = f
        .require
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let at = |e: String| format!("[[tensor_core_policy.require]] #{}: {e}", i + 1);
            if r.min_rows == 0 {
                return Err(at("min_rows is at least 1".into()));
            }
            Ok(Require {
                ops: ops(&r.ops).map_err(at)?,
                modes: modes(&r.modes).map_err(at)?,
                min_rows: r.min_rows,
                weights: weights(&r.weights).map_err(at)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let exempt = f
        .exempt
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let at = |m: String| format!("[[tensor_core_policy.exempt]] #{}: {m}", i + 1);
            let kind = match e.kind.as_str() {
                "measured" => ExemptKind::Measured,
                "shape" => ExemptKind::Shape,
                "backlog" => ExemptKind::Backlog,
                other => {
                    return Err(at(format!("kind `{other}` (measured | shape | backlog)")));
                }
            };
            if kind == ExemptKind::Measured && e.evidence.as_deref().is_none_or(str::is_empty) {
                return Err(at("a measured exemption names its evidence".into()));
            }
            if e.rows[0] == 0 || e.rows[0] > e.rows[1] {
                return Err(at(format!("rows {:?} is not a range from 1", e.rows)));
            }
            if e.reason.trim().is_empty() || e.kernels.is_empty() {
                return Err(at("states its reason and the kernels it allows".into()));
            }
            let kernels = e
                .kernels
                .iter()
                .map(|k| {
                    k.split_once("::")
                        .filter(|(m, n)| !m.is_empty() && !n.is_empty())
                        .map(|(m, n)| KernelId {
                            module: m.to_string(),
                            func: n.to_string(),
                        })
                        .ok_or_else(|| at(format!("kernel `{k}` is not `module::function`")))
                })
                .collect::<Result<_, _>>()?;
            Ok(Exempt {
                ops: ops(&e.ops).map_err(at)?,
                modes: modes(&e.modes).map_err(at)?,
                rows: (e.rows[0], e.rows[1]),
                kernels,
                kind,
                reason: e.reason.clone(),
                evidence: e.evidence.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Some(TcPolicy { require, exempt }))
}
