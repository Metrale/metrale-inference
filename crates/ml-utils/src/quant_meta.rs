// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The mock's quantization metadata must declare, for every kept module, exactly the
//! precision the source declares for the module it came from. Renaming exact names is not
//! enough: a compressed-tensors pattern such as `re:.*layers\.(56|...|63)\.mlp\..*` no longer
//! names a renumbered layer. Each module whose declared precision changed is pinned by an exact
//! target in a config group with the same scheme (exact names take precedence over patterns),
//! or added to `ignore` when it is unquantized; whatever still differs is refused.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - The precision of a module is what `DeclaredPrecisionPlan::resolve` says, the resolution
//!   the engine serves with: nothing here re-implements matching.
//! - Pins are added only to compressed-tensors metadata; another dialect that changed is refused.

use metrale_config::{DeclaredPrecisionPlan, LayerPrecision};
use serde_json::Value;

use crate::error::{MlError, Result};

/// 2026-10-03: One module to check: its source name and its mock name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePair {
    /// 2026-10-03: The module in the source checkpoint.
    pub source: String,
    /// 2026-10-03: The same module in the mock.
    pub mock: String,
}

fn plan_of(qc: &Value, what: &str) -> Result<DeclaredPrecisionPlan> {
    DeclaredPrecisionPlan::from_quantization_config(qc)
        .map_err(|e| MlError::Quant(format!("{what}: {e:#}")))
}

/// 2026-10-03: The precision each group of a compressed-tensors block declares on its own, by
/// group name.
fn group_precisions(qc: &Value) -> Result<Vec<(String, LayerPrecision)>> {
    let Some(groups) = qc.get("config_groups").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (name, g) in groups {
        let mut one = qc.clone();
        if let Some(o) = one.as_object_mut() {
            o.insert(
                "config_groups".into(),
                Value::Object([(name.clone(), g.clone())].into_iter().collect()),
            );
            o.insert("ignore".into(), Value::Array(Vec::new()));
        }
        let p = plan_of(&one, &format!("config group `{name}`"))?;
        if let Some(rule) = p.rules.first() {
            out.push((name.clone(), rule.precision));
        }
    }
    Ok(out)
}

fn is_compressed_tensors(qc: &Value) -> bool {
    qc.get("quant_method").and_then(Value::as_str) == Some("compressed-tensors")
}

fn mismatches(
    src: &DeclaredPrecisionPlan,
    mock: &DeclaredPrecisionPlan,
    modules: &[ModulePair],
) -> Vec<(ModulePair, LayerPrecision, LayerPrecision)> {
    modules
        .iter()
        .filter_map(|m| {
            let want = src.resolve(&m.source);
            let got = mock.resolve(&m.mock);
            (want != got).then(|| (m.clone(), want, got))
        })
        .collect()
}

/// 2026-10-03: Make `mock_qc` (already renamed) declare for every pair what `source_qc` declares,
/// pinning exact targets where a compressed-tensors pattern no longer matches. Returns the
/// number of modules pinned.
pub fn reconcile(source_qc: &Value, mock_qc: &mut Value, modules: &[ModulePair]) -> Result<usize> {
    let src = plan_of(source_qc, "source")?;
    let first = mismatches(&src, &plan_of(mock_qc, "mock")?, modules);
    if first.is_empty() {
        return Ok(0);
    }
    if !is_compressed_tensors(mock_qc) {
        return Err(refusal(&first));
    }
    let groups = group_precisions(mock_qc)?;
    for (m, want, _) in &first {
        let list_key = if *want == LayerPrecision::UNQUANTIZED {
            None
        } else {
            let g = groups.iter().find(|(_, p)| p == want).ok_or_else(|| {
                MlError::Quant(format!(
                    "`{}` declares {} in the source, and no config group of the mock has that \
                     scheme",
                    m.source,
                    want.label()
                ))
            })?;
            Some(g.0.clone())
        };
        let list = match &list_key {
            Some(g) => mock_qc
                .get_mut("config_groups")
                .and_then(|c| c.get_mut(g))
                .and_then(|g| g.get_mut("targets")),
            None => mock_qc.get_mut("ignore"),
        };
        match list.and_then(Value::as_array_mut) {
            Some(a) => a.push(Value::String(m.mock.clone())),
            None => {
                return Err(MlError::Quant(format!(
                    "cannot pin `{}`: the mock's {} list is missing",
                    m.mock,
                    list_key.map_or("ignore".to_string(), |g| format!("`{g}` targets"))
                )));
            }
        }
    }
    let left = mismatches(&src, &plan_of(mock_qc, "mock (pinned)")?, modules);
    if !left.is_empty() {
        return Err(refusal(&left));
    }
    Ok(first.len())
}

fn refusal(list: &[(ModulePair, LayerPrecision, LayerPrecision)]) -> MlError {
    let shown: Vec<String> = list
        .iter()
        .take(5)
        .map(|(m, want, got)| {
            format!(
                "{} -> {}: {} != {}",
                m.source,
                m.mock,
                got.label(),
                want.label()
            )
        })
        .collect();
    MlError::Quant(format!(
        "{} module(s) would change declared precision: {}{}",
        list.len(),
        shown.join("; "),
        if list.len() > 5 { "; ..." } else { "" }
    ))
}

#[cfg(test)]
#[path = "quant_meta_tests.rs"]
mod quant_meta_tests;
