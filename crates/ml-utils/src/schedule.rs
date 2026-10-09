// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Which layers a mock keeps. Layers are grouped into *units* (one layout period
//! under an `interval` rule, else one layer), and units into *signatures*: the layer kinds plus
//! every tensor's (name suffix, dtype, shape) and every module's declared precision. A mock keeps
//! the first `n` units of each signature, in source order, and renumbers them from 0.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Keeping whole periods keeps the layout rule true of the renumbered layers.
//! - Two units share a signature only if every per-layer kernel they launch has the same shapes
//!   and formats, so a mock's per-unit cost is the full model's per-unit cost.

use std::collections::BTreeMap;

use metrale_circuit::LayerSchedule;
use metrale_circuit::circuit_toml::LayoutRule;
use metrale_config::DeclaredPrecisionPlan;
use sha2::{Digest, Sha256};

use crate::error::{MlError, Result};
use crate::index::{TensorEntry, TensorIndex, hex};

/// 2026-10-03: One layer signature and how a mock samples it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// 2026-10-03: The layer kinds of one unit, in order.
    pub kinds: Vec<String>,
    /// 2026-10-03: Distinct `module=precision` lines of one unit, layer indices as `*`.
    pub precision: Vec<String>,
    /// 2026-10-03: sha256 of the full signature text.
    pub digest: String,
    /// 2026-10-03: Source layers of every unit with this signature, in order.
    pub units: Vec<Vec<usize>>,
    /// 2026-10-03: How many of those units the mock keeps (the first ones).
    pub kept: usize,
}

/// 2026-10-03: The layers a mock keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// 2026-10-03: Signatures in order of first appearance.
    pub signatures: Vec<Signature>,
    /// 2026-10-03: Source layer -> mock layer, for every kept layer.
    pub renumber: BTreeMap<usize, usize>,
}

impl Selection {
    /// 2026-10-03: The kept source layers, ascending.
    pub fn kept_layers(&self) -> Vec<usize> {
        self.renumber.keys().copied().collect()
    }
}

/// 2026-10-03: The units of `s`: whole periods under an `interval` rule, else single layers.
pub fn units(s: &LayerSchedule) -> Result<Vec<Vec<usize>>> {
    let n = s.layer_kinds.len();
    match s.layout {
        LayoutRule::Interval { period } => {
            if !n.is_multiple_of(period) {
                return Err(MlError::Checkpoint(format!(
                    "{n} layers is not a whole number of {period}-layer periods; whole periods \
                     are what a mock keeps"
                )));
            }
            Ok((0..n / period)
                .map(|p| (p * period..(p + 1) * period).collect())
                .collect())
        }
        LayoutRule::List => Ok((0..n).map(|i| vec![i]).collect()),
    }
}

/// 2026-10-03: The signature text of layer `layer`, relative to its own module path.
fn layer_text(
    s: &LayerSchedule,
    entries: &[&TensorEntry],
    plan: &DeclaredPrecisionPlan,
    layer: usize,
) -> (String, Vec<String>) {
    let prefix = s.module_of(layer);
    let mut text = format!("kind={}\n", s.layer_kinds[layer].name());
    let mut modules = BTreeMap::new();
    for e in entries {
        let rel = &e.name[prefix.len()..];
        text.push_str(&format!("t {rel} {} {:?}\n", e.dtype.name(), e.shape));
        // 2026-10-03: A module's declared precision is a property of its weight; `A_log` and
        // similar parameters sit on modules no quantization block describes.
        let weight = |last: &str| last == "weight" || last == "weight_packed";
        if let Some((module, _)) = e.name.rsplit_once('.').filter(|(_, last)| weight(last)) {
            modules
                .entry(module[prefix.len()..].to_string())
                .or_insert_with(|| plan.resolve(module));
        }
    }
    let mut lines = std::collections::BTreeSet::new();
    for (module, p) in &modules {
        text.push_str(&format!("p {module} {p:?}\n"));
        lines.insert(format!("{}={}", star_digits(module), p.label()));
    }
    (text, lines.into_iter().collect())
}

/// 2026-10-03: `module` with every all-digit segment written `*` (expert indices).
fn star_digits(module: &str) -> String {
    module
        .split('.')
        .map(|seg| {
            if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
                "*"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// 2026-10-03: The signatures of the checkpoint, each with the units the counts keep. `counts`
/// receives the number of signatures and returns one count per signature.
pub fn select(
    s: &LayerSchedule,
    index: &TensorIndex,
    plan: &DeclaredPrecisionPlan,
    counts: impl FnOnce(usize) -> Result<Vec<u32>>,
) -> Result<Selection> {
    let mut by_layer: BTreeMap<usize, Vec<&TensorEntry>> = BTreeMap::new();
    for e in index.iter() {
        if let Some(l) = s.layer_of(&e.name) {
            by_layer.entry(l).or_default().push(e);
        }
    }
    let mut signatures: Vec<Signature> = Vec::new();
    for unit in units(s)? {
        let mut text = String::new();
        let mut kinds = Vec::new();
        let mut precision = std::collections::BTreeSet::new();
        for (pos, &l) in unit.iter().enumerate() {
            let entries = by_layer.get(&l).map_or(&[][..], Vec::as_slice);
            let (t, p) = layer_text(s, entries, plan, l);
            text.push_str(&format!("layer {pos}\n{t}"));
            kinds.push(s.layer_kinds[l].name().to_string());
            precision.extend(p);
        }
        let digest = hex(&Sha256::digest(text.as_bytes()));
        match signatures.iter_mut().find(|g| g.digest == digest) {
            Some(g) => g.units.push(unit),
            None => signatures.push(Signature {
                kinds,
                precision: precision.into_iter().collect(),
                digest,
                units: vec![unit],
                kept: 0,
            }),
        }
    }
    let want = counts(signatures.len())?;
    let mut kept_layers = Vec::new();
    for (g, &n) in signatures.iter_mut().zip(&want) {
        let n = n as usize;
        if n > g.units.len() {
            return Err(MlError::Spec(format!(
                "a signature ({}) has {} units in the checkpoint; the spec keeps {n}",
                g.kinds.join(","),
                g.units.len()
            )));
        }
        g.kept = n;
        kept_layers.extend(g.units[..n].iter().flatten().copied());
    }
    kept_layers.sort_unstable();
    let renumber = kept_layers
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect();
    Ok(Selection {
        signatures,
        renumber,
    })
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;
