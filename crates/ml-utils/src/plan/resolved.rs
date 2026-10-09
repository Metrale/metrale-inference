// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The resolved spec: the spec as parsed (a profile named by its digest, not its
//! path), the source's identity, and everything the plan derived. Written beside every mock and
//! hashed into the digest records disclose.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A pure rendering of the inputs and the plan: equal inputs give equal text.
//! - No local path enters the text, so the same mock planned on two machines has one digest.

use sha2::{Digest, Sha256};

use super::{MockInputs, MockPlan};
use crate::index::hex;
use crate::routing::BiasChannel;
use crate::spec::{MockSpec, RoutingMode, ValuesMode, toml_str};

fn list<T: ToString>(v: impl IntoIterator<Item = T>) -> String {
    format!(
        "[{}]",
        v.into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn str_list<'a>(v: impl IntoIterator<Item = &'a String>) -> String {
    list(v.into_iter().map(|s| toml_str(s)))
}

fn sha(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

/// 2026-10-03: The resolved spec text.
pub(super) fn render(
    inp: &MockInputs<'_>,
    arch: &str,
    plan: &MockPlan,
    channel: Option<BiasChannel>,
) -> String {
    let mut spec: MockSpec = inp.spec.clone();
    if let (RoutingMode::Histogram { path, calibration }, Some(p)) =
        (&mut spec.routing, inp.routing)
    {
        *path = format!("sha256:{}", p.digest);
        if let (Some(c), Some(cal)) = (calibration.as_mut(), inp.calibration) {
            *c = format!("sha256:{}", cal.digest);
        }
    }
    if let (ValuesMode::Stats { path }, Some(st)) = (&mut spec.values, inp.stats) {
        *path = format!("sha256:{}", st.digest);
    }
    let mut t = String::from("# met ml-utils: a mock (rehearsal) checkpoint's resolved spec.\n");
    t.push_str(&spec.canonical());
    t.push_str("\n[source]\n");
    t.push_str(&format!("id = {}\n", toml_str(inp.source_id)));
    if let Some(r) = inp.revision {
        t.push_str(&format!("revision = {}\n", toml_str(r)));
    }
    t.push_str(&format!(
        "config_sha256 = {}\n",
        toml_str(&sha(inp.config_json))
    ));
    if let Some(h) = inp.hf_quant_config {
        t.push_str(&format!("hf_quant_config_sha256 = {}\n", toml_str(&sha(h))));
    }
    t.push_str(&format!(
        "index_digest = {}\n",
        toml_str(&inp.index.digest())
    ));
    t.push_str("\n[derived]\n");
    t.push_str(&format!("arch = {}\n", toml_str(arch)));
    t.push_str(&format!("layers_full = {}\n", plan.layers_full));
    t.push_str(&format!(
        "layers_kept = {}\n",
        list(plan.selection.kept_layers())
    ));
    t.push_str(&format!(
        "residual_gain = {}\n",
        crate::values::residual_gain(plan.layers_full)
    ));
    if let Some(c) = channel {
        t.push_str(&format!(
            "bias_channel = {}\nbias_channel_value = {}\n",
            c.channel, c.k
        ));
    }
    t.push_str(&format!("quant_pins = {}\n", plan.pinned));
    t.push_str(&format!(
        "tensors = {}\nbytes = {}\n",
        plan.tensors.len(),
        plan.bytes()
    ));
    for g in &plan.selection.signatures {
        t.push_str("\n[[derived.signature]]\n");
        t.push_str(&format!("kinds = {}\n", str_list(&g.kinds)));
        t.push_str(&format!(
            "units_full = {}\nunits_kept = {}\n",
            g.units.len(),
            g.kept
        ));
        t.push_str(&format!("first_unit = {}\n", list(&g.units[0])));
        t.push_str(&format!("precision = {}\n", str_list(&g.precision)));
        t.push_str(&format!("digest = {}\n", toml_str(&g.digest)));
    }
    for r in &plan.routers {
        t.push_str("\n[[derived.router]]\n");
        t.push_str(&format!("tensor = {}\n", toml_str(&r.tensor)));
        t.push_str(&format!("source_layer = {}\n", r.source_layer));
        t.push_str(&format!(
            "fit_total_variation = {:.6}\nfloored = {}\ncalibration_gain = {}\n",
            r.tv, r.floored, r.gain
        ));
    }
    t
}
