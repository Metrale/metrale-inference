// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `plan_mock`: from a checkpoint's config, quantization metadata and tensor index,
//! plus a spec, the complete plan of a mock checkpoint: its config.json and metadata, every
//! output tensor and how its bytes are synthesized, and the resolved spec with its digest.
//! `synthesize` turns one unit of the plan into bytes. The checkpoint writer (`mockify`) and the
//! `--mock` weight loader both run exactly these two functions.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Pure: the plan is a function of the inputs.
//! - Every kept source tensor appears in exactly one unit; every unit's bytes have the size its
//!   tensors' dtype and shape give (`synthesize` refuses otherwise).
//! - A tensor's bytes depend on the seed, its source name, dtype and shape, the source layer
//!   count and (for routers) the profile, never on which other layers the mock keeps.

use metrale_circuit::{layer_schedule, map_checkpoint};
use metrale_config::DeclaredPrecisionPlan;
use serde_json::Value;

use crate::error::{MlError, Result};
use crate::index::{Dtype, TensorIndex};
use crate::quant_meta::{ModulePair, reconcile};
use crate::rename::Renamer;
use crate::rng::Stream;
use crate::routing::RoutingProfile;
use crate::schedule::{Selection, select};
use crate::scheme::QuantGroup;
use crate::spec::MockSpec;
use crate::synth::{GroupDtypes, encode, quantize_group};
use crate::values::{ValueClass, fill};

mod resolved;
mod units;

/// 2026-10-03: What a mock is planned from.
#[derive(Debug, Clone, Copy)]
pub struct MockInputs<'a> {
    /// 2026-10-03: The source checkpoint's id (`org/name`) or directory.
    pub source_id: &'a str,
    /// 2026-10-03: The source revision (commit), when known.
    pub revision: Option<&'a str>,
    /// 2026-10-03: The source config.json text.
    pub config_json: &'a str,
    /// 2026-10-03: The source `hf_quant_config.json` text, when it has one.
    pub hf_quant_config: Option<&'a str>,
    /// 2026-10-03: The source tensor index.
    pub index: &'a TensorIndex,
    /// 2026-10-03: The spec.
    pub spec: &'a MockSpec,
    /// 2026-10-03: The routing profile the spec names (`routing.mode = "histogram"`).
    pub routing: Option<&'a RoutingProfile>,
}

/// 2026-10-03: One tensor of the mock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutTensor {
    /// 2026-10-03: Its name in the source.
    pub source: String,
    /// 2026-10-03: Its name in the mock.
    pub name: String,
    /// 2026-10-03: Element type (the source's).
    pub dtype: Dtype,
    /// 2026-10-03: Shape (the source's).
    pub shape: Vec<u64>,
}

impl OutTensor {
    /// 2026-10-03: Stored bytes.
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.dtype.bytes()
    }
}

/// 2026-10-03: One synthesis unit: a plain tensor, or a quantized weight with its scales.
#[derive(Debug, Clone, PartialEq)]
pub enum Unit {
    /// 2026-10-03: A tensor written from its values as they are.
    Plain {
        /// 2026-10-03: Index into [`MockPlan::tensors`].
        tensor: usize,
        /// 2026-10-03: Its values.
        class: ValueClass,
    },
    /// 2026-10-03: A quantized weight: values, then the scheme's encoding.
    Group {
        /// 2026-10-03: The group (source names).
        group: QuantGroup,
        /// 2026-10-03: Indices into [`MockPlan::tensors`], in [`QuantGroup::tensors`] order.
        tensors: Vec<usize>,
        /// 2026-10-03: The weight's values before quantization.
        class: ValueClass,
        /// 2026-10-03: The stored dtypes.
        dtypes: GroupDtypes,
    },
}

/// 2026-10-03: How one router was fitted to the profile.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterFit {
    /// 2026-10-03: The router tensor in the mock.
    pub tensor: String,
    /// 2026-10-03: The source layer whose profile row it reproduces.
    pub source_layer: usize,
    /// 2026-10-03: Total-variation distance of the fit on its own samples.
    pub tv: f64,
    /// 2026-10-03: Experts raised to the floor share.
    pub floored: usize,
}

/// 2026-10-03: A complete mock plan.
#[derive(Debug, Clone)]
pub struct MockPlan {
    /// 2026-10-03: The synthesis seed.
    pub seed: u64,
    /// 2026-10-03: The resolved spec (TOML) written beside the output.
    pub resolved: String,
    /// 2026-10-03: sha256 of [`MockPlan::resolved`]: the mock's disclosed identity.
    pub digest: String,
    /// 2026-10-03: The mock's config.json.
    pub config_json: String,
    /// 2026-10-03: The mock's `hf_quant_config.json`, when the source has one.
    pub hf_quant_config: Option<String>,
    /// 2026-10-03: Every output tensor, in unit order.
    pub tensors: Vec<OutTensor>,
    /// 2026-10-03: The synthesis units.
    pub units: Vec<Unit>,
    /// 2026-10-03: The kept layers.
    pub selection: Selection,
    /// 2026-10-03: The source's layer count.
    pub layers_full: usize,
    /// 2026-10-03: Modules pinned by an exact quantization target.
    pub pinned: usize,
    /// 2026-10-03: Router fits (histogram routing only).
    pub routers: Vec<RouterFit>,
}

impl MockPlan {
    /// 2026-10-03: Total stored bytes.
    pub fn bytes(&self) -> u64 {
        self.tensors.iter().map(OutTensor::bytes).sum()
    }
}

fn parse_json(text: &str, what: &str) -> Result<Value> {
    serde_json::from_str(text).map_err(|e| MlError::Checkpoint(format!("{what}: {e}")))
}

fn to_text(v: &Value) -> String {
    serde_json::to_string_pretty(v).expect("a JSON value serializes") + "\n"
}

/// 2026-10-03: Plan the mock `inp` describes.
pub fn plan_mock(inp: &MockInputs<'_>) -> Result<MockPlan> {
    let schedule = layer_schedule(inp.config_json)?;
    let mapped = map_checkpoint(inp.config_json)?;
    let config = parse_json(inp.config_json, "config.json")?;
    let sidecar = inp
        .hf_quant_config
        .map(|t| parse_json(t, "hf_quant_config.json"))
        .transpose()?;
    let config_qc = config.get("quantization_config").filter(|v| !v.is_null());
    let declared_qc = config_qc.or(sidecar.as_ref());
    let declared = match declared_qc {
        Some(qc) => DeclaredPrecisionPlan::from_quantization_config(qc)
            .map_err(|e| MlError::Quant(format!("source: {e:#}")))?,
        None => DeclaredPrecisionPlan::UNDECLARED,
    };
    let selection = select(&schedule, inp.index, &declared, |n| inp.spec.counts(n))?;
    let renamer = Renamer::new(&schedule, &selection.renumber);

    let mut config_out = renamer.rewrite_config(&config)?;
    let mut sidecar_out = sidecar.clone();
    if let Some(s) = sidecar_out.as_mut() {
        renamer.rewrite_metadata(s);
    }
    let pairs = module_pairs(inp.index, &renamer, &schedule);
    let mut pinned = 0;
    if let (Some(src), Some(dst)) = (config_qc, config_out.get_mut("quantization_config")) {
        pinned += reconcile(src, dst, &pairs)?;
    }
    if let (Some(src), Some(dst)) = (sidecar.as_ref(), sidecar_out.as_mut()) {
        pinned += reconcile(src, dst, &pairs)?;
    }

    let block = declared_qc.and_then(crate::inspect::block_size);
    let built = units::build(&units::Ctx {
        index: inp.index,
        renamer: &renamer,
        schedule: &schedule,
        dims: &mapped.shape.dims,
        spec: inp.spec,
        routing: inp.routing,
        block,
    })?;
    let mut plan = MockPlan {
        seed: inp.spec.seed,
        resolved: String::new(),
        digest: String::new(),
        config_json: to_text(&config_out),
        hf_quant_config: sidecar_out.as_ref().map(to_text),
        tensors: built.tensors,
        units: built.units,
        selection,
        layers_full: schedule.layer_kinds.len(),
        pinned,
        routers: built.routers,
    };
    plan.resolved = resolved::render(inp, &mapped.arch, &plan, built.bias_channel);
    plan.digest = resolved_digest(plan.resolved.as_bytes());
    Ok(plan)
}

/// 2026-10-03: A mock's digest: sha256 of its resolved spec's bytes. The one definition, used by
/// the plan and by every reader that recognises a mock from its `mock.resolved.toml`.
pub fn resolved_digest(resolved: &[u8]) -> String {
    crate::index::hex(&<sha2::Sha256 as sha2::Digest>::digest(resolved))
}

/// 2026-10-03: Source and mock module of every kept tensor under a layer.
fn module_pairs(
    index: &TensorIndex,
    renamer: &Renamer<'_>,
    schedule: &metrale_circuit::LayerSchedule,
) -> Vec<ModulePair> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for e in index.iter() {
        if schedule.layer_of(&e.name).is_none() {
            continue;
        }
        let Some((module, _)) = e.name.rsplit_once('.') else {
            continue;
        };
        if !seen.insert(module.to_string()) {
            continue;
        }
        if let Some(mock) = renamer.name(module) {
            out.push(ModulePair {
                source: module.to_string(),
                mock,
            });
        }
    }
    out
}

/// 2026-10-03: The bytes of unit `u`: `(tensor index, bytes)` for each of its tensors.
pub fn synthesize(plan: &MockPlan, u: usize) -> Result<Vec<(usize, Vec<u8>)>> {
    let unit = plan
        .units
        .get(u)
        .ok_or_else(|| MlError::Spec(format!("unit {u} of {}", plan.units.len())))?;
    let stream_of =
        |t: &OutTensor| Stream::for_tensor(plan.seed, &t.source, t.dtype.name(), &t.shape);
    let out = match unit {
        Unit::Plain { tensor, class } => {
            let t = &plan.tensors[*tensor];
            let n = t.shape.iter().product::<u64>() as usize;
            let values = fill(class, stream_of(t), n);
            vec![(*tensor, encode(&t.name, &values, t.dtype)?)]
        }
        Unit::Group {
            group,
            tensors,
            class,
            dtypes,
        } => {
            let w = &plan.tensors[tensors[0]];
            let values = fill(class, stream_of(w), (group.rows * group.cols) as usize);
            let bytes = quantize_group(group, &values, *dtypes)?;
            if bytes.len() != tensors.len() {
                return Err(MlError::Tensor {
                    name: w.source.clone(),
                    why: format!("{} tensors encoded, {} planned", bytes.len(), tensors.len()),
                });
            }
            tensors.iter().copied().zip(bytes).collect()
        }
    };
    for (t, b) in &out {
        let want = plan.tensors[*t].bytes();
        if b.len() as u64 != want {
            return Err(MlError::Tensor {
                name: plan.tensors[*t].name.clone(),
                why: format!("{} bytes synthesized, {want} planned", b.len()),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod plan_tests;

#[cfg(test)]
#[path = "plan_routing_tests.rs"]
mod plan_routing_tests;
