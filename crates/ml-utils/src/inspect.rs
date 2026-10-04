// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: What `met ml-utils inspect` reports about a checkpoint, from its config,
//! quantization metadata and tensor index alone: the arch, the layer signatures a mock samples
//! (with their bytes), the storage schemes, and for a MoE the expected number of distinct experts
//! a decode step touches at each concurrency, which is what shrinking the expert count would
//! change (ml-utils DESIGN.md 3).
//!
//! Owner: metrale-ml-utils.
//! Invariants: pure; equal inputs give equal reports.

use std::collections::BTreeMap;

use metrale_circuit::circuit_toml::LayoutRule;
use metrale_circuit::{layer_schedule, map_checkpoint};
use metrale_config::DeclaredPrecisionPlan;
use serde::Serialize;
use serde_json::Value;

use crate::error::{MlError, Result};
use crate::index::TensorIndex;
use crate::schedule::select;
use crate::scheme::{Nvfp4Global, Scheme, find_groups};

/// 2026-10-03: One layer signature.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SignatureInfo {
    /// 2026-10-03: Layer kinds of one unit.
    pub kinds: Vec<String>,
    /// 2026-10-03: Units with this signature in the checkpoint.
    pub units: usize,
    /// 2026-10-03: The first unit's layers.
    pub first_unit: Vec<usize>,
    /// 2026-10-03: `module=precision` lines.
    pub precision: Vec<String>,
    /// 2026-10-03: Stored bytes of one unit.
    pub bytes_per_unit: u64,
}

/// 2026-10-03: The MoE shape and its expert traffic.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoeInfo {
    /// 2026-10-03: Routed experts per layer.
    pub experts: u64,
    /// 2026-10-03: Experts per token.
    pub top_k: u64,
    /// 2026-10-03: `(concurrency, expected distinct experts per step)` under uniform routing,
    /// `E * (1 - (1 - k/E)^C)`.
    pub unique_experts: Vec<(u64, f64)>,
}

/// 2026-10-03: The report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Inspection {
    /// 2026-10-03: Circuit arch.
    pub arch: String,
    /// 2026-10-03: config.json `model_type`.
    pub model_type: String,
    /// 2026-10-03: Main-stack layers.
    pub layers: usize,
    /// 2026-10-03: `interval/<period>` or `list`.
    pub layout: String,
    /// 2026-10-03: Tensors.
    pub tensors: usize,
    /// 2026-10-03: Stored bytes.
    pub bytes: u64,
    /// 2026-10-03: Bytes outside every layer (embedding, head, draft head, vision).
    pub fixed_bytes: u64,
    /// 2026-10-03: Layer signatures in order of first appearance.
    pub signatures: Vec<SignatureInfo>,
    /// 2026-10-03: Quantized weights per storage scheme.
    pub schemes: BTreeMap<String, usize>,
    /// 2026-10-03: The MoE, when the arch has one.
    pub moe: Option<MoeInfo>,
}

/// 2026-10-03: Concurrencies the expert-traffic table covers.
pub const CONCURRENCIES: [u64; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// 2026-10-03: The `weight_block_size` an HF `fp8` block declares.
pub fn block_size(qc: &Value) -> Option<(u64, u64)> {
    let b = qc.get("weight_block_size")?.as_array()?;
    match b.as_slice() {
        [r, c] => Some((r.as_u64()?, c.as_u64()?)),
        _ => None,
    }
}

/// 2026-10-03: The quantization block that declares the checkpoint's precision: config.json's
/// own, else the sidecar's (the serve rule).
pub fn declared_block(config: &Value, sidecar: Option<&Value>) -> Option<Value> {
    config
        .get("quantization_config")
        .filter(|v| !v.is_null())
        .cloned()
        .or_else(|| sidecar.cloned())
}

fn scheme_name(s: Scheme) -> String {
    match s {
        Scheme::Nvfp4(Nvfp4Global::ModelOpt) => "nvfp4 (ModelOpt global)".into(),
        Scheme::Nvfp4(Nvfp4Global::CompressedTensors) => "nvfp4 (compressed-tensors global)".into(),
        Scheme::Fp8Block { bn, bk } => format!("fp8 block {bn}x{bk}"),
        Scheme::Fp8Channel => "fp8 per channel".into(),
        Scheme::Fp8Tensor => "fp8 per tensor".into(),
    }
}

/// 2026-10-03: Inspect a checkpoint.
pub fn inspect(
    config_json: &str,
    hf_quant: Option<&str>,
    index: &TensorIndex,
) -> Result<Inspection> {
    let schedule = layer_schedule(config_json)?;
    let mapped = map_checkpoint(config_json)?;
    let config: Value = serde_json::from_str(config_json)
        .map_err(|e| MlError::Checkpoint(format!("config.json: {e}")))?;
    let sidecar = hf_quant
        .map(serde_json::from_str::<Value>)
        .transpose()
        .map_err(|e| MlError::Checkpoint(format!("hf_quant_config.json: {e}")))?;
    let qc = declared_block(&config, sidecar.as_ref());
    let declared = match &qc {
        Some(q) => DeclaredPrecisionPlan::from_quantization_config(q)
            .map_err(|e| MlError::Quant(format!("{e:#}")))?,
        None => DeclaredPrecisionPlan::UNDECLARED,
    };
    let sel = select(&schedule, index, &declared, |n| Ok(vec![1; n]))?;
    let mut layer_bytes = vec![0u64; schedule.layer_kinds.len()];
    let mut fixed_bytes = 0;
    for e in index.iter() {
        match schedule.layer_of(&e.name) {
            Some(l) if l < layer_bytes.len() => layer_bytes[l] += e.bytes(),
            _ => fixed_bytes += e.bytes(),
        }
    }
    let signatures = sel
        .signatures
        .iter()
        .map(|g| SignatureInfo {
            kinds: g.kinds.clone(),
            units: g.units.len(),
            first_unit: g.units[0].clone(),
            precision: g.precision.clone(),
            bytes_per_unit: g.units[0].iter().map(|&l| layer_bytes[l]).sum(),
        })
        .collect();
    let mut schemes = BTreeMap::new();
    for g in find_groups(index, qc.as_ref().and_then(block_size))? {
        *schemes.entry(scheme_name(g.scheme)).or_insert(0) += 1;
    }
    let dims = &mapped.shape.dims;
    let moe = match (dims.get("experts"), dims.get("top_k")) {
        (Some(&e), Some(&k)) if e > 0 => Some(MoeInfo {
            experts: e,
            top_k: k,
            unique_experts: CONCURRENCIES
                .iter()
                .map(|&c| {
                    (
                        c,
                        e as f64 * (1.0 - (1.0 - k as f64 / e as f64).powi(c as i32)),
                    )
                })
                .collect(),
        }),
        _ => None,
    };
    Ok(Inspection {
        arch: mapped.arch,
        model_type: mapped.model_type,
        layers: schedule.layer_kinds.len(),
        layout: match schedule.layout {
            LayoutRule::Interval { period } => format!("interval/{period}"),
            LayoutRule::List => "list".into(),
        },
        tensors: index.len(),
        bytes: index.bytes(),
        fixed_bytes,
        signatures,
        schemes,
        moe,
    })
}

#[cfg(test)]
#[path = "inspect_tests.rs"]
mod inspect_tests;
