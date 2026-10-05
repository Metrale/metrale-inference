// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: A circuit from a checkpoint alone: its `config.json` and quantization metadata
//! pick the architecture, map to an arch shape through the architecture's declarative config
//! map (`kernels/circuits/<arch>.config.toml`), and give every projection its declared
//! formats. The model axis of `met circuit plan --checkpoint --hardware`.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Pure: the circuit, block and config-map TOMLs are embedded at build time; the caller
//!   passes the checkpoint's file text. No I/O, no environment.
//! - The architecture is chosen by the config's top-level `model_type` alone; a model type no
//!   map serves is refused, never approximated by a neighbour.
//! - Precision comes from config.json's own `quantization_config` when it has one, else from
//!   the sidecar `hf_quant_config.json` (the serve rule, `merge_sidecar_quant_config`).

use std::collections::BTreeMap;

use metrale_config::DeclaredPrecisionPlan;
use serde_json::Value;

use crate::circuit_toml::CircuitError;
use crate::config_map::{ConfigMap, ConfigMapError, MappedConfig, map_config};
use crate::declared_precision::DeclaredPrecision;
use crate::format::{Format, Scale};
use crate::instantiate::instantiate;
use crate::ir::{ArchShape, Circuit};
use crate::precision::{LinearFormats, PrecisionError};
use crate::precision_policy::{PolicyPrecision, caps_named, tier_named};

/// 2026-09-30: The quantization metadata a checkpoint ships beside its `config.json`.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantMetadata<'a> {
    /// 2026-09-30: `hf_quant_config.json` (ModelOpt), verbatim; `None` when absent. Read only
    /// when `config.json` has no `quantization_config`.
    pub hf_quant_config: Option<&'a str>,
}

/// 2026-09-30: Which formats the circuit's edges take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServePrecision {
    /// 2026-09-30: The checkpoint's declared formats ([`DeclaredPrecision`]).
    Declared,
    /// 2026-09-30: The engine's serving policy over the declared plan: the
    /// `--weight-quantization` tier, the kernel capabilities present, and the formats the
    /// engine chooses itself (as `INSTANCES.toml` states them for a golden instance).
    Policy {
        /// 2026-09-30: `declared` or `nvfp4`.
        tier: String,
        /// 2026-09-30: `KernelCaps` field names.
        caps: Vec<String>,
        /// 2026-09-30: Module glob to formats, first match wins.
        engine: Vec<(String, LinearFormats)>,
    },
}

/// 2026-09-30: A checkpoint resolved to its circuit.
#[derive(Debug, Clone)]
pub struct ResolvedCheckpoint {
    /// 2026-09-30: The circuit arch (`qwen3_5`, `qwen3_6_moe`, `nemotron_h`, `dense_gqa`).
    pub arch: String,
    /// 2026-09-30: The config's top-level `model_type`.
    pub model_type: String,
    /// 2026-09-30: Layer kinds and dims, from config.json.
    pub shape: ArchShape,
    /// 2026-09-30: The math parameters config.json states (RoPE, norm eps, routing, ...), as
    /// JSON text by name.
    pub params: BTreeMap<String, String>,
    /// 2026-09-30: The declared KV-cache format; `None` is 16-bit.
    pub kv_cache: Option<Format>,
    /// 2026-09-30: The instantiated circuit.
    pub circuit: Circuit,
}

/// 2026-09-30: Why a checkpoint has no circuit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CheckpointError {
    /// 2026-09-30: A file is not the JSON it should be.
    #[error("{file}: {detail}")]
    Json {
        /// 2026-09-30: `config.json` or `hf_quant_config.json`.
        file: &'static str,
        /// 2026-09-30: What was wrong.
        detail: String,
    },
    /// 2026-09-30: No config map serves this `model_type`.
    #[error("no circuit serves model_type `{model_type}` (served: {served})")]
    UnknownModelType {
        /// 2026-09-30: The config's `model_type`.
        model_type: String,
        /// 2026-09-30: The model types the embedded maps serve.
        served: String,
    },
    /// 2026-09-30: The config does not map.
    #[error(transparent)]
    Map(#[from] ConfigMapError),
    /// 2026-09-30: The quantization metadata is malformed, or declares a format the circuit
    /// has no edge or weight format for.
    #[error("quantization: {0}")]
    Quant(String),
    /// 2026-09-30: A serving-policy name is unknown.
    #[error(transparent)]
    Precision(#[from] PrecisionError),
    /// 2026-09-30: The circuit does not instantiate under the mapped shape.
    #[error(transparent)]
    Circuit(#[from] CircuitError),
}

mod states;
pub use states::{CacheStates, cache_states, recurrent_states};

/// 2026-09-30: One architecture's embedded files.
struct Arch {
    circuit: &'static str,
    config_map: &'static str,
}

macro_rules! circuits_file {
    ($f:literal) => {
        include_str!(concat!("../../../kernels/circuits/", $f))
    };
}

const ARCHES: [Arch; 4] = [
    Arch {
        circuit: circuits_file!("qwen3_5.toml"),
        config_map: circuits_file!("qwen3_5.config.toml"),
    },
    Arch {
        circuit: circuits_file!("qwen3_6_moe.toml"),
        config_map: circuits_file!("qwen3_6_moe.config.toml"),
    },
    Arch {
        circuit: circuits_file!("nemotron_h.toml"),
        config_map: circuits_file!("nemotron_h.config.toml"),
    },
    Arch {
        circuit: circuits_file!("dense_gqa.toml"),
        config_map: circuits_file!("dense_gqa.config.toml"),
    },
];

/// 2026-09-30: The block libraries the embedded circuits include.
const BLOCKS: [(&str, &str); 2] = [
    ("qwen3_hybrid", circuits_file!("blocks/qwen3_hybrid.toml")),
    ("transformer", circuits_file!("blocks/transformer.toml")),
];

/// 2026-09-30: The embedded config maps, parsed.
pub fn config_maps() -> Result<Vec<ConfigMap>, ConfigMapError> {
    ARCHES
        .iter()
        .map(|a| ConfigMap::parse(a.config_map))
        .collect()
}

/// 2026-09-30: The circuit of the checkpoint `config_json` describes, every edge at the
/// checkpoint's declared formats.
pub fn instantiate_from_checkpoint(
    config_json: &str,
    quant: QuantMetadata<'_>,
) -> Result<Circuit, CheckpointError> {
    Ok(resolve_checkpoint(config_json, quant, &ServePrecision::Declared)?.circuit)
}

/// 2026-09-30: The checkpoint's arch, shape, params, KV-cache format and circuit, with the
/// edge formats `serve` gives.
pub fn resolve_checkpoint(
    config_json: &str,
    quant: QuantMetadata<'_>,
    serve: &ServePrecision,
) -> Result<ResolvedCheckpoint, CheckpointError> {
    let config = parse_config(config_json)?;
    let (i, mapped) = map_checkpoint_value(&config)?;
    let qc = quantization_config(&config, quant)?;
    let plan = match &qc {
        Some(qc) => DeclaredPrecisionPlan::from_quantization_config(qc)
            .map_err(|e| CheckpointError::Quant(format!("{e:#}")))?,
        None => DeclaredPrecisionPlan::default(),
    };
    if let Some(qc) = &qc
        && plan.rules.is_empty()
    {
        return Err(CheckpointError::Quant(format!(
            "the quantization block declares no scheme for any layer (a missing quant group): {}",
            serde_json::to_string(qc)
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect::<String>()
        )));
    }
    let kv_cache = kv_cache_format(qc.as_ref())?;
    let text = ARCHES[i].circuit;
    let circuit = match serve {
        ServePrecision::Declared => {
            let precision = DeclaredPrecision::new(&plan);
            let circuit = instantiate(text, &BLOCKS, &mapped.shape, &precision)?;
            let refused = precision.refusals();
            if !refused.is_empty() {
                return Err(CheckpointError::Quant(refused.join("; ")));
            }
            circuit
        }
        ServePrecision::Policy { tier, caps, engine } => {
            let policy =
                metrale_config::WeightQuantPolicy::new(tier_named(tier)?, &plan, caps_named(caps)?);
            instantiate(
                text,
                &BLOCKS,
                &mapped.shape,
                &PolicyPrecision::new(policy, engine),
            )?
        }
    };
    Ok(ResolvedCheckpoint {
        arch: mapped.arch,
        model_type: mapped.model_type,
        shape: mapped.shape,
        params: mapped.params,
        kv_cache,
        circuit,
    })
}

/// 2026-09-30: `config.json` mapped to its circuit arch, arch shape and params, without
/// instantiating: the shape `INSTANCES.toml` must restate for a golden instance.
pub fn map_checkpoint(config_json: &str) -> Result<MappedConfig, CheckpointError> {
    Ok(map_checkpoint_value(&parse_config(config_json)?)?.1)
}

fn parse_config(config_json: &str) -> Result<Value, CheckpointError> {
    serde_json::from_str(config_json).map_err(|e| CheckpointError::Json {
        file: "config.json",
        detail: e.to_string(),
    })
}

/// 2026-09-30: The index of the arch whose map serves `config`'s model type, and the mapping.
fn map_checkpoint_value(config: &Value) -> Result<(usize, MappedConfig), CheckpointError> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let maps = config_maps()?;
    let Some((i, map)) = maps
        .iter()
        .enumerate()
        .find(|(_, m)| m.model_types().contains(&model_type))
    else {
        let mut served: Vec<&str> = maps
            .iter()
            .flat_map(|m| m.model_types().iter().map(String::as_str))
            .collect();
        served.sort_unstable();
        return Err(CheckpointError::UnknownModelType {
            model_type,
            served: served.join(", "),
        });
    };
    Ok((i, map_config(map, config)?))
}

/// 2026-09-30: The checkpoint's `quantization_config`: config.json's own, else the sidecar's.
fn quantization_config(
    config: &Value,
    quant: QuantMetadata<'_>,
) -> Result<Option<Value>, CheckpointError> {
    if let Some(qc) = config.get("quantization_config").filter(|v| !v.is_null()) {
        return Ok(Some(qc.clone()));
    }
    quant
        .hf_quant_config
        .map(|text| {
            serde_json::from_str(text).map_err(|e| CheckpointError::Json {
                file: "hf_quant_config.json",
                detail: e.to_string(),
            })
        })
        .transpose()
}

/// 2026-09-30: The KV-cache format a quantization block declares: compressed-tensors
/// `kv_cache_scheme`, or ModelOpt `kv_cache_quant_algo` (flat, or nested under `quantization`
/// in a sidecar).
fn kv_cache_format(qc: Option<&Value>) -> Result<Option<Format>, CheckpointError> {
    let Some(qc) = qc else {
        return Ok(None);
    };
    let fp8 = Format::Fp8E4m3 {
        scale: Scale::PerTensor,
    };
    if let Some(s) = qc.get("kv_cache_scheme").filter(|v| !v.is_null()) {
        let bits = s.get("num_bits").and_then(Value::as_u64);
        let ty = s.get("type").and_then(Value::as_str);
        return match (bits, ty) {
            (Some(8), Some("float")) => Ok(Some(fp8)),
            _ => Err(CheckpointError::Quant(format!(
                "kv_cache_scheme {s} has no KV-cache format in the circuit"
            ))),
        };
    }
    let algo = qc
        .get("kv_cache_quant_algo")
        .or_else(|| qc.get("quantization")?.get("kv_cache_quant_algo"))
        .filter(|v| !v.is_null());
    match algo.map(|a| a.as_str()) {
        None => Ok(None),
        Some(Some("FP8")) => Ok(Some(fp8)),
        Some(other) => Err(CheckpointError::Quant(format!(
            "kv_cache_quant_algo {other:?} has no KV-cache format in the circuit"
        ))),
    }
}
