// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `quantization_config` dialects into a [`DeclaredPrecisionPlan`].
//!
//! Owner: config (quantization).
//! Invariants:
//! - A field the plan interprets must have the documented type; anything else is an error
//!   naming the field. Fields the plan does not interpret are left alone.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use super::{
    DeclaredPrecisionPlan, Granularity, LayerPrecision, NumKind, Operand, PlanSource, Rule,
    ScaleTiming, Target,
};

pub(super) fn plan(qc: &Value) -> Result<DeclaredPrecisionPlan> {
    // 2026-09-28: A ModelOpt `hf_quant_config.json` nests its fields under `quantization`.
    let qc = match qc.get("quantization") {
        Some(inner @ Value::Object(_)) => inner,
        _ => qc,
    };
    ensure!(
        qc.is_object(),
        "quantization_config must be an object, got {qc}"
    );
    let method = str_field(qc, "quant_method")?.unwrap_or_default();
    let producer_modelopt = qc
        .get("producer")
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .is_some_and(|n| n.eq_ignore_ascii_case("modelopt"));
    match method.as_str() {
        "compressed-tensors" => compressed_tensors(qc),
        "fp8" => fp8(qc),
        "modelopt" => modelopt(qc),
        "" if producer_modelopt || qc.get("quant_algo").is_some() => modelopt(qc),
        _ => Ok(DeclaredPrecisionPlan::default()),
    }
}

fn str_field(v: &Value, key: &str) -> Result<Option<String>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => bail!("quantization_config.{key} must be a string, got {other}"),
    }
}

fn str_list(v: &Value, key: &str) -> Result<Vec<String>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|e| {
                e.as_str().map(str::to_string).with_context(|| {
                    format!("quantization_config.{key}: entry {e} is not a string")
                })
            })
            .collect(),
        Some(other) => bail!("quantization_config.{key} must be a list, got {other}"),
    }
}

/// 2026-09-28: One compressed-tensors `QuantizationArgs` object, or `None` for null/absent/{}.
fn operand(v: Option<&Value>, what: &str) -> Result<Option<Operand>> {
    let v = match v {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(o)) if o.is_empty() => return Ok(None),
        Some(v @ Value::Object(_)) => v,
        Some(other) => bail!("{what} must be an object or null, got {other}"),
    };
    let bits = v
        .get("num_bits")
        .and_then(Value::as_u64)
        .filter(|b| (1..=16).contains(b))
        .with_context(|| format!("{what}.num_bits must be an integer in 1..=16"))?;
    // 2026-09-28: compressed-tensors' `QuantizationArgs.type` defaults to "int".
    let kind = match str_field(v, "type")?.as_deref() {
        Some("float") => NumKind::Float,
        Some("int") | None => NumKind::Int,
        Some(other) => bail!("{what}.type {other:?} is neither \"float\" nor \"int\""),
    };
    let group = match v.get("group_size") {
        None | Some(Value::Null) => None,
        Some(g) => Some(
            g.as_u64()
                .filter(|g| *g > 0 && *g <= u32::MAX as u64)
                .with_context(|| format!("{what}.group_size must be a positive integer"))?
                as u32,
        ),
    };
    let block = match v.get("block_structure") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) if a.len() == 2 => {
            let d = |i: usize| a[i].as_u64().filter(|x| *x > 0).map(|x| x as u32);
            Some((d(0), d(1)))
        }
        Some(other) => bail!("{what}.block_structure must be a two-element list, got {other}"),
    };
    let granularity = match (str_field(v, "strategy")?.as_deref(), group, block) {
        (Some("tensor"), ..) => Granularity::Tensor,
        (Some("channel"), ..) => Granularity::Channel,
        (Some("token"), ..) => Granularity::Token,
        (Some("group"), Some(g), _) => Granularity::Group(g),
        (Some("tensor_group"), Some(g), _) => Granularity::TensorGroup(g),
        (Some("block"), _, Some((Some(r), Some(c)))) => Granularity::Block(r, c),
        (Some(s @ ("group" | "tensor_group" | "block")), ..) => {
            bail!("{what}.strategy {s:?} needs its size (group_size / block_structure)")
        }
        (Some(other), ..) => bail!("{what}.strategy {other:?} is not a known strategy"),
        // 2026-09-28: ModelOpt omits `strategy`; a group size is then a per-tensor-scaled
        // group (NVFP4).
        (None, Some(g), _) => Granularity::TensorGroup(g),
        (None, None, _) => Granularity::Unstated,
    };
    let timing = match v.get("dynamic") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => ScaleTiming::Static,
        Some(Value::Bool(true)) => ScaleTiming::Dynamic,
        Some(Value::String(s)) if s == "local" => ScaleTiming::Local,
        Some(other) => bail!("{what}.dynamic must be a bool or \"local\", got {other}"),
    };
    Ok(Some(Operand {
        kind,
        bits: bits as u8,
        granularity,
        timing,
    }))
}

/// 2026-09-28: `config_groups` (compressed-tensors; ModelOpt writes the same shape).
fn config_groups(qc: &Value, target: fn(&str) -> Result<Target>) -> Result<Vec<Rule>> {
    let groups = match qc.get("config_groups") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(g)) => g,
        Some(other) => bail!("quantization_config.config_groups must be an object, got {other}"),
    };
    let mut rules = Vec::with_capacity(groups.len());
    for (name, g) in groups {
        let what = format!("config_groups.{name}");
        ensure!(g.is_object(), "{what} must be an object");
        let targets = str_list(g, "targets")
            .with_context(|| format!("{what}.targets"))?
            .iter()
            .map(|t| target(t))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!targets.is_empty(), "{what}.targets is empty");
        let weight = operand(g.get("weights"), &format!("{what}.weights"))?;
        ensure!(weight.is_some(), "{what} declares no weights scheme");
        let activation = operand(
            g.get("input_activations"),
            &format!("{what}.input_activations"),
        )?;
        rules.push(Rule {
            targets,
            precision: LayerPrecision { weight, activation },
        });
    }
    Ok(rules)
}

fn compressed_tensors(qc: &Value) -> Result<DeclaredPrecisionPlan> {
    let rules = config_groups(qc, Target::compressed_tensors)?;
    let ignore = str_list(qc, "ignore")?
        .iter()
        .map(|s| Target::compressed_tensors(s))
        .collect::<Result<Vec<_>>>()?;
    let source = if rules.is_empty() {
        PlanSource::Undeclared
    } else {
        PlanSource::CompressedTensors
    };
    Ok(DeclaredPrecisionPlan {
        source,
        rules,
        ignore,
    })
}

/// 2026-09-28: A ModelOpt `quant_algo`. `None` for an algorithm the plan does not know,
/// which leaves the module undeclared rather than guessed.
fn modelopt_algo(algo: &str, group: Option<u32>) -> Option<LayerPrecision> {
    // 2026-09-28: NVFP4 is defined with 16-element groups; ModelOpt omits the size from a
    // global `quant_algo` block.
    let g = group.unwrap_or(16);
    let fp4 = Operand {
        granularity: Granularity::TensorGroup(g),
        ..Operand::NVFP4
    };
    let fp8 = Operand {
        kind: NumKind::Float,
        bits: 8,
        granularity: Granularity::Tensor,
        timing: ScaleTiming::Static,
    };
    Some(match algo {
        "NVFP4" => LayerPrecision {
            weight: Some(fp4),
            activation: Some(Operand {
                timing: ScaleTiming::Local,
                ..fp4
            }),
        },
        "W4A16_NVFP4" => LayerPrecision {
            weight: Some(fp4),
            activation: None,
        },
        "FP8" => LayerPrecision {
            weight: Some(fp8),
            activation: Some(fp8),
        },
        _ => return None,
    })
}

fn modelopt(qc: &Value) -> Result<DeclaredPrecisionPlan> {
    let mut rules = Vec::new();
    // 2026-09-28: `quantized_layers` names every layer with its algorithm and is what the
    // ModelOpt loaders read, so it outranks `config_groups` (exact names are tier 0).
    // nvidia/Qwen3.6-35B-A3B-NVFP4 is the case where they disagree: its `config_groups`
    // lists FP4 input activations and its `quantized_layers` says `W4A16_NVFP4`.
    match qc.get("quantized_layers") {
        None | Some(Value::Null) => {}
        Some(Value::Object(layers)) => {
            for (name, spec) in layers {
                let algo = str_field(spec, "quant_algo")
                    .with_context(|| format!("quantized_layers.{name}"))?
                    .with_context(|| format!("quantized_layers.{name} has no quant_algo"))?;
                let group = spec
                    .get("group_size")
                    .and_then(Value::as_u64)
                    .map(|g| g as u32);
                if let Some(precision) = modelopt_algo(&algo, group) {
                    rules.push(Rule {
                        targets: vec![Target::Exact(name.clone())],
                        precision,
                    });
                }
            }
        }
        Some(other) => bail!("quantization_config.quantized_layers must be an object, got {other}"),
    }
    rules.extend(config_groups(qc, |s| Ok(Target::modelopt(s)))?);
    let algo = str_field(qc, "quant_algo")?.unwrap_or_default();
    if rules.is_empty() {
        let group = qc
            .get("group_size")
            .and_then(Value::as_u64)
            .map(|g| g as u32);
        match (algo.as_str(), modelopt_algo(&algo, group)) {
            ("MIXED_PRECISION", _) => {
                bail!("quant_algo MIXED_PRECISION without quantized_layers or config_groups")
            }
            (_, Some(precision)) => rules.push(Rule {
                targets: vec![Target::Class("Linear".into())],
                precision,
            }),
            (_, None) => {}
        }
    }
    let mut ignore: Vec<Target> = Vec::new();
    for key in ["ignore", "exclude_modules"] {
        for s in str_list(qc, key)? {
            if !ignore.iter().any(|t| t.text() == s) {
                ignore.push(Target::modelopt(&s));
            }
        }
    }
    let source = if rules.is_empty() {
        PlanSource::Undeclared
    } else {
        PlanSource::ModelOpt
    };
    Ok(DeclaredPrecisionPlan {
        source,
        rules,
        ignore,
    })
}

/// 2026-09-28: The HF `fp8` method: E4M3 weights (block-scaled with `weight_block_size`,
/// else per tensor) and E4M3 activations, dynamic per token group of the block width, or
/// static per tensor.
fn fp8(qc: &Value) -> Result<DeclaredPrecisionPlan> {
    let block = match qc.get("weight_block_size") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a))
            if a.len() == 2 && a.iter().all(|x| x.as_u64().is_some_and(|x| x > 0)) =>
        {
            Some((
                a[0].as_u64().unwrap_or(0) as u32,
                a[1].as_u64().unwrap_or(0) as u32,
            ))
        }
        Some(other) => bail!(
            "quantization_config.weight_block_size must be two positive integers, got {other}"
        ),
    };
    let per_tensor = |key: &str| qc.get(key).and_then(Value::as_bool).unwrap_or(false);
    let e4m3 = |granularity, timing| Operand {
        kind: NumKind::Float,
        bits: 8,
        granularity,
        timing,
    };
    let weight = e4m3(
        match block {
            Some((r, c)) if !per_tensor("weight_per_tensor") => Granularity::Block(r, c),
            _ => Granularity::Tensor,
        },
        ScaleTiming::Static,
    );
    let activation = match str_field(qc, "activation_scheme")?.as_deref() {
        Some("dynamic") | None => e4m3(
            match block {
                Some((_, c)) if !per_tensor("act_per_tensor") => Granularity::Group(c),
                _ if per_tensor("act_per_tensor") => Granularity::Tensor,
                _ => Granularity::Token,
            },
            ScaleTiming::Dynamic,
        ),
        Some("static") => e4m3(Granularity::Tensor, ScaleTiming::Static),
        Some(other) => {
            bail!("quantization_config.activation_scheme {other:?} is neither dynamic nor static")
        }
    };
    let mut ignore = Vec::new();
    for key in ["modules_to_not_convert", "ignored_layers"] {
        for entry in str_list(qc, key)? {
            ignore.push(Target::hf_module(&entry)?);
        }
    }
    Ok(DeclaredPrecisionPlan {
        source: PlanSource::Fp8,
        rules: vec![Rule {
            targets: vec![Target::Class("Linear".into())],
            precision: LayerPrecision {
                weight: Some(weight),
                activation: Some(activation),
            },
        }],
        ignore,
    })
}
