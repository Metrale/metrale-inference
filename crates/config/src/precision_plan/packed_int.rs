// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: compressed-tensors `pack-quantized` integer weights: which declarations the
//! engine admits, and the one byte layout it admits them under.
//!
//! Owner: config (quantization).
//! Invariants:
//! - Pure: no environment, no I/O.
//! - [`admit_packed_int`] accepts only symmetric, static, group-128 INT4/INT8 weight-only
//!   groups in `pack-quantized` format with no zero points, no activation ordering and no
//!   online transform. Every other variant is an error naming the field, never a guess.
//! - [`PackedIntScheme::from_operand`] maps only that shape; a plan operand that is not one
//!   of the admitted schemes maps to `None`.
//!
//! Layout (compressed-tensors `pack_to_int32`, verified against the INT4/INT8 tensors of
//! `poolside/Laguna-XS-2.1-INT4` @ 4b7e28ab, see docs/laguna-xs-2.1-strix-int4.md):
//! - `weight_packed`: I32 `[N, K * bits / 32]`, row-major over the `[N, K]` weight. Word `j`
//!   of row `n` holds elements `k = j * (32 / bits) + i` for `i` in `0..32 / bits`, element
//!   `i` in bits `[bits * i, bits * (i + 1))` (least significant first).
//! - Each field is offset binary: the stored unsigned code `u` is `q + 2^(bits - 1)`, so
//!   `q = u - 8` for INT4 and `q = u - 128` for INT8 (not two's complement).
//! - `weight_scale`: `[N, K / group_size]` in the checkpoint's dtype (BF16 for Laguna);
//!   `w[n][k] = q * scale[n][k / group_size]`. No zero point tensor exists.
//!
//! Usage:
//! ```
//! use metrale_config::precision_plan::packed_int::{admit_packed_int, PackedIntScheme};
//! let qc = serde_json::json!({
//!     "quant_method": "compressed-tensors", "format": "pack-quantized",
//!     "quantization_status": "compressed",
//!     "config_groups": { "group_0": { "targets": ["Linear"], "format": "pack-quantized",
//!         "weights": { "num_bits": 4, "type": "int", "symmetric": true,
//!                      "strategy": "group", "group_size": 128 } } }
//! });
//! assert_eq!(admit_packed_int(&qc).unwrap(), vec![PackedIntScheme::INT4_G128]);
//! ```

use anyhow::{Result, bail, ensure};
use serde_json::Value;

use super::{Granularity, NumKind, Operand, ScaleTiming};

/// 2026-10-07: The only group size admitted. The HIP kernels hold one scale per 128-wide K
/// slice (kernels/strix-hip/laguna-xs-2.1/int4).
pub const PACKED_INT_GROUP_SIZE: u32 = 128;

/// 2026-10-07: One admitted packed-int weight scheme: symmetric, offset-binary codes of
/// `bits` bits packed least-significant-first into little-endian 32-bit words, one scale
/// per `group_size` consecutive K values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedIntScheme {
    /// 2026-10-07: Bits per weight code, 4 or 8.
    pub bits: u8,
    /// 2026-10-07: K values per scale.
    pub group_size: u32,
}

impl PackedIntScheme {
    /// 2026-10-07: INT4, group 128 (W4A16).
    pub const INT4_G128: PackedIntScheme = PackedIntScheme {
        bits: 4,
        group_size: PACKED_INT_GROUP_SIZE,
    };

    /// 2026-10-07: INT8, group 128 (W8A16).
    pub const INT8_G128: PackedIntScheme = PackedIntScheme {
        bits: 8,
        group_size: PACKED_INT_GROUP_SIZE,
    };

    /// 2026-10-07: Codes per packed 32-bit word (8 for INT4, 4 for INT8).
    pub fn codes_per_word(&self) -> usize {
        32 / self.bits as usize
    }

    /// 2026-10-07: The offset subtracted from a stored unsigned code (8 or 128).
    pub fn code_offset(&self) -> i32 {
        1 << (self.bits - 1)
    }

    /// 2026-10-07: The scheme a declared-precision operand stands for, or `None` when it is
    /// not an admitted packed-int weight (float, another width, another granularity, dynamic).
    pub fn from_operand(op: &Operand) -> Option<Self> {
        let admitted = op.kind == NumKind::Int
            && matches!(op.bits, 4 | 8)
            && op.granularity == Granularity::Group(PACKED_INT_GROUP_SIZE)
            && op.timing == ScaleTiming::Static;
        admitted.then_some(PackedIntScheme {
            bits: op.bits,
            group_size: PACKED_INT_GROUP_SIZE,
        })
    }
}

/// 2026-10-07: Fields a JSON value may leave unset: absent, null, `false` or `{}`.
fn unset(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => true,
        Some(Value::Object(o)) => o.is_empty(),
        _ => false,
    }
}

/// 2026-10-07: The integer weight schemes a compressed-tensors `quantization_config` (the
/// value, not the whole config.json) declares, in `config_groups` order, after refusing
/// every variant the engine cannot run. Empty when the block is another dialect or declares
/// no integer weights, so float (NVFP4, FP8) checkpoints pass through untouched.
pub fn admit_packed_int(qc: &Value) -> Result<Vec<PackedIntScheme>> {
    if qc.get("quant_method").and_then(Value::as_str) != Some("compressed-tensors") {
        return Ok(Vec::new());
    }
    let Some(groups) = qc.get("config_groups").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut schemes = Vec::new();
    let mut float_groups = Vec::new();
    for (name, group) in groups {
        let what = format!("config_groups.{name}");
        let Some(weights) = group.get("weights").filter(|w| !w.is_null()) else {
            continue;
        };
        // 2026-10-07: compressed-tensors' QuantizationArgs.type defaults to "int".
        match weights.get("type").and_then(Value::as_str).unwrap_or("int") {
            "int" => schemes.push(admit_group(qc, group, weights, &what)?),
            _ => float_groups.push(what),
        }
    }
    ensure!(
        schemes.is_empty() || float_groups.is_empty(),
        "quantization_config mixes integer and float weight groups ({}); no kernel set serves both",
        float_groups.join(", ")
    );
    if !schemes.is_empty() {
        admit_block(qc)?;
    }
    Ok(schemes)
}

/// 2026-10-07: One integer weight group.
fn admit_group(qc: &Value, group: &Value, w: &Value, what: &str) -> Result<PackedIntScheme> {
    let format = group
        .get("format")
        .filter(|f| !f.is_null())
        .or_else(|| qc.get("format"))
        .and_then(Value::as_str);
    ensure!(
        format == Some("pack-quantized"),
        "{what}.format {format:?}: only \"pack-quantized\" integer weights are supported"
    );
    let bits = w.get("num_bits").and_then(Value::as_u64);
    let bits = match bits {
        Some(b @ (4 | 8)) => b as u8,
        other => bail!("{what}.weights.num_bits {other:?}: only 4 and 8 are supported"),
    };
    // 2026-10-07: QuantizationArgs.symmetric defaults to true; false means zero points.
    ensure!(
        w.get("symmetric").is_none_or(|s| s == &Value::Bool(true)),
        "{what}.weights.symmetric {}: asymmetric integer weights (zero points) are not supported",
        w["symmetric"]
    );
    ensure!(
        unset(w.get("zp_dtype")),
        "{what}.weights.zp_dtype {}: zero points are not supported",
        w["zp_dtype"]
    );
    ensure!(
        unset(w.get("actorder")),
        "{what}.weights.actorder {}: activation-ordered groups (g_idx) are not supported",
        w["actorder"]
    );
    let strategy = w.get("strategy").and_then(Value::as_str);
    ensure!(
        strategy == Some("group"),
        "{what}.weights.strategy {strategy:?}: only \"group\" is supported"
    );
    let group_size = w.get("group_size").and_then(Value::as_u64);
    ensure!(
        group_size == Some(PACKED_INT_GROUP_SIZE as u64),
        "{what}.weights.group_size {group_size:?}: only {PACKED_INT_GROUP_SIZE} is supported"
    );
    ensure!(
        unset(w.get("block_structure")),
        "{what}.weights.block_structure {}: block scales are not supported",
        w["block_structure"]
    );
    ensure!(
        unset(w.get("dynamic")),
        "{what}.weights.dynamic {}: weight scales must be static",
        w["dynamic"]
    );
    ensure!(
        unset(w.get("scale_dtype")),
        "{what}.weights.scale_dtype {}: only scales stored in the checkpoint dtype are supported",
        w["scale_dtype"]
    );
    for key in ["input_activations", "output_activations"] {
        ensure!(
            unset(group.get(key)),
            "{what}.{key} {}: only weight-only (A16) integer groups are supported",
            group[key]
        );
    }
    Ok(PackedIntScheme {
        bits,
        group_size: PACKED_INT_GROUP_SIZE,
    })
}

/// 2026-10-07: Block-level fields that change what the stored integer tensors mean.
fn admit_block(qc: &Value) -> Result<()> {
    let status = qc.get("quantization_status").and_then(Value::as_str);
    ensure!(
        status == Some("compressed"),
        "quantization_status {status:?}: integer weights must be stored \"compressed\" (packed)"
    );
    ensure!(
        unset(qc.get("sparsity_config")),
        "sparsity_config {}: sparse integer weights are not supported",
        qc["sparsity_config"]
    );
    // 2026-10-07: A transform at `weight_input` / `weight_output` is fused into the stored
    // weights before quantization (the Laguna INT4 tensors match the BF16 release only after
    // that rotation), so the runtime runs the checkpoint as written. Any other location is an
    // online transform the runtime would have to apply, and is refused.
    let Some(groups) = qc.get("transform_config").filter(|t| !unset(Some(t))) else {
        return Ok(());
    };
    let Some(groups) = groups.get("config_groups").and_then(Value::as_object) else {
        bail!("transform_config without config_groups is not understood");
    };
    for (name, scheme) in groups {
        let Some(applies) = scheme.get("apply").and_then(Value::as_array) else {
            bail!("transform_config.config_groups.{name}.apply must be a list");
        };
        for (i, apply) in applies.iter().enumerate() {
            let location = apply.get("location").and_then(Value::as_str);
            ensure!(
                matches!(location, Some("weight_input" | "weight_output")),
                "transform_config.config_groups.{name}.apply[{i}].location {location:?}: only \
                 transforms fused into the weights (weight_input, weight_output) are supported"
            );
        }
    }
    Ok(())
}

/// 2026-10-07: The one scheme every routed expert projection under `mlp_prefix` declares
/// (`<mlp_prefix>.experts.<e>.{gate,up,down}_proj`, `e` in `0..num_experts`), or `None` when
/// all of them are unquantized. A layer whose experts disagree, declare activations, or
/// declare a weight that is not an admitted packed-int scheme is refused: one grouped expert
/// kernel serves the whole layer.
pub fn routed_expert_scheme(
    plan: &super::DeclaredPrecisionPlan,
    mlp_prefix: &str,
    num_experts: usize,
) -> Result<Option<PackedIntScheme>> {
    let mut seen: Option<Option<PackedIntScheme>> = None;
    for e in 0..num_experts {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let module = format!("{mlp_prefix}.experts.{e}.{proj}");
            let p = plan.resolve(&module);
            ensure!(
                p.activation.is_none(),
                "{module} declares {}: only weight-only experts are supported",
                p.label()
            );
            let scheme = match p.weight {
                None => None,
                Some(w) => Some(PackedIntScheme::from_operand(&w).ok_or_else(|| {
                    anyhow::anyhow!("{module} declares {w:?}, not a packed INT4/INT8 g128 weight")
                })?),
            };
            match seen {
                None => seen = Some(scheme),
                Some(first) => ensure!(
                    first == scheme,
                    "{module} declares {scheme:?} but {mlp_prefix}.experts.0.gate_proj declares \
                     {first:?}; one layer's experts must share a scheme"
                ),
            }
        }
    }
    Ok(seen.flatten())
}

#[cfg(test)]
#[path = "packed_int_tests.rs"]
mod tests;
