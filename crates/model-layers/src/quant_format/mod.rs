// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The quantized-weight layout of a checkpoint, and its ignore
//! list: [`CompressedTensorsFormat`], [`ModeloptFormat`] or
//! [`Fp8BlockScaledFormat`], chosen by [`detect_quant_format`].
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_config::precision_plan::{IgnoreDialect, Target};
use metrale_model_weights::weights::WeightStore;

use crate::weight_map::Nvfp4Variant;

mod compressed_tensors;
mod fp8_blockscaled;
mod modelopt;
mod mxfp4;
pub use mxfp4::Mxfp4Format;

pub use compressed_tensors::CompressedTensorsFormat;
pub use fp8_blockscaled::Fp8BlockScaledFormat;
pub use modelopt::ModeloptFormat;

/// 2026-09-25: A quantized-weight layout: its log name, the [`Nvfp4Variant`]
/// it maps to, and the module globs its ignore list keeps unquantized. The
/// weight loaders pick their variant through `weight_map::detect_nvfp4_variant`,
/// not through this trait.
pub trait QuantFormat: Send + Sync + std::fmt::Debug {
    /// 2026-09-25: Name for logs.
    fn name(&self) -> &'static str;

    /// 2026-10-07: Legacy mapping; native MXFP4 has no NVFP4 equivalent and returns None.
    fn base_variant(&self) -> Option<Nvfp4Variant>;

    /// 2026-09-30: Whether the ignore list names the module `module_path` (a module path,
    /// not a tensor name: no `.weight`), under the format's own matching ([`IgnoreList`]).
    fn is_ignored(&self, module_path: &str) -> bool;

    /// 2026-09-25: `Bf16Raw` for an ignored module, else [`Self::base_variant`].
    fn variant_for(&self, module_path: &str) -> Option<Nvfp4Variant> {
        if self.is_ignored(module_path) {
            Some(Nvfp4Variant::Bf16Raw)
        } else {
            self.base_variant()
        }
    }
}

/// 2026-09-25: Pick the [`QuantFormat`] for a checkpoint.
///
/// 1. A `quantization_config` whose `quant_method` is `modelopt`,
///    `compressed-tensors` or `fp8` selects that layout directly. The
///    `quant_algo` and `format` are logged or stored, not used to choose.
/// 2. Otherwise the layout comes from `weight_map::detect_nvfp4_variant`
///    (an unrecognised method is warned about first).
/// 3. When that finds no quantized weights (`Bf16Raw`), it warns and returns
///    a [`ModeloptFormat`].
///
/// The ignore list is the config's `ignore_modules`, or empty without a
/// config.
pub fn detect_quant_format(
    config: &ModelConfig,
    store: &WeightStore,
) -> anyhow::Result<Box<dyn QuantFormat>> {
    if let Some(qc) = &config.quantization_config {
        let method = qc.quant_method.as_str();
        let algo = qc.quant_algo.as_str();
        let format = qc.format.as_str();
        let ignore = qc.ignore_modules.clone();

        match method {
            "mxfp4" => return Ok(Box::new(Mxfp4Format::from_checkpoint(config, store)?)),
            "modelopt" => {
                tracing::info!(
                    "QuantFormat: modelopt (algo={algo:?}), {} ignored module(s)",
                    ignore.len(),
                );
                return Ok(Box::new(ModeloptFormat::new(algo.to_string(), &ignore)?));
            }
            "compressed-tensors" => {
                tracing::info!(
                    "QuantFormat: compressed-tensors (format={format:?}), {} ignored module(s)",
                    ignore.len(),
                );
                return Ok(Box::new(CompressedTensorsFormat::new(
                    format.to_string(),
                    &ignore,
                )?));
            }
            "fp8" => {
                tracing::info!(
                    "QuantFormat: fp8 (block-scaled), {} ignored module(s)",
                    ignore.len(),
                );
                return Ok(Box::new(Fp8BlockScaledFormat::new(&ignore)?));
            }
            other if !other.is_empty() => {
                tracing::warn!(
                    "QuantFormat: config declares unrecognized quant_method={other:?}; \
                     falling back to tensor-name heuristic. Metrale Engine currently understands \
                     {{compressed-tensors, modelopt, fp8}}. Checkpoint load may fail."
                );
            }
            _ => {
                // 2026-09-25: An empty `quant_method` falls through to the
                // tensor-name detection below.
            }
        }
    }

    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let ignore = config
        .quantization_config
        .as_ref()
        .map(|qc| qc.ignore_modules.clone())
        .unwrap_or_default();
    Ok(match variant {
        Nvfp4Variant::CompressedTensors => {
            tracing::info!("QuantFormat: compressed-tensors (detected from tensor names)");
            Box::new(CompressedTensorsFormat::new(String::new(), &ignore)?)
        }
        Nvfp4Variant::Fp8Dequanted => {
            tracing::info!("QuantFormat: fp8-blockscaled (detected from tensor names)");
            Box::new(Fp8BlockScaledFormat::new(&ignore)?)
        }
        Nvfp4Variant::Standard => {
            tracing::info!("QuantFormat: modelopt-style NVFP4 (detected from tensor names)");
            Box::new(ModeloptFormat::new(String::new(), &ignore)?)
        }
        Nvfp4Variant::Bf16Raw => {
            tracing::warn!(
                "QuantFormat: no quantization declared and no pre-quantized weights found; \
                 treating checkpoint as BF16 raw (weights will be runtime-quantized). \
                 Quality will be inferior to a calibrated NVFP4 release."
            );
            Box::new(ModeloptFormat::new(String::new(), &ignore)?) as Box<dyn QuantFormat>
        }
    })
}

/// 2026-09-30: A checkpoint's ignore list, parsed with the matching rules of its format
/// (`metrale_config::precision_plan::Target`, the matcher the declared precision plan uses):
/// compressed-tensors `re:` regexes or exact names, ModelOpt exact names or globs, HF `fp8`
/// module paths on whole dotted segments.
#[derive(Debug, Clone)]
pub struct IgnoreList(Vec<Target>);

impl IgnoreList {
    /// 2026-09-30: Parse `entries` under `dialect`; a malformed entry is refused.
    pub fn new(dialect: IgnoreDialect, entries: &[String]) -> anyhow::Result<Self> {
        entries
            .iter()
            .map(|e| Target::ignore_entry(dialect, e))
            .collect::<anyhow::Result<_>>()
            .map(Self)
    }

    /// 2026-09-30: Whether an entry names the module `module_path`.
    pub fn matches(&self, module_path: &str) -> bool {
        self.0.iter().any(|t| t.matches_name(module_path))
    }
}

#[cfg(test)]
#[path = "quant_format_tests.rs"]
mod tests;
