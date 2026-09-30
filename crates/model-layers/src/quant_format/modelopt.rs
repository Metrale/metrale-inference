// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The ModelOpt NVFP4 layout (`.weight`, `.weight_scale`,
//! `.weight_scale_2`, optional `.input_scale`), mapped to
//! [`Nvfp4Variant::Standard`].
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use crate::quant_format::{IgnoreList, QuantFormat};
use crate::weight_map::Nvfp4Variant;
use metrale_config::precision_plan::IgnoreDialect;

/// 2026-09-25: A ModelOpt checkpoint, and the fallback of
/// [`super::detect_quant_format`] when no quantized weights are found.
#[derive(Debug)]
pub struct ModeloptFormat {
    /// 2026-09-25: The config's `quant_algo`; nothing in this module reads it,
    /// and [`QuantFormat::base_variant`] is `Standard` whatever it says.
    pub algo: String,
    /// 2026-09-30: The modules that ship unquantized, matched as the format specifies.
    pub ignore: IgnoreList,
}

impl ModeloptFormat {
    /// 2026-09-30: A malformed ignore entry is refused.
    pub fn new(algo: String, ignore_modules: &[String]) -> anyhow::Result<Self> {
        Ok(Self {
            algo,
            ignore: IgnoreList::new(IgnoreDialect::ModelOpt, ignore_modules)?,
        })
    }
}

impl QuantFormat for ModeloptFormat {
    fn name(&self) -> &'static str {
        "modelopt"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        Nvfp4Variant::Standard
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore.matches(module_path)
    }
}
