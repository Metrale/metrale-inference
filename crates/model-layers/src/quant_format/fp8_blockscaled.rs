// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The FP8 E4M3 block-scaled layout (`.weight` with
//! `.weight_scale_inv`), mapped to [`Nvfp4Variant::Fp8Dequanted`].
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use crate::quant_format::{IgnoreList, QuantFormat};
use crate::weight_map::Nvfp4Variant;
use metrale_config::precision_plan::IgnoreDialect;

/// 2026-09-25: An FP8 block-scaled checkpoint.
#[derive(Debug)]
pub struct Fp8BlockScaledFormat {
    /// 2026-09-30: The config's `ignore` / `exclude_modules` (`QuantizationConfig::ignore_modules`;
    /// an HF `modules_to_not_convert` list does not reach it). The FP8 checkpoints that carry
    /// one are ModelOpt exports (nvidia/DeepSeek-V4-Flash-NVFP4: `*.attn.*`, `mtp.*`), so it
    /// matches as ModelOpt does: exact names or globs.
    pub ignore: IgnoreList,
}

impl Fp8BlockScaledFormat {
    /// 2026-09-30: A malformed ignore entry is refused.
    pub fn new(ignore_modules: &[String]) -> anyhow::Result<Self> {
        Ok(Self {
            ignore: IgnoreList::new(IgnoreDialect::ModelOpt, ignore_modules)?,
        })
    }
}

impl QuantFormat for Fp8BlockScaledFormat {
    fn name(&self) -> &'static str {
        "fp8-blockscaled"
    }

    fn base_variant(&self) -> Option<Nvfp4Variant> {
        Some(Nvfp4Variant::Fp8Dequanted)
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore.matches(module_path)
    }
}
