// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--dense-quantization`: the precision a checkpoint's 16-bit (unquantized) dense
//! projections are served at. `declared` (the default) serves them at their checkpoint width.
//! `fp8` quantizes them at load to FP8 E4M3 with one F32 scale per output channel and decodes
//! them W8A8 with dynamic per-token FP8 activations (`ops::w8a8_decode`, per-row layout):
//! BELOW the checkpoint's declared precision, opt-in, and disclosed in the boot log and on
//! records. Which projections it covers is the model loader's decision (GLM-5.3:
//! `glm5next_fp8_dense`).
//!
//! Owner: model-layers (quantization dispatch).
//! Invariants:
//! - The first publication or read wins (`OnceLock`); the serve publishes before the model is
//!   built, and anything that reads first fixes it at `declared`.

use std::sync::OnceLock;

/// 2026-10-09: The dense-projection precision tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DenseQuantization {
    /// 2026-10-09: The checkpoint's declared width (16-bit projections stay 16-bit).
    #[default]
    Declared,
    /// 2026-10-09: 16-bit dense projections served W8A8 FP8 (per-channel weights, per-token
    /// activations), below declared.
    Fp8,
}

impl DenseQuantization {
    /// 2026-10-09: Every tier, in the order the flag lists them.
    pub const ALL: [Self; 2] = [Self::Declared, Self::Fp8];

    /// 2026-10-09: The flag value, recipe value and record value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Fp8 => "fp8",
        }
    }

    /// 2026-10-09: Whether this tier runs below the checkpoint's declared precision.
    pub fn below_declared(self) -> bool {
        self != Self::Declared
    }
}

static DENSE_QUANTIZATION: OnceLock<DenseQuantization> = OnceLock::new();

/// 2026-10-09: Publish `--dense-quantization`. Returns the tier in force; a caller that gets a
/// different one should warn.
pub fn set_dense_quantization_from_cli(q: DenseQuantization) -> DenseQuantization {
    let _ = DENSE_QUANTIZATION.set(q);
    *DENSE_QUANTIZATION.get().expect("just set")
}

/// 2026-10-09: The tier in force: `declared` unless the serve published another.
pub fn dense_quantization() -> DenseQuantization {
    *DENSE_QUANTIZATION.get_or_init(DenseQuantization::default)
}
