// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `--dsa-quantization`: the precision GLM-5.3's DSA query-down (`q_a_proj`),
//! absorbed query (`q_absorb`) and absorbed output (`o_absorb`) projections are served at,
//! independently of `--dense-quantization` (owner decision B3). `declared` (the default) serves
//! them at the checkpoint's 16-bit width (BF16; the absorbed products are formed at load in FP32
//! and kept BF16). `w8a8` quantizes them at load to FP8 E4M3 with one F32 scale per output
//! channel and decodes them with dynamic per-token FP8 activations, what the reference engine
//! executes there. `w4a16` quantizes them to NVFP4 and decodes them with 16-bit activations
//! (the former `METRALE_GLM_DSA_W4A16=1`). Both are below declared and disclosed in the boot log.
//! The GLM loader moves exactly these three projections (`dense_tiers.rs`); the knob covers
//! the absorbed sites. 2026-10-15: under `METRALE_GLM_MLA_UNABSORB=declared` the unabsorbed
//! prefill's `q_b_proj` and `o_proj` follow the checkpoint-declared precision (BF16) whatever
//! this tier is; under `=dsa` the loader moves them too (the DSA site tier table,
//! `glm5next_dsa::unabsorb::lever::SITE_RULES`, says which as data).
//!
//! Owner: model-layers (quantization dispatch).
//! Invariants:
//! - The first publication or read wins (`OnceLock`); the serve publishes before the model is
//!   built, and anything that reads first fixes it at `declared`.

use std::sync::OnceLock;

/// 2026-10-11: The DSA projection precision tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DsaQuantization {
    /// 2026-10-11: The checkpoint's declared width (BF16).
    #[default]
    Declared,
    /// 2026-10-11: FP8 per-channel weights, per-token FP8 activations; below declared.
    W8a8,
    /// 2026-10-11: NVFP4 weights, 16-bit activations; below `W8a8`.
    W4a16,
}

impl DsaQuantization {
    /// 2026-10-11: Every tier, in the order the flag lists them.
    pub const ALL: [Self; 3] = [Self::Declared, Self::W8a8, Self::W4a16];

    /// 2026-10-11: The flag value, recipe value and record value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::W8a8 => "w8a8",
            Self::W4a16 => "w4a16",
        }
    }

    /// 2026-10-11: Whether this tier runs below the checkpoint's declared precision.
    pub fn below_declared(self) -> bool {
        self != Self::Declared
    }

    /// 2026-10-11: What the tier serves the three projections as, for the boot log.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Declared => "BF16, the checkpoint's",
            Self::W8a8 => "FP8 per-channel weights, per-token FP8 activations (W8A8)",
            Self::W4a16 => "NVFP4 weights, 16-bit activations (W4A16)",
        }
    }
}

static DSA_QUANTIZATION: OnceLock<DsaQuantization> = OnceLock::new();

/// 2026-10-11: Publish `--dsa-quantization`. Returns the tier in force; a caller that gets a
/// different one should warn.
pub fn set_dsa_quantization_from_cli(q: DsaQuantization) -> DsaQuantization {
    let _ = DSA_QUANTIZATION.set(q);
    *DSA_QUANTIZATION.get().expect("just set")
}

/// 2026-10-11: The tier in force: `declared` unless the serve published another.
pub fn dsa_quantization() -> DsaQuantization {
    *DSA_QUANTIZATION.get_or_init(DsaQuantization::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: The names are the flag's values, distinct and in `ALL` order; only
    /// `declared` is at the declared precision, and it is the default.
    #[test]
    fn names_order_and_the_declared_default() {
        let names: Vec<&str> = DsaQuantization::ALL.iter().map(|q| q.name()).collect();
        assert_eq!(names, ["declared", "w8a8", "w4a16"]);
        assert_eq!(DsaQuantization::default(), DsaQuantization::Declared);
        assert!(!DsaQuantization::Declared.below_declared());
        assert!(DsaQuantization::W8a8.below_declared());
        assert!(DsaQuantization::W4a16.below_declared());
    }
}
