// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `--dsa-quantization`: the precision GLM-5.3's DSA query-down (`q_a_proj`),
//! absorbed query (`q_absorb`) and absorbed output (`o_absorb`) projections are served at,
//! independently of `--dense-quantization` (owner decision B3). `declared` (the default) serves
//! them at the checkpoint's 16-bit width (BF16; the absorbed products are formed at load in FP32
//! and kept BF16). `w8a8` quantizes them at load to FP8 E4M3 with one F32 scale per output
//! channel and decodes them with dynamic per-token FP8 activations, what the reference engine
//! executes there. `w4a16` quantizes them to NVFP4 and decodes them with 16-bit activations
//! (the former `METRALE_GLM_DSA_W4A16=1`). Both are below declared and disclosed in the boot log.
//! The GLM loader moves exactly these three projections (`dense_tiers.rs`).
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

    /// 2026-10-14: The tier named `s` (its [`Self::name`]).
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|q| q.name() == s)
    }

    /// 2026-10-14: Whether `p` is a served phase pair: one tier in both phases, or W4A16 decode
    /// with W8A8 prefill (the three projections keep an FP8 copy for the prompt pass).
    pub fn phase_pair_ok(p: super::Phased<Self>) -> bool {
        !p.is_split() || (p.decode, p.prefill) == (Self::W4a16, Self::W8a8)
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
static DSA_PREFILL_QUANTIZATION: OnceLock<DsaQuantization> = OnceLock::new();

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

/// 2026-10-14: Publish the prompt pass's tier (`--dsa-quantization decode=X,prefill=Y`).
pub fn set_dsa_prefill_quantization_from_cli(q: DsaQuantization) -> DsaQuantization {
    let _ = DSA_PREFILL_QUANTIZATION.set(q);
    *DSA_PREFILL_QUANTIZATION.get().expect("just set")
}

/// 2026-10-14: The prompt pass's tier: [`dsa_quantization`] (decode's) unless the serve
/// published another.
pub fn dsa_prefill_quantization() -> DsaQuantization {
    *DSA_PREFILL_QUANTIZATION.get_or_init(dsa_quantization)
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

    /// 2026-10-14: Each tier parses back from its name and nothing else does; of the split
    /// pairs, only W4A16 decode over the FP8 prefill each flag's loader serves is accepted.
    #[test]
    fn names_parse_back_and_only_the_served_split_pairs_pass() {
        use crate::layers::{DenseQuantization as D, Phased};
        for q in DsaQuantization::ALL {
            assert_eq!(DsaQuantization::from_name(q.name()), Some(q));
        }
        for q in D::ALL {
            assert_eq!(D::from_name(q.name()), Some(q));
        }
        assert_eq!(DsaQuantization::from_name("fp8"), None);
        assert_eq!(D::from_name("w8a8"), None);
        let dsa_ok: Vec<_> = DsaQuantization::ALL
            .into_iter()
            .flat_map(|d| {
                DsaQuantization::ALL.map(|p| Phased {
                    decode: d,
                    prefill: p,
                })
            })
            .filter(|&p| p.is_split() && DsaQuantization::phase_pair_ok(p))
            .collect();
        assert_eq!(
            dsa_ok,
            [Phased {
                decode: DsaQuantization::W4a16,
                prefill: DsaQuantization::W8a8
            }]
        );
        let dense_ok: Vec<_> = D::ALL
            .into_iter()
            .flat_map(|d| {
                D::ALL.map(|p| Phased {
                    decode: d,
                    prefill: p,
                })
            })
            .filter(|&p| p.is_split() && D::phase_pair_ok(p))
            .collect();
        assert_eq!(
            dense_ok,
            [Phased {
                decode: D::W4a16,
                prefill: D::Fp8
            }]
        );
        for q in D::ALL {
            assert!(D::phase_pair_ok(Phased::both(q)));
        }
        for q in DsaQuantization::ALL {
            assert!(DsaQuantization::phase_pair_ok(Phased::both(q)));
        }
    }
}
