// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The DFlash drafter's weight-quantization tier (`--draft-weight-quantization`):
//! which copies of the drafter's BF16 GEMM weights `from_weights` builds, and in what format.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - The tier moves only the draft numerics: the target model is untouched, so a greedy verify
//!   still emits the target's tokens whatever the tier.
//! - Activations stay BF16 at every tier (weight-only quantization).
//! - `Fp8` is the drafter build's behaviour before the tier existed (its default).

/// 2026-10-10: The formats of the drafter's GEMM weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftWeightQuantization {
    /// 2026-10-10: The checkpoint's own weights (BF16 for every DFlash drafter so far): no
    /// copies.
    Declared,
    /// 2026-10-10: FP8 E4M3 copies, one f32 scale per row, of each layer's q/k/v/o/gate/up/down
    /// and of the LM head (or the target's native FP8 LM head); the ctx `fc` and fused K/V, the
    /// DFlash2 conv and selector projections stay BF16.
    Fp8,
    /// 2026-10-10: `Fp8`, plus FP8 copies of the ctx `fc` and fused K/V and of the DFlash2 conv
    /// and selector projections.
    Fp8Aux,
    /// 2026-10-10: NVFP4 copies (packed E2M1, one E4M3 scale per 16 weights, one f32 scale per
    /// tensor) of every weight `Fp8Aux` copies, run as W4A16 (`w4a16_tc_rows`). Each tensor's
    /// scale is the absolute maximum of the whole unsharded tensor, so a tensor-parallel rank's
    /// bytes are the one-rank drafter's.
    Nvfp4,
}

impl DraftWeightQuantization {
    /// 2026-10-10: Every tier, in the order `--help` lists them.
    pub const ALL: [Self; 4] = [Self::Declared, Self::Fp8, Self::Fp8Aux, Self::Nvfp4];

    /// 2026-10-10: The flag value naming this tier.
    pub fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Fp8 => "fp8",
            Self::Fp8Aux => "fp8-aux",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// 2026-10-10: The tier a flag value names; an unknown value is an error naming them all.
    pub fn parse(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|t| t.name() == s)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|t| t.name()).collect();
                format!("expected one of {}, got {s:?}", names.join(", "))
            })
    }

    /// 2026-10-10: Copies of the q/k/v/o/gate/up/down weights and the LM head exist.
    pub fn quantizes_layers(self) -> bool {
        !matches!(self, Self::Declared)
    }

    /// 2026-10-10: Copies of the ctx `fc` and fused K/V and the DFlash2 conv and selector
    /// projections exist.
    pub fn quantizes_aux(self) -> bool {
        matches!(self, Self::Fp8Aux | Self::Nvfp4)
    }

    /// 2026-10-10: The copies' bytes from what the FP8 copies of the same weights take:
    /// unchanged for the FP8 tiers, none for `Declared`, and for `Nvfp4` the 4-bit codes and
    /// one scale byte per 16 weights (9 / 16 of FP8), rounded up.
    pub fn copy_bytes_from_fp8(self, fp8_bytes: u64) -> u64 {
        match self {
            Self::Declared => 0,
            Self::Fp8 | Self::Fp8Aux => fp8_bytes,
            Self::Nvfp4 => (fp8_bytes * 9).div_ceil(16),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DraftWeightQuantization as Q;

    #[test]
    fn every_name_parses_back_to_its_tier() {
        for t in Q::ALL {
            assert_eq!(Q::parse(t.name()), Ok(t));
        }
    }

    #[test]
    fn unknown_or_differently_spelled_values_are_refused_with_the_list() {
        for bad in ["", "FP8", "fp4", "nvfp4 ", "bf16", "fp8_aux"] {
            let err = Q::parse(bad).unwrap_err();
            assert!(
                err.contains("declared, fp8, fp8-aux, nvfp4"),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn which_weights_each_tier_copies() {
        let got: Vec<(bool, bool)> = Q::ALL
            .iter()
            .map(|t| (t.quantizes_layers(), t.quantizes_aux()))
            .collect();
        assert_eq!(
            got,
            [(false, false), (true, false), (true, true), (true, true)]
        );
    }

    #[test]
    fn copy_bytes_scale_with_the_format() {
        let fp8 = 1_000_003;
        assert_eq!(Q::Declared.copy_bytes_from_fp8(fp8), 0);
        assert_eq!(Q::Fp8.copy_bytes_from_fp8(fp8), fp8);
        assert_eq!(Q::Fp8Aux.copy_bytes_from_fp8(fp8), fp8);
        // 2026-10-10: 1_000_003 * 9 / 16 = 562_501.69, rounded up.
        assert_eq!(Q::Nvfp4.copy_bytes_from_fp8(fp8), 562_502);
    }
}
