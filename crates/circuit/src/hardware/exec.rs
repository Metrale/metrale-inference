// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: How a weight-reading node executes its DECLARED formats on a device. The declared
//! precision is never changed here; what varies per device is the instruction that carries it:
//! - the activation format picks the MMA kind (BF16 activations: BF16 MMA, any weight format
//!   widened in registers; FP8: the FP8 MMA; NVFP4: the block-scaled FP4 MMA);
//! - a device without the FP4 MMA but with the FP8 one runs NVFP4 operands through the exact
//!   E2M1 -> E4M3 conversion (`(q & 0x80808080) | ((q & 0x70707070) >> 2)` is value x 2^-6 for
//!   all 16 codes; the 2^6 folds into the group scale), block scales applied in FP32 per K=16
//!   step: the same products as the declared W4A4, not an upcast;
//! - anything else has no path, which the plan states.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: no silent upcast. An activation format with no native or exact path is
//! [`Exec::NoPath`], never re-planned at a wider format.

use super::device::{Device, MmaKind};
use crate::format::Format;

/// 2026-09-30: The execution of one node's declared formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Exec {
    /// 2026-09-30: The device's MMA of this kind runs it (weights narrower than the activation
    /// widen in registers: W4A16, W8A16).
    Native(MmaKind),
    /// 2026-09-30: NVFP4 operands converted exactly to E4M3 and run on the FP8 MMA.
    ExactFp8Emulation,
    /// 2026-09-30: No MMA of the device can run the activation format.
    NoPath(MmaKind),
}

impl Exec {
    /// 2026-09-30: The report spelling.
    pub fn describe(self) -> String {
        match self {
            Exec::Native(k) => format!("native {}", k.name()),
            Exec::ExactFp8Emulation => {
                "exact E2M1->E4M3 on the FP8 MMA, group-16 scales in FP32 (no native MMA for the pair)".into()
            }
            Exec::NoPath(k) => format!("no path: the device has no {} MMA", k.name()),
        }
    }
}

/// 2026-09-30: The MMA kind the activation format needs.
pub fn kind_of(activation: Format) -> MmaKind {
    match activation {
        Format::Fp8E4m3 { .. } => MmaKind::Fp8,
        Format::Nvfp4 { .. } => MmaKind::Fp4BlockScale,
        Format::Bf16 | Format::F32 | Format::I32 => MmaKind::Bf16,
    }
}

/// 2026-09-30: How `device` executes a node of `weight` reading `activation`. NVFP4 weights
/// under FP8 activations (W4A8 with group-16 E4M3 scales) have no single-instruction MMA on any
/// device unless `native_mma` says so (`fp4_fp8_nvfp4_scaled`); they take the exact E4M3 path.
pub fn exec_of(device: &Device, weight: Format, activation: Format) -> Exec {
    let kind = kind_of(activation);
    let w4a8 = matches!(weight, Format::Nvfp4 { .. }) && kind == MmaKind::Fp8;
    if w4a8 {
        return if device.runs(MmaKind::Fp4Fp8Nvfp4) {
            Exec::Native(MmaKind::Fp4Fp8Nvfp4)
        } else if device.runs(MmaKind::Fp8) {
            Exec::ExactFp8Emulation
        } else {
            Exec::NoPath(MmaKind::Fp8)
        };
    }
    if device.runs(kind) {
        Exec::Native(kind)
    } else if kind == MmaKind::Fp4BlockScale && device.runs(MmaKind::Fp8) {
        Exec::ExactFp8Emulation
    } else {
        Exec::NoPath(kind)
    }
}

/// 2026-09-30: The declared pair as `W4A16`-style text.
pub fn pair_name(weight: Format, activation: Format) -> String {
    let bits = |f: Format| match f {
        Format::Nvfp4 { .. } => "4",
        Format::Fp8E4m3 { .. } => "8",
        Format::Bf16 | Format::F32 | Format::I32 => "16",
    };
    format!("W{}A{}", bits(weight), bits(activation))
}
