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
//! A device's instruction is not enough: a node runs on the FP4 MMA only where the device's class
//! compiles an FP4 block-scale kernel for it ([`super::avail::without_fp4_kernel`]); elsewhere it
//! takes the same path as on a device without that MMA. [`node_exec`] is the one answer per node:
//! the report's execution columns, its "FP4 costing" row and every node's cost read it.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: no silent upcast. An activation format with no native or exact path is
//! [`Exec::NoPath`], never re-planned at a wider format.

use super::device::{Device, MmaKind};
use super::estimate::activation_of;
use crate::format::Format;
use crate::ir::{Circuit, Node};
use crate::venn::families::Roofline;

/// 2026-09-30: The execution of one node's declared formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Exec {
    /// 2026-09-30: The device's MMA of this kind runs it (weights narrower than the activation
    /// widen in registers: W4A16, W8A16).
    Native(MmaKind),
    /// 2026-10-06: Storage understood, but no verified E8M0/group32 circuit lowering.
    UnsupportedMxfp4,
    /// 2026-09-30: NVFP4 operands converted exactly to E4M3 and run on the FP8 MMA.
    ExactFp8Emulation,
    /// 2026-09-30: No MMA of the device can run the activation format.
    NoPath(MmaKind),
    /// 2026-10-01: The device has the FP4 block-scale MMA but its class compiles no FP4 kernel
    /// for the node: the exact E2M1 -> E4M3 path on the FP8 MMA where the device has one
    /// (`fp8`), else no path.
    NoFp4Kernel {
        /// 2026-10-01: The device has the FP8 MMA.
        fp8: bool,
    },
}

impl Exec {
    /// 2026-09-30: The report spelling.
    pub fn describe(self) -> String {
        match self {
            Exec::UnsupportedMxfp4 => "no path: MXFP4 E8M0/group32 circuit lowering is not implemented".into(),
            Exec::Native(k) => format!("native {}", k.name()),
            Exec::ExactFp8Emulation => {
                "exact E2M1->E4M3 on the FP8 MMA, group-16 scales in FP32 (no native MMA for the pair)".into()
            }
            Exec::NoPath(k) => format!("no path: the device has no {} MMA", k.name()),
            Exec::NoFp4Kernel { fp8: true } => {
                "no FP4 block-scale kernel compiled for this class: exact E2M1->E4M3 on the FP8 MMA, group-16 scales in FP32".into()
            }
            Exec::NoFp4Kernel { fp8: false } => {
                "no path: no FP4 block-scale kernel compiled for this class and no FP8 MMA".into()
            }
        }
    }

    /// 2026-10-01: The tensor peak of `r` this execution runs at, and its name.
    pub fn peak(self, r: &Roofline) -> (f64, &'static str) {
        match self {
            Exec::Native(MmaKind::Fp4BlockScale) => (r.nvfp4_tflops, "the NVFP4 peak"),
            Exec::ExactFp8Emulation | Exec::NoFp4Kernel { fp8: true } => {
                (r.fp8_tflops, "the FP8 peak (exact E2M1->E4M3)")
            }
            Exec::Native(MmaKind::Fp8 | MmaKind::Fp4Fp8Nvfp4) => (r.fp8_tflops, "the FP8 peak"),
            Exec::Native(_)
            | Exec::NoPath(_)
            | Exec::NoFp4Kernel { fp8: false }
            | Exec::UnsupportedMxfp4 => (r.bf16_tflops, "the BF16 peak"),
        }
    }
}

/// 2026-10-01: How `device` runs an NVFP4 activation without an FP4 block-scale kernel: on the
/// FP8 MMA through the exact conversion where it has one, else no path; [`Exec::NoFp4Kernel`]
/// when the device has the FP4 MMA and only the kernel is missing.
pub fn fp4_fallback(device: &Device) -> Exec {
    let fp8 = device.runs(MmaKind::Fp8);
    if device.runs(MmaKind::Fp4BlockScale) {
        Exec::NoFp4Kernel { fp8 }
    } else if fp8 {
        Exec::ExactFp8Emulation
    } else {
        Exec::NoPath(MmaKind::Fp4BlockScale)
    }
}

/// 2026-10-01: How node `n` of `c` runs on `device` (weight-reading nodes only): its declared
/// formats on the device's instructions ([`exec_of`]), or [`fp4_fallback`] when the class
/// compiles no FP4 block-scale kernel for it (`no_fp4_kernel`).
pub fn node_exec(device: &Device, c: &Circuit, n: &Node, no_fp4_kernel: bool) -> Option<Exec> {
    let (w, a) = (n.weight?, activation_of(c, n)?);
    Some(
        if matches!(w, Format::Mxfp4) || matches!(a, Format::Mxfp4) {
            Exec::UnsupportedMxfp4
        } else if no_fp4_kernel {
            fp4_fallback(device)
        } else {
            exec_of(device, w, a)
        },
    )
}

/// 2026-09-30: The MMA kind the activation format needs.
pub fn kind_of(activation: Format) -> MmaKind {
    match activation {
        Format::Fp8E4m3 { .. } => MmaKind::Fp8,
        Format::Nvfp4 { .. } => MmaKind::Fp4BlockScale,
        Format::Mxfp4 => MmaKind::Mxf8f6f4,
        Format::Bf16 | Format::F32 | Format::I32 => MmaKind::Bf16,
    }
}

/// 2026-09-30: How `device` executes a node of `weight` reading `activation`. NVFP4 weights
/// under FP8 activations (W4A8 with group-16 E4M3 scales) have no single-instruction MMA on any
/// device unless `native_mma` says so (`fp4_fp8_nvfp4_scaled`); they take the exact E4M3 path.
pub fn exec_of(device: &Device, weight: Format, activation: Format) -> Exec {
    if matches!(weight, Format::Mxfp4) || matches!(activation, Format::Mxfp4) {
        return Exec::UnsupportedMxfp4;
    }
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
    } else if kind == MmaKind::Fp4BlockScale {
        fp4_fallback(device)
    } else {
        Exec::NoPath(kind)
    }
}

/// 2026-09-30: The declared pair as `W4A16`-style text.
pub fn pair_name(weight: Format, activation: Format) -> String {
    let bits = |f: Format| match f {
        Format::Nvfp4 { .. } | Format::Mxfp4 => "4",
        Format::Fp8E4m3 { .. } => "8",
        Format::Bf16 | Format::F32 | Format::I32 => "16",
    };
    format!("W{}A{}", bits(weight), bits(activation))
}
