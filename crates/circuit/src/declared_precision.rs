// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: [`DeclaredPrecision`], the [`EdgePrecision`] that gives every projection the
//! formats its checkpoint DECLARES (`DeclaredPrecisionPlan`), before any serving policy: the
//! weight as stored, and the input activation as the checkpoint declares it (FP8 per token,
//! per tensor with a static scale, NVFP4 group 16, or 16-bit).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A declared format the circuit has no edge format for (an integer scheme, an FP8 weight
//!   with an unstated granularity, ...) is recorded, never guessed; [`DeclaredPrecision::refusals`]
//!   lists them and the caller refuses the checkpoint.
//! - A module the plan does not cover is 16-bit (`LayerPrecision::UNQUANTIZED`).

use std::cell::RefCell;

use metrale_config::DeclaredPrecisionPlan;
use metrale_config::precision_plan::{Granularity, NumKind, Operand};

use crate::format::{Format, Scale};
use crate::precision::{EdgePrecision, LinearFormats};

/// 2026-09-30: The checkpoint's declared formats as an [`EdgePrecision`].
pub struct DeclaredPrecision<'a> {
    plan: &'a DeclaredPrecisionPlan,
    refusals: RefCell<Vec<String>>,
}

impl<'a> DeclaredPrecision<'a> {
    /// 2026-09-30: Answer from `plan`.
    pub fn new(plan: &'a DeclaredPrecisionPlan) -> Self {
        DeclaredPrecision {
            plan,
            refusals: RefCell::new(Vec::new()),
        }
    }

    /// 2026-09-30: Every module asked whose declaration has no circuit format, with why.
    pub fn refusals(&self) -> Vec<String> {
        self.refusals.borrow().clone()
    }

    fn refuse(&self, module: &str, what: &str, o: Operand) {
        self.refusals.borrow_mut().push(format!(
            "`{module}` declares a {what} the circuit has no format for: {o:?}"
        ));
    }
}

/// 2026-09-30: The weight format `o` declares, if the circuit has one.
fn weight_format(o: Operand) -> Option<Format> {
    if o.kind != NumKind::Float {
        return None;
    }
    match (o.bits, o.granularity) {
        (4, Granularity::TensorGroup(g) | Granularity::Group(g)) => {
            Some(Format::Nvfp4 { group: g })
        }
        (8, g) => Some(Format::Fp8E4m3 {
            scale: match g {
                Granularity::Tensor => Scale::PerTensor,
                Granularity::Channel => Scale::PerChannel,
                Granularity::Group(n) => Scale::Group(n),
                Granularity::Block(r, c) => Scale::Block(r, c),
                Granularity::Token | Granularity::TensorGroup(_) | Granularity::Unstated => {
                    return None;
                }
            },
        }),
        _ => None,
    }
}

/// 2026-09-30: The activation (edge) format `o` declares, if the circuit has one.
fn activation_format(o: Operand) -> Option<Format> {
    if o.kind != NumKind::Float {
        return None;
    }
    match (o.bits, o.granularity) {
        (4, Granularity::TensorGroup(g) | Granularity::Group(g)) => {
            Some(Format::Nvfp4 { group: g })
        }
        (8, Granularity::Tensor) => Some(Format::Fp8E4m3 {
            scale: Scale::PerTensor,
        }),
        (8, Granularity::Token | Granularity::Unstated) => Some(Format::Fp8E4m3 {
            scale: Scale::PerToken,
        }),
        (8, Granularity::Group(g)) => Some(Format::Fp8E4m3 {
            scale: Scale::Group(g),
        }),
        _ => None,
    }
}

impl EdgePrecision for DeclaredPrecision<'_> {
    fn linear(&self, module: &str) -> LinearFormats {
        let declared = self.plan.resolve(module);
        let weight = match declared.weight {
            None => Format::Bf16,
            Some(o) => weight_format(o).unwrap_or_else(|| {
                self.refuse(module, "weight", o);
                Format::Bf16
            }),
        };
        let activation = match declared.activation {
            None => Format::Bf16,
            Some(o) => activation_format(o).unwrap_or_else(|| {
                self.refuse(module, "activation", o);
                Format::Bf16
            }),
        };
        LinearFormats { weight, activation }
    }
}

#[cfg(test)]
#[path = "declared_precision_tests.rs"]
mod declared_precision_tests;
