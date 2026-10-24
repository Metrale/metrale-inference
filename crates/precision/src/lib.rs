// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Precision declarations: what a checkpoint's `quantization_config` declares per
//! linear layer ([`precision_plan`]), the `--weight-quantization` policy over it
//! ([`weight_quantization`]) and `--activation-quantization` ([`activation_quantization`]).
//!
//! Split from metrale-config, which re-exports each module at its old path, so the engine's
//! loaders and the architecture circuits (metrale-circuit) ask one definition.
//!
//! Owner: precision (quantization).
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`, no I/O, no environment; builds for both wasm32 targets
//!   (NSCC).
//! - Every error is a registry code (CERF): `0x0700–0x070F`.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

pub mod activation_quantization;
pub mod precision_plan;
pub mod weight_quantization;

pub use activation_quantization::{ActQuantFormat, ActivationQuantization, ProjFamily};
pub use precision_plan::{DeclaredPrecisionPlan, LayerPrecision};
pub use weight_quantization::{
    Nvfp4Act, W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization,
};

/// 2026-10-22: The result every fallible function here returns.
pub type Result<T> = core::result::Result<T, metrale_error::Error>;
