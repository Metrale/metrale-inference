// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The `--activation-quantization` value in force for this process
//! (`metrale_config::ActivationQuantization`), and the one question decode dispatch asks of it:
//! does this projection family run one fixed activation format at every row count, and which.
//!
//! Owner: model-layers (quantization dispatch).
//! Invariants:
//! - The first publication or read wins (`OnceLock`). The serve publishes before the model is
//!   built. A process that never publishes (a test, an example) runs `adaptive`, today's
//!   routing, so kernel tests keep exercising the kernels they name.
//! - `adaptive` (a family whose ladder is all `adaptive`) takes no new branch anywhere: every
//!   dispatch site runs its pre-flag code for it.

use std::sync::OnceLock;

use metrale_config::{ActQuantFormat, ActivationQuantization, ProjFamily};

static VALUE: OnceLock<ActivationQuantization> = OnceLock::new();

/// 2026-09-30: Publish `--activation-quantization`. Returns the value in force; a caller that
/// gets a different one should warn.
pub fn set_activation_quantization_from_cli(
    v: ActivationQuantization,
) -> &'static ActivationQuantization {
    let _ = VALUE.set(v);
    VALUE.get().expect("just set")
}

/// 2026-09-30: The value in force: `adaptive` unless the serve published another.
pub fn activation_quantization() -> &'static ActivationQuantization {
    VALUE.get_or_init(ActivationQuantization::adaptive)
}

/// 2026-09-30: The format `family` runs at `rows` rows (`Adaptive`: the site's own routing).
pub fn act_route(family: ProjFamily, rows: usize) -> ActQuantFormat {
    activation_quantization().route(family, u32::try_from(rows).unwrap_or(u32::MAX))
}

/// 2026-09-30: `Some(format)` when `family` runs one fixed activation format at `rows` rows,
/// `None` when that row count is `adaptive` (the site's own routing, unchanged).
pub fn fixed_act(family: ProjFamily, rows: usize) -> Option<ActQuantFormat> {
    match act_route(family, rows) {
        ActQuantFormat::Adaptive => None,
        f => Some(f),
    }
}

/// 2026-09-30: Whether `family` runs a fixed format at some row count.
pub fn family_fixed(family: ProjFamily) -> bool {
    activation_quantization()
        .ladder(family)
        .rungs()
        .iter()
        .any(|r| r.format != ActQuantFormat::Adaptive)
}

/// 2026-09-30: Whether any family runs a fixed format at any row count. The serve then publishes
/// the canonical row tiers (`row_tiers.rs`), whose single-order kernels are the fixed `bf16`
/// path of the W8A16 projections and of an NVFP4 LM head.
pub fn any_fixed() -> bool {
    ProjFamily::ALL.into_iter().any(family_fixed)
}
