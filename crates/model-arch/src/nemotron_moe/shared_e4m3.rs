// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The `--nemotron-shared-expert-e4m3` switch, published once by the serve.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: read by the weight loader only after the serve published it.

use std::sync::OnceLock;

static E4M3: OnceLock<bool> = OnceLock::new();

/// 2026-09-29: Publish the command line's value; returns the value in force, which differs
/// when something read it first.
pub fn set_shared_expert_e4m3_from_cli(on: bool) -> bool {
    let _ = E4M3.set(on);
    *E4M3.get().expect("just set")
}

/// 2026-09-29: Whether the Nemotron-H shared expert's prefill runs the E4M3 tile GEMM. Off
/// unless the serve published otherwise: the checkpoint's precision (NVFP4 weights, 16-bit
/// activations on the BF16-MMA `w4a16_gemm`) is the default.
pub fn shared_expert_e4m3() -> bool {
    *E4M3.get_or_init(|| false)
}
