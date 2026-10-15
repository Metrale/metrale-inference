// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Gate for the warp-specialized prefill DSA attend
//! `glm5next_dsa_mla_prefill_tc_phi` (`METRALE_GLM_DSA_HB_PREFILL_TC`) against the kernel it
//! replaces, `glm5next_dsa_mla_decode_hb_fp8_phi`: byte equality (E4M3 NaN bytes included),
//! row invariance (sub-chunks, neighbours, block-table stride), the f64 bound, the
//! negative controls and timing, as listed in common/glm5next_dsa_twin_gate.rs, at one split
//! (the only split count the kernel runs).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error when any check of the shared gate fails.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_prefill_tc_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_hb_fixture.rs"]
mod fixture;
#[path = "common/glm5next_dsa_prefill_fixture.rs"]
mod prefill;
#[path = "common/glm5next_dsa_twin_gate.rs"]
mod twin;

use anyhow::Result;
use metrale_model_arch::glm5next_dsa::attend::{DSA_PREFILL_TC_MODULE, TC_BLOCK, TC_SMEM_BYTES};

fn main() -> Result<()> {
    twin::run_gate(&twin::Twin {
        label: "tc",
        module: DSA_PREFILL_TC_MODULE,
        entry: "glm5next_dsa_mla_prefill_tc_phi",
        block: TC_BLOCK,
        smem: TC_SMEM_BYTES,
        splits: &[1],
    })
}
