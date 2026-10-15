// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Gate for the pipelined prefill DSA attend `glm5next_dsa_mla_prefill_pipe_phi`
//! (`METRALE_GLM_DSA_HB_PREFILL_PIPE`) against the kernel it replaces,
//! `glm5next_dsa_mla_decode_hb_fp8_phi`: byte equality, row invariance, the f64 bound, the
//! negative controls and timing, as listed in common/glm5next_dsa_twin_gate.rs, at 1 and 8
//! splits.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error when any check of the shared gate fails.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_prefill_pipe_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_hb_fixture.rs"]
mod fixture;
#[path = "common/glm5next_dsa_prefill_fixture.rs"]
mod prefill;
#[path = "common/glm5next_dsa_twin_gate.rs"]
mod twin;

use anyhow::Result;
use metrale_model_arch::glm5next_dsa::attend::{DSA_PREFILL_PIPE_MODULE, PIPE_SMEM_BYTES};

fn main() -> Result<()> {
    twin::run_gate(&twin::Twin {
        label: "pipe",
        module: DSA_PREFILL_PIPE_MODULE,
        entry: "glm5next_dsa_mla_prefill_pipe_phi",
        block: 256,
        smem: PIPE_SMEM_BYTES,
        splits: &[1, 8],
    })
}
