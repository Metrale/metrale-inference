// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_PDL` (on unless switched off): programmatic dependent launch (PDL) for
//! the GLM-5.3 decode step. The kernels whose PTX entry waits on its grid dependencies
//! (`griddepcontrol.wait`, gb10 `common/pdl.cuh`) issue their weight loads before the wait and let
//! their successor launch right after it, so each launch's prologue overlaps its predecessor's
//! tail; the backend launches exactly those kernels programmatically once this switch is applied
//! (`GpuBackend::enable_programmatic_launches`). Their bytes do not change: without the attribute
//! the wait and the trigger are no-ops, and with it every read of another kernel's output still
//! follows the wait (`pdl_chain_gate`, gpu examples, replays the step's chain against a serialised
//! reference and carries a sabotage that the gate must catch).
//!
//! The kernels (2026-10-21): `w4a16_gemv_pipe_m{1,2}`, the W8A8 skinny GEMVs (`w8a8_gemv_*`),
//! `dense_gemv_bf16_batchm_few[_fp32out]`, the W4A4 union and own-slot sweeps and static
//! quantizers, `kda_gates_w8a8_k128[_rows]`, `glm5next_hc_pre_norm_c1`, `glm5next_hc_post`, the
//! KDA pack, conv, recurrence and output norm, `w8a8_act_quant_*`, the router top-k, row union,
//! SwiGLU, MoE combine and `bf16_add_rank_sum`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Read once at load ([`apply`], from `GlmLevers::pdl`); the backend's mode is one-way, so every
//!   launch after the load, captured or eager, takes the same path.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-10-21: Switch the backend's programmatic launches on when `on`, and report the mode. A
/// backend without them keeps every launch serialised and says so; that is not an error (the
/// kernels' bytes are the same either way).
pub fn apply(gpu: &dyn GpuBackend, on: bool) -> Result<()> {
    if !on {
        tracing::info!("METRALE_GLM_PDL=0: every GLM launch fully serialised");
        return Ok(());
    }
    if !gpu.enable_programmatic_launches()? {
        tracing::warn!(
            "METRALE_GLM_PDL: this backend has no programmatic dependent launch; launches stay \
             serialised"
        );
    }
    Ok(())
}
