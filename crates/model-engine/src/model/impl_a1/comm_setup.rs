// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Comm-backend setup for `TransformerModel::new`: the startup check that the
//! ranks agree on the collective-shaping levers, and the world-size-2 reduce-buffer
//! registration.
//!
//! Owner: model-engine.
//! Invariants:
//! - `register_reduce_buffers` never fails construction: each step logs its failure and continues.

use std::sync::Arc;

use anyhow::Result;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-09-26: Fail unless every rank resolved the same collective-shaping levers.
pub(super) fn assert_rank_levers_agree(
    gpu: &dyn GpuBackend,
    comm: &Arc<dyn metrale_comm::CommBackend>,
) -> Result<()> {
    metrale_model_layers::speculative::check_k2_fused_wire(
        metrale_model_layers::speculative::k2_fused_requested(),
        metrale_model_layers::ep_wire::EpWire::get(),
    )?;
    crate::rank_agree::assert_ranks_agree(
        gpu,
        comm.as_ref(),
        &[
            // 2026-09-25: Splits a prefill chunk into sub-chunks, and each
            // sub-chunk issues its own collectives.
            (
                "METRALE_GLM_PREFILL_ROWS",
                metrale_model_arch::glm5next_layer::prefill_rows() as u64,
            ),
            // 2026-09-25: Performance only (the MLP reduces once per site
            // whichever arm runs); checked because the check is free.
            (
                "METRALE_GLM_MOE_ROW_BATCH_MAX",
                metrale_model_arch::glm5next_mlp::forward::row_batch_max() as u64,
            ),
            // 2026-09-25: The EP command protocol (`ep_protocol_v2`), which
            // both ranks must share. 2026-10-10: the whole wire (v1, v2 or v3).
            (
                "METRALE_EP_PROTOCOL",
                metrale_model_layers::ep_wire::EpWire::get().agreement_value(),
            ),
            // 2026-10-11: The fused K=2 MTP wire changes the command sequence of every
            // K=2 step, so a rank running the other shape would read the wrong words.
            (
                metrale_model_layers::speculative::K2_FUSED_ENV,
                u64::from(metrale_model_layers::speculative::k2_fused_requested()),
            ),
            // 2026-10-11: The tensor-parallel drafter `eh_proj` adds an all-reduce to every
            // propose.
            (
                "METRALE_GLM_MTP_EH_TP",
                u64::from(metrale_model_arch::glm5next_mtp_head::eh_tp_enabled()),
            ),
            // 2026-10-13: The drafter's row space: a rank with another cap would restart its
            // drafter at another propose (`propose_start`) and draft other tokens.
            (
                "METRALE_GLM_MTP_CONTEXT_CAP",
                metrale_model_arch::glm5next_mtp_head::drafter_context_cap()? as u64,
            ),
            // 2026-10-11: A device-step verify reads row 1 from each rank's own drafter cell, so
            // every rank must leave the draft there the same way.
            (
                "METRALE_GLM_MTP_DEVICE_STEP",
                u64::from(super::super::trait_impl::device_step::device_step_enabled()),
            ),
            // 2026-10-11: The device draft exchange moves an all-reduce into the drafter graph.
            (
                "METRALE_GLM_MTP_XCHG_DEVICE",
                u64::from(std::env::var("METRALE_GLM_MTP_XCHG_DEVICE").as_deref() == Ok("1")),
            ),
            // 2026-10-12: SP-mHC changes every prefill sub-chunk's collectives from one
            // all-reduce to a reduce-scatter and all-gathers, so every rank must take it alike.
            (
                "METRALE_GLM_SP_MHC",
                u64::from(metrale_model_arch::glm5next_sp_mhc::enabled()?),
            ),
            // 2026-10-14: SP overlap cuts every SP-mHC exchange into row pieces on a comm
            // stream, so every rank must enqueue the same pieces.
            (
                "METRALE_GLM_SP_OVERLAP",
                u64::from(metrale_model_arch::glm5next_sp_overlap::enabled()?),
            ),
            // 2026-10-14: The SP pipeline reorders a layer's SP-mHC exchanges across its
            // sub-chunks, so every rank must enqueue the same program.
            (
                "METRALE_GLM_SP_PIPELINE",
                u64::from(metrale_model_arch::glm5next_sp_pipeline::enabled()?),
            ),
            // 2026-10-10: The resolved context window determines DFlash context-broadcast
            // chunking, so every rank must use the same value.
            (
                "METRALE_DFLASH_CTX_WINDOW",
                std::env::var("METRALE_DFLASH_CTX_WINDOW")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4096) as u64,
            ),
        ],
    )?;
    Ok(())
}

/// 2026-09-26: Register the reduce targets with the comm backend and hand it the
/// `bf16_add_inplace` kernel.
pub(super) fn register_reduce_buffers(
    comm: &Arc<dyn metrale_comm::CommBackend>,
    buffers: &BufferArena,
    gpu: &dyn GpuBackend,
) {
    let moe_ptr = buffers.moe_output().0;
    let moe_bytes = buffers.sizes().moe_output;
    match comm.register_buffer(moe_ptr, moe_bytes) {
        Ok(_) => {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "Registered moe_output ({moe_bytes} B) with NCCL")
        }
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "ncclCommRegister moe_output failed (non-fatal): {e}")
        }
    }
    let norm_ptr = buffers.norm_output().0;
    let norm_bytes = buffers.sizes().norm_output;
    match comm.register_buffer(norm_ptr, norm_bytes) {
        Ok(_) => {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "Registered norm_output ({norm_bytes} B) with NCCL")
        }
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "ncclCommRegister norm_output failed (non-fatal): {e}")
        }
    }
    let logits_ptr = buffers.logits().0;
    let logits_bytes = buffers.sizes().logits;
    match comm.register_buffer(logits_ptr, logits_bytes) {
        Ok(_) => {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "Registered logits ({logits_bytes} B) with NCCL")
        }
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "ncclCommRegister logits failed (non-fatal): {e}")
        }
    }
    match gpu.kernel("bf16_add", "bf16_add_inplace") {
        Ok(k) => comm.set_add_kernel(k.0),
        Err(e) => {
            tracing::warn!(target: "metrale_model_engine::model::impl_a1", "bf16_add_inplace kernel not found (send/recv disabled): {e}")
        }
    }
    // 2026-10-08: The one-shot all-reduce (world >= 3) refuses to run without this kernel, so a
    // missing kernel surfaces as an error at the first collective, never as a silent NCCL fallback.
    if comm.world_size() >= 3
        && let Ok(k) = gpu.kernel("bf16_add", "bf16_add_rank_sum")
    {
        comm.set_rank_sum_kernel(k.0);
    }
}
