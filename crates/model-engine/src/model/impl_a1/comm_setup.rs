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
// 2026-10-17: The DFlash context window every rank agrees on is the one the head is built with.
use metrale_model_arch::dflash_head::dflash_ctx_window;

/// 2026-09-26: Fail unless every rank resolved the same collective-shaping levers.
pub(super) fn assert_rank_levers_agree(
    gpu: &dyn GpuBackend,
    comm: &Arc<dyn metrale_comm::CommBackend>,
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<()> {
    metrale_model_layers::speculative::check_k2_fused_wire(
        metrale_model_layers::speculative::k2_fused_requested(),
        metrale_model_layers::ep_wire::EpWire::get(),
    )?;
    crate::model::ep_ctrl::check_config(
        settings.runtime.glm_ep_ctrl_comm,
        metrale_model_layers::ep_wire::EpWire::get(),
    )?;
    if settings.speculative.glm_graph_early_launch && !settings.runtime.nccl_host_hoist {
        tracing::error!(
            "METRALE_GLM_GRAPH_EARLY_LAUNCH=1 is refused without METRALE_NCCL_HOST_HOIST=1 (a \
             graph's first NCCL host callback would still wait for its producers): the graphs \
             launch on the compute stream as before"
        );
    }
    crate::rank_agree::assert_ranks_agree(gpu, comm.as_ref(), &rank_lever_table(settings)?)?;
    Ok(())
}

/// 2026-10-21: The collective-shaping levers every rank must resolve alike, as
/// `assert_rank_levers_agree` compares them.
pub(super) fn rank_lever_table(
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<Vec<(&'static str, u64)>> {
    let ctrl = ctrl_comm_agreement(settings);
    let glm = metrale_model_arch::glm5next_layer::GlmLevers::from_settings(settings)?;
    let mtp = metrale_model_arch::glm5next_mtp_head::MtpLevers::from_settings(settings)?;
    Ok(vec![
        // 2026-09-25: Splits a prefill chunk into sub-chunks, and each
        // sub-chunk issues its own collectives.
        ("METRALE_GLM_PREFILL_ROWS", glm.prefill_rows as u64),
        // 2026-09-25: Performance only (the MLP reduces once per site
        // whichever arm runs); checked because the check is free.
        (
            "METRALE_GLM_MOE_ROW_BATCH_MAX",
            glm.mlp.row_batch_max as u64,
        ),
        // 2026-09-25: The EP command protocol (`ep_protocol_v2`), which
        // both ranks must share. 2026-10-10: the whole wire (v1, v2 or v3).
        (
            "METRALE_EP_PROTOCOL",
            metrale_model_layers::ep_wire::EpWire::get().agreement_value(),
        ),
        // 2026-10-21: The control channel moves every command block and K=2 verdict to a
        // second communicator, which every rank bootstraps on the same port.
        ctrl[0],
        ctrl[1],
        // 2026-10-11: The fused K=2 MTP wire changes the command sequence of every
        // K=2 step, so a rank running the other shape would read the wrong words.
        (
            metrale_model_layers::speculative::K2_FUSED_ENV,
            u64::from(metrale_model_layers::speculative::k2_fused_requested()),
        ),
        // 2026-10-11: The tensor-parallel drafter `eh_proj` adds an all-reduce to every
        // propose.
        ("METRALE_GLM_MTP_EH_TP", u64::from(mtp.eh_tp)),
        // 2026-10-13: The drafter's row space: a rank with another cap would restart its
        // drafter at another propose (`propose_start`) and draft other tokens.
        ("METRALE_GLM_MTP_CONTEXT_CAP", mtp.context_cap as u64),
        // 2026-10-11: A device-step verify reads row 1 from each rank's own drafter cell, so
        // every rank must leave the draft there the same way.
        (
            "METRALE_GLM_MTP_DEVICE_STEP",
            u64::from(settings.speculative.glm_mtp_device_step),
        ),
        // 2026-10-11: The device draft exchange moves an all-reduce into the drafter graph.
        ("METRALE_GLM_MTP_XCHG_DEVICE", u64::from(mtp.xchg_device)),
        // 2026-10-15: The DSA SP selection's table decides which prefill passes enqueue
        // two all-gathers of selected rows, so every rank must hold the same table; the
        // payload sub-lever (`_SEL`, low bit) sets their row size.
        ("METRALE_GLM_DSA_SP_SELECT", glm.dsa.sp_agreement),
        // 2026-10-12: SP-mHC changes every prefill sub-chunk's collectives from one
        // all-reduce to a reduce-scatter and all-gathers, so every rank must take it alike.
        ("METRALE_GLM_SP_MHC", u64::from(glm.mhc.sp_mhc)),
        // 2026-10-14: SP overlap cuts every SP-mHC exchange into row pieces on a comm
        // stream, so every rank must enqueue the same pieces.
        ("METRALE_GLM_SP_OVERLAP", u64::from(glm.mhc.sp_overlap)),
        // 2026-10-14: The SP pipeline reorders a layer's SP-mHC exchanges across its
        // sub-chunks, so every rank must enqueue the same program.
        ("METRALE_GLM_SP_PIPELINE", u64::from(glm.mhc.sp_pipeline)),
        // 2026-10-10: The resolved context window determines DFlash context-broadcast
        // chunking, so every rank must use the same value.
        (
            "METRALE_DFLASH_CTX_WINDOW",
            dflash_ctx_window(settings) as u64,
        ),
        // 2026-10-21: The early launch moves the verify and drafter graphs to side streams
        // gated on the compute stream; resolved with the NCCL host hoist it needs.
        (
            "METRALE_GLM_GRAPH_EARLY_LAUNCH",
            u64::from(mtp.graph_early_launch),
        ),
        // 2026-10-21: Host timing only (the slot upload is stream-ordered either way); checked
        // because the check is free.
        ("METRALE_GLM_MTP_ROW_NOSYNC", u64::from(mtp.row_nosync)),
    ])
}

/// 2026-10-21: The rank-agreed entries of the control channel: the lever, and its port (an
/// unset port as `u64::MAX`, so it can equal no set one).
pub(super) fn ctrl_comm_agreement(
    settings: &metrale_settings::InferenceServerSettings,
) -> [(&'static str, u64); 2] {
    [
        (
            crate::model::ep_ctrl::ENV,
            u64::from(settings.runtime.glm_ep_ctrl_comm),
        ),
        (
            "METRALE_GLM_EP_CTRL_PORT",
            settings
                .runtime
                .glm_ep_ctrl_port
                .map_or(u64::MAX, |p| p as u64),
        ),
    ]
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
    // 2026-10-08: The one-shot all-reduce refuses to run without this kernel, so a missing kernel
    // surfaces as an error at the first collective, never as a silent NCCL fallback. 2026-10-16:
    // From world 2 (was 3), where the one-shot is offered too.
    if comm.world_size() >= 2
        && let Ok(k) = gpu.kernel("bf16_add", "bf16_add_rank_sum")
    {
        comm.set_rank_sum_kernel(k.0);
    }
}

#[cfg(test)]
#[path = "comm_setup_tests.rs"]
mod tests;
