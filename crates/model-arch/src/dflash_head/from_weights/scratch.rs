// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The drafter head's scratch allocation, moved out of `from_weights.rs` unchanged
//! (that file passed 500 lines with the tensor-parallel fields).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::dflash_head::DflashScratch;
use crate::weight_loader::DflashWeights;

/// 2026-10-10: The sizes `alloc_scratch` reads, as `from_weights` resolved them.
pub(super) struct ScratchDims<'a> {
    pub nb: usize,
    pub g: usize,
    pub rows_max: usize,
    pub hidden_size: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub intermediate_size: usize,
    pub ctx_window: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub target_layer_ids: &'a [usize],
    pub target_hidden_size: usize,
    pub max_seq_len: usize,
    pub gamma_val: usize,
    pub tp: Option<crate::dflash_head::tp::DflashTp>,
}

/// 2026-10-10: Allocate the head's scratch (see the field docs of [`DflashScratch`]).
pub(super) fn alloc_scratch(
    gpu: &dyn GpuBackend,
    weights: &DflashWeights,
    d: &ScratchDims<'_>,
) -> Result<DflashScratch> {
    let ScratchDims {
        nb,
        g,
        rows_max,
        hidden_size,
        q_dim,
        kv_dim,
        intermediate_size,
        ctx_window,
        num_layers,
        num_kv_heads,
        head_dim,
        vocab_size,
        target_layer_ids,
        target_hidden_size,
        max_seq_len,
        gamma_val,
        tp,
    } = *d;
    let bf16 = 2usize;
    Ok(DflashScratch {
        stream_buf: gpu.alloc(rows_max * hidden_size * bf16)?,
        norm_buf: gpu.alloc(rows_max * hidden_size * bf16)?,
        q_buf: gpu.alloc(rows_max * q_dim * bf16)?,
        k_buf: gpu.alloc(rows_max * kv_dim * bf16)?,
        v_buf: gpu.alloc(rows_max * kv_dim * bf16)?,
        attn_out: gpu.alloc(rows_max * q_dim * bf16)?,
        mlp_intermediate: gpu.alloc(rows_max * intermediate_size * bf16)?,
        mlp_up: gpu.alloc(rows_max * intermediate_size * bf16)?,
        stream_acc: gpu.alloc(rows_max * hidden_size * bf16)?,
        fc_proj: gpu.alloc(ctx_window * hidden_size * bf16)?,
        // 2026-09-25: The precompute's fused K/V output: up to `ctx_window` rows
        // (`precompute_ctx_kv` refuses more), `L * 2 * kv_dim` BF16 per row.
        fused_kv_out: gpu.alloc(ctx_window * num_layers * 2 * num_kv_heads * head_dim * bf16)?,
        // 2026-09-25: `ctx_window` i64 cache slots (`long long*` in
        // `fill_slots_from_block_table`): the precompute's ctx rows, and the block
        // rows in `forward_block`.
        slot_mapping_dev: gpu.alloc(ctx_window * 8)?,
        precompute_in: gpu.alloc(
            crate::dflash_head::PRECOMPUTE_BATCH_ROWS
                * target_layer_ids.len()
                * target_hidden_size
                * bf16,
        )?,
        // 2026-09-25: One `[kv_len, q_offset, q_rope_pos]` u32 triple (12 bytes) per
        // band, read by the indirect paged attention at kernel entry.
        option_b_indirect_args_dev: gpu.alloc(nb * 12)?,
        // 2026-09-25: Host staging for the drafted tokens (`nb * gamma` u32; page-locked
        // on CUDA), and the event `forward_block` records after copying into it.
        draft_tokens_host_pinned: std::sync::atomic::AtomicPtr::new(
            gpu.alloc_host_pinned(nb * gamma_val * 4)?,
        ),
        draft_tokens_event: gpu.create_event()?,
        // 2026-09-25: `nb * gamma` rows, not `rows_max`: the lm_head writes one row per
        // block row, and no path indexes the logits by a ctx offset.
        logits: gpu.alloc(nb * g * vocab_size * bf16)?,
        draft_tokens_dev: gpu.alloc(rows_max * 4)?,
        position_ids: gpu.alloc(rows_max * 4)?,
        // 2026-09-25: DSpark Markov scratch, allocated only when the drafter config
        // declares a Markov head (`markov_rank > 0`); `DevicePtr(0)` otherwise.
        markov_embed: if weights.config.markov_rank > 0 {
            gpu.alloc(weights.config.markov_rank * bf16)?
        } else {
            DevicePtr(0)
        },
        markov_bias: if weights.config.markov_rank > 0 {
            gpu.alloc(vocab_size * bf16)?
        } else {
            DevicePtr(0)
        },
        conf_out: if weights.confidence_proj.is_some() {
            gpu.alloc(g * bf16)?
        } else {
            DevicePtr(0)
        },
        // 2026-09-25: DFlash2 scratch, allocated only when the checkpoint ships the
        // selector. `dflash2_active` requires the selector, so the convs and the walk
        // never run without these buffers.
        conv_dyn: if weights.selector_pred.is_some() {
            let cfg = weights.config.dflash_config.as_ref();
            let ksz = cfg.map(|c| c.conv_kernel_size).unwrap_or(0).max(1);
            let gsz = cfg.map(|c| c.conv_group_size).unwrap_or(0).max(1);
            gpu.alloc(nb * g * 2 * ksz * (hidden_size / gsz) * bf16)?
        } else {
            DevicePtr(0)
        },
        conv_tmp: if weights.selector_pred.is_some() {
            gpu.alloc(nb * g * hidden_size * bf16)?
        } else {
            DevicePtr(0)
        },
        sel_vals: if weights.selector_pred.is_some() {
            gpu.alloc(nb * g * 16 * 4)?
        } else {
            DevicePtr(0)
        },
        sel_idx: if weights.selector_pred.is_some() {
            gpu.alloc(nb * g * 16 * 4)?
        } else {
            DevicePtr(0)
        },
        sel_cand_vals: if weights.selector_pred.is_some() {
            gpu.alloc(nb * g * crate::dflash_head::dflash2::topk16_parts(vocab_size) * 16 * 4)?
        } else {
            DevicePtr(0)
        },
        sel_cand_idx: if weights.selector_pred.is_some() {
            gpu.alloc(nb * g * crate::dflash_head::dflash2::topk16_parts(vocab_size) * 16 * 4)?
        } else {
            DevicePtr(0)
        },
        tp_header: match tp {
            Some(_) => gpu.alloc((crate::dflash_head::tp::HEADER_WORDS + max_seq_len) * 4)?,
            None => DevicePtr(0),
        },
        tp_send: match tp {
            Some(_) => gpu.alloc(nb * g * 32 * 4)?,
            None => DevicePtr(0),
        },
        tp_recv: match tp {
            Some(t) => gpu.alloc(t.world * nb * g * 32 * 4)?,
            None => DevicePtr(0),
        },
        sel_hproj: if weights.selector_pred.is_some() {
            let rank = weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_rank)
                .unwrap_or(256)
                .max(1);
            gpu.alloc(nb * g * rank * bf16)?
        } else {
            DevicePtr(0)
        },
    })
}
