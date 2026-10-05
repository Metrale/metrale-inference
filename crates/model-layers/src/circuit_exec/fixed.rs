// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The device addresses a program is compiled against: the model's buffers and
//! metadata uploads ([`Fixed`]) and the MTP draft head's ([`DraftFixed`]). Split from
//! `compile.rs`.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every address here is fixed after boot, so a captured program stays valid.

use metrale_gpu_runtime::gpu::DevicePtr;

use super::bindings::DraftRows;
use crate::layer::AttnMetadataDev;

/// 2026-09-28: Device addresses that never move after boot.
#[derive(Clone)]
pub struct Fixed {
    /// 2026-09-28: The residual stream (`BufferArena::hidden_states`).
    pub hidden: DevicePtr,
    /// 2026-09-28: The copy of the stream the norms write (`BufferArena::residual`).
    pub residual: DevicePtr,
    /// 2026-09-28: The logits buffer the step returns.
    pub logits: DevicePtr,
    /// 2026-09-29: Where a verify's argmax writes each row's token (`i32` per row): the
    /// scratch buffer's start, which the verify reads back.
    pub tokens: DevicePtr,
    /// 2026-09-28: The single-sequence step's attention metadata, at its fixed upload
    /// address. `max_blocks_per_seq` is ignored: a step supplies it.
    pub meta: AttnMetadataDev,
    /// 2026-09-28: The multi-sequence step's attention metadata (one row per sequence), at its
    /// fixed upload address; `max_blocks_per_seq` is ignored.
    pub batch_meta: AttnMetadataDev,
    /// 2026-09-29: The MTP verify step's attention metadata (one row per verified token), at
    /// its fixed upload address; `max_blocks_per_seq` is ignored.
    pub verify_meta: AttnMetadataDev,
    /// 2026-09-29: The MTP draft head's buffers; `None` without a bound draft head.
    pub draft: Option<DraftFixed>,
    /// 2026-09-30: The batched MTP verify's attention metadata (one row per verified row,
    /// `stage_verify_metadata` at scratch + 32768); `max_blocks_per_seq` is ignored.
    pub verify_batch_meta: AttnMetadataDev,
    /// 2026-09-30: The batched verify's per-GDN-layer WY pointer tables
    /// (`upload_verify_wy_tables`, `VERIFY_WY_LAYER_STRIDE_BYTES` apart); NULL without MTP.
    pub verify_wy_tables: DevicePtr,
    /// 2026-09-30: Where the batched verify's argmax writes each row's token: the mapped host
    /// blob's device alias, or scratch when there is none (`verify_rows_argmax`).
    pub verify_batch_tokens: DevicePtr,
    /// 2026-09-28: The quantized-activation scratch the NVFP4 MMQ GEMMs read
    /// (`BufferArena::ffn_act_q8`), sized for the widest batch.
    pub ffn_act_q8: DevicePtr,
    /// 2026-09-28: K pool per attention layer (`PagedKvCache::k_pool_ptr`).
    pub k_pools: Vec<DevicePtr>,
    /// 2026-09-28: V pool per attention layer.
    pub v_pools: Vec<DevicePtr>,
    /// 2026-09-28: Tokens per KV block.
    pub block_size: u32,
    /// 2026-09-28: `PagedKvCache::cache_stride`.
    pub cache_stride: u64,
    /// 2026-10-03: The arena scratch the grouped MoE decode sorts into, with the arena
    /// capacities its width check reads (`MoeScratch::from_arena`); `None` where no layer binds
    /// a MoE.
    pub moe: Option<crate::layers::moe::MoeScratch>,
}

/// 2026-09-29: The MTP draft head's fixed buffers.
#[derive(Clone)]
pub struct DraftFixed {
    /// 2026-09-29: Where the host puts the token's embedding row (`MtpHead::forward_one`'s
    /// `ssm_qkvz`).
    pub embed: DevicePtr,
    /// 2026-09-29: The draft step's attention metadata (`mtp_meta::mtp_attn_meta_dev`);
    /// `max_blocks_per_seq` is ignored.
    pub meta: AttnMetadataDev,
    /// 2026-09-29: The draft cache's pools and geometry.
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_size: u32,
    pub cache_stride: u64,
    /// 2026-09-29: The vocabulary rows the draft lm_head scores (`DraftBinding::vocab`).
    pub vocab: u32,
    /// 2026-09-30: The head's n-row draft (the batched propose) facts (`DraftBinding::rows`);
    /// `None` when the head drafts one row at a time.
    pub rows: Option<DraftRows>,
}

impl DraftFixed {
    /// 2026-09-30: The attention metadata an `n`-row draft step uploads: the batched
    /// propose's, at its allocation (`mtp_meta::mtp_attn_meta_batch_dev`); one row reads the
    /// single-row step's.
    pub fn meta_rows(&self, n: u64) -> anyhow::Result<AttnMetadataDev> {
        if n == 1 {
            return Ok(self.meta);
        }
        let rows = self.rows.as_ref().ok_or_else(|| {
            anyhow::anyhow!("a {n}-row draft plan for a head outside the batched propose")
        })?;
        Ok(crate::layers::mtp_meta::mtp_attn_meta_batch_dev(
            rows.meta,
            usize::try_from(n)?,
        ))
    }
}
