// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The head-batched launch of `decode_attention`, moved out of `attend.rs` unchanged
//! but for the dynamic shared memory the entry declares ([`HeadBatched::smem`]).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: a row's output does not depend on the launch it is in, so the row windowing below
//! never changes a byte.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{
    DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaConfig, HB_BLOCK, HB_HEADS, HB_LAUNCH_ROWS,
    HB_MERGE_BLOCK, HeadBatched,
};
use crate::glm5next_dsa::select::DsaSelectGeometry;

/// 2026-10-09: The head-batched decode of `decode_attention`, in launches of at most
/// [`HB_LAUNCH_ROWS`] rows; from two splits up each launch is followed by the merge.
pub(super) fn decode_head_batched(
    gpu: &dyn GpuBackend,
    hb: HeadBatched,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    // 2026-10-09: The kernel reads one latent buffer as K and V; distinct ones would need the
    // per-head kernel's second load.
    if inputs.k_cache != inputs.v_cache {
        bail!("DSA head-batched decode: K and V must be one latent buffer (absorbed MLA)");
    }
    let heads = paging.num_q_heads;
    let ws = match (hb.splits, inputs.split_ws) {
        (1, _) => None,
        (_, Some(ws)) if ws.splits == hb.splits && ws.heads == heads => Some(ws),
        (s, ws) => bail!(
            "DSA head-batched decode: {s} splits over {heads} heads needs a split workspace of \
             that shape, got {ws:?}"
        ),
    };
    // 2026-10-12: At one split there are no partials to bound, so every row goes in one launch
    // (a row's output does not depend on the launch it is in).
    let launch_rows = match ws {
        Some(w) => w.rows.min(HB_LAUNCH_ROWS),
        None if hb.splits == 1 => paging.num_seqs,
        None => HB_LAUNCH_ROWS,
    };
    let (ws_o, ws_ml) = ws.map_or((DevicePtr(0), DevicePtr(0)), |w| (w.o, w.ml));
    let latent_row = heads * cfg.kv_lora_rank * 2;
    let mut row0 = 0;
    while row0 < paging.num_seqs {
        let rows = launch_rows.min(paging.num_seqs - row0);
        let q = inputs.q.offset(row0 * latent_row);
        let out = inputs.out.offset(row0 * latent_row);
        let bt = inputs
            .block_tables
            .offset(row0 * paging.max_blocks_per_seq * 4);
        let sl = inputs.seq_lens.offset(row0 * 4);
        KernelLaunch::new(gpu, hb.decode)
            .grid([
                hb.splits as u32,
                rows as u32,
                heads.div_ceil(HB_HEADS) as u32,
            ])
            .block([HB_BLOCK, 1, 1])
            .shared_mem(hb.smem)
            .arg_ptr(q)
            .arg_ptr(inputs.k_cache)
            .arg_ptr(out)
            .arg_ptr(bt)
            .arg_ptr(sl)
            .arg_ptr(inputs.sel_indices.offset(row0 * geom.out_width * 4))
            .arg_ptr(ws_o)
            .arg_ptr(ws_ml)
            .arg_u32(geom.out_width as u32)
            .arg_u32(paging.max_blocks_per_seq as u32)
            .arg_u32(heads as u32)
            .arg_u32(paging.block_size as u32)
            .arg_f32((cfg.kv_lora_rank as f32).powf(-0.5))
            .arg_f32(inputs.k_scale)
            .arg_f32(inputs.v_scale)
            .arg_u64(paging.cache_stride_bytes)
            .arg_u32(hb.splits as u32)
            .launch(stream)?;
        if hb.splits > 1 {
            KernelLaunch::new(gpu, hb.merge)
                .grid([heads as u32, rows as u32, 1])
                .block([HB_MERGE_BLOCK, 1, 1])
                .arg_ptr(ws_o)
                .arg_ptr(ws_ml)
                .arg_ptr(sl)
                .arg_ptr(out)
                .arg_u32(heads as u32)
                .arg_u32(hb.splits as u32)
                .arg_f32(inputs.v_scale)
                .launch(stream)?;
        }
        row0 += rows;
    }
    Ok(())
}
