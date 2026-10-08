// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `Glm5NextDsaLayer::decode_rows`, one decode token for each of `n` sequences in
//! one call: the batched multi-sequence decode's DSA mixer.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Every check (row counts, metadata rows, each row's lockstep and room) runs before the
//!   first launch.
//! - Row `r` reads and writes only sequence `r`'s state and metadata row
//!   `meta_row_base + r`; the projections, the latent write and the attend are the only
//!   launches that span rows, and each computes every row on its own.
//!
//! # Why a row's output equals the single-sequence decode's
//!
//! Per row, this issues the launches `decode_k` issues for that sequence alone at `k = 1` on a
//! decode step: `indexer_forward` (M = 1 GEMVs), the device geometry when capturing, and
//! `select_row` over the row's own indexer cache. The spanning launches give each row the
//! single-row bits: the latent write runs one block per row, reading the row's metadata slot;
//! `project_in`/`project_out` run the
//! M = 1 GEMV at one row and `dense_gemv_bf16_batchm` at 2..=`DENSE_GEMV_BATCHM_MAX_M`
//! (`kernels/gb10/common/dense_gemv_bf16_batchm.cu`: each row's result is bit-identical to
//! `dense_gemv_bf16`); the RMSNorm runs one block per row; `glm5next_dsa_mla_decode_fp8` runs
//! one block per (head, row) reading that row's block table, `seq_len` and selection row.
//! Above `DENSE_GEMV_BATCHM_MAX_M` rows the projections move to cuBLASLt and the identity is
//! lost, so the layer above hands this at most that many rows (`multi_seq_chunk_rows`).

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState};

use super::super::attend::DsaDecodePaging;
use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;

/// 2026-10-08: Row `r`'s `Glm5NextDsaState`; errors on any other state type.
fn dsa_row<'a>(
    states: &'a mut [&mut (dyn LayerState + 'static)],
    r: usize,
) -> Result<&'a mut Glm5NextDsaState> {
    states[r]
        .as_any_mut()
        .downcast_mut::<Glm5NextDsaState>()
        .ok_or_else(|| {
            anyhow::anyhow!("Glm5NextDsaLayer: row {r} got a state that is not Glm5NextDsaState")
        })
}

impl Glm5NextDsaLayer {
    /// 2026-10-08: One decode token for each of `states.len()` sequences: row `r` of `hidden`
    /// is sequence `r`'s token at position `seq_lens[r]`, and its position, KV slot,
    /// `seq_len` and block table are row `meta_row_base + r` of `meta`. The output projection
    /// is written over `hidden`, as `decode_k` writes it.
    ///
    /// Errors before any launch when the row counts disagree, the rows do not fit the
    /// workspace or the metadata, a capture lacks the replay-safe kernels, or a row's
    /// indexer cache is behind its sequence or full. An indexer cache ahead of its sequence
    /// (a rejected draft) is rewound first, as in `decode_k`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_rows(
        &self,
        hidden: DevicePtr,
        states: &mut [&mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        meta_row_base: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let n = states.len();
        if n == 0 || n != seq_lens.len() || n > self.workspace.max_rows {
            bail!(
                "DSA layer {}: {n} row states and {} seq_lens for a workspace built for {} rows",
                self.layer_idx,
                seq_lens.len(),
                self.workspace.max_rows
            );
        }
        if meta_row_base + n > meta.num_seqs as usize {
            bail!(
                "DSA layer {}: rows {meta_row_base}..{} pass the {} metadata rows",
                self.layer_idx,
                meta_row_base + n,
                meta.num_seqs
            );
        }
        // 2026-10-08: A captured graph replays over later positions, so it must place the
        // indexer row and size the selection from device memory; without these two kernels
        // it would bake this step's host offsets.
        let replay_safe = ctx.graph_capture;
        if replay_safe
            && (self.select_kernels.indexer_store.0 == 0 || self.select_kernels.write_geom.0 == 0)
        {
            bail!(
                "DSA layer {}: a captured batched decode needs dsa_indexer_store and \
                 dsa_write_geom, which this target lacks",
                self.layer_idx
            );
        }
        for (r, &seq_len) in seq_lens.iter().enumerate() {
            let st = dsa_row(states, r)?;
            self.check_lockstep(st, seq_len)?;
            st.ensure_room(1)?;
        }

        let gpu = ctx.gpu;
        let block_size = kv_cache.config().block_size;
        let t_proj = profile::start();
        self.project_in(gpu, hidden, n, stream)?;
        // 2026-10-08: One latent-write launch for every row: the metadata's slots are a
        // contiguous i64 array, and the kernel writes row `i` to `slot[i]`.
        self.write_latent_rows(
            gpu,
            0,
            n,
            kv_cache,
            meta.slot.offset(meta_row_base * 8),
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        for r in 0..n {
            let mr = meta_row_base + r;
            let t = profile::start();
            let st = dsa_row(states, r)?;
            let pos_dev = replay_safe.then(|| meta.positions.offset(mr * 4));
            self.indexer_forward(
                gpu,
                hidden.offset(r * self.cfg.hidden * 2),
                st,
                pos_dev,
                stream,
            )?;
            profile::end(profile::DSA_INDEXER, t, gpu, stream);
            if replay_safe {
                self.write_geom(gpu, meta.seq_len.offset(mr * 4), stream)?;
            }
            self.select_row(
                gpu,
                r,
                st,
                meta.positions.offset(mr * 4),
                replay_safe,
                stream,
            )?;
        }

        let bt_stride = meta.max_blocks_per_seq as usize * 4;
        let paging = DsaDecodePaging {
            num_seqs: n,
            num_q_heads: self.cfg.local_heads,
            num_kv_heads: 1,
            max_blocks_per_seq: meta.max_blocks_per_seq as usize,
            block_size,
            cache_stride_bytes: (block_size * self.cfg.kv_lora_rank) as u64,
        };
        let st0 = dsa_row(states, 0)?;
        self.attend_rows(
            gpu,
            n,
            st0,
            kv_cache,
            meta.block_table.offset(meta_row_base * bt_stride),
            meta.seq_len.offset(meta_row_base * 4),
            &paging,
            stream,
        )?;
        self.project_out(gpu, hidden, n, stream)
    }
}

#[cfg(test)]
#[path = "decode_rows_tests.rs"]
mod tests;
