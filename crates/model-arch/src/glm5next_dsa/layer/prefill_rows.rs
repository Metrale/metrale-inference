// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The batched indexer of a DSA prefill sub-chunk (`decode_k_with`, `indexer_rows`):
//! the FP32-out projections of several rows in groups of at most `DENSE_GEMV_BATCHM_MAX_M`
//! (`f32_proj_rows`), and the indexer rows of one sequence's consecutive tokens projected,
//! normed, weighted and placed in one launch per stage (`stage_prefill_rows`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Each row's results carry the per-row path's bits: the projections run the batched GEMV
//!   (or the row-invariant W8A8 family) in groups of at most `DENSE_GEMV_BATCHM_MAX_M` rows,
//!   never the cuBLASLt arm `glm_mm` takes above that, and the store copies staging row `r` to
//!   row `r`'s position as the per-row store does.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::state::Glm5NextDsaState;
use super::{Glm5NextDsaLayer, gemm};

impl Glm5NextDsaLayer {
    /// 2026-10-09: `out[r] = a[r] @ wt^T` (FP32 out, `n_out` wide, K = `kk`, `a` rows `kk` BF16
    /// apart) for `n` rows, in groups of at most `DENSE_GEMV_BATCHM_MAX_M`: each row is the M = 1
    /// `gemv_f32`'s result for that row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn f32_proj_rows(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        wt: DevicePtr,
        out: DevicePtr,
        n: usize,
        n_out: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        let group = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
        let k = &self.kernels;
        for (r0, m) in crate::glm5next_layer::multi_seq_chunks(n, group) {
            gemm(
                gpu,
                k.gemm_f32,
                k.gemv_f32,
                k.batchm_f32(),
                a.offset(r0 * kk * 2),
                wt,
                out.offset(r0 * n_out * 4),
                m,
                n_out,
                kk,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-10-09: The indexer rows of `k` consecutive tokens of one sequence (`hidden` rows
    /// `0..k`) on a paged cache: key, gate and key norm into staging rows `0..k`
    /// (`indexer_project_rows`), the head weights into `head_weights_rows`, then one
    /// `dsa_indexer_store_rows` placing row `r` at `pos0[r]` through block-table row
    /// `bt0 + r * bt_rows` (`bt_rows` 0: every row reads one table), and the state advanced
    /// by `k` rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_prefill_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        st: &mut Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        pos0: DevicePtr,
        bt0: DevicePtr,
        bt_rows: u32,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        if w.head_weights_rows.0 == 0 || self.select_kernels.rows.store.0 == 0 {
            bail!(
                "DSA layer {}: the batched prefill indexer needs the batched-selector buffers \
                 and dsa_indexer_store_rows",
                self.layer_idx
            );
        }
        st.ensure_room(k)?;
        let c = &self.cfg;
        self.indexer_project_rows(gpu, hidden, k, stream)?;
        self.f32_proj_rows(
            gpu,
            hidden,
            self.weights.weights_proj,
            w.head_weights_rows,
            k,
            c.index_heads,
            c.hidden,
            stream,
        )?;
        let rows = self.indexer_rows(st, kv_cache, bt0)?;
        if rows.valid.0 != 0 || rows.bt.0 == 0 {
            bail!(
                "DSA layer {}: the batched prefill indexer is for paged caches",
                self.layer_idx
            );
        }
        let d = c.index_head_dim;
        KernelLaunch::new(gpu, self.select_kernels.rows.store)
            .grid([1, k as u32, 1])
            .block([d.min(1024) as u32, 1, 1])
            .arg_ptr(w.stage_k)
            .arg_ptr(w.stage_gate)
            .arg_ptr(pos0)
            .arg_ptr(rows.k)
            .arg_ptr(rows.gate)
            .arg_u32(d as u32)
            .arg_ptr(bt0)
            .arg_u32(bt_rows)
            .arg_u32(rows.block_size)
            .arg_u32(rows.blk_elems)
            .launch(stream)?;
        st.advance(k)
    }
}
