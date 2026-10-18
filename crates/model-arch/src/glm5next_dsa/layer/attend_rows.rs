// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `Glm5NextDsaLayer::attend_rows`, the gather-attend of a decode step or prefill
//! sub-chunk, moved out of `layer.rs` unchanged but for taking the prefill levers and the
//! head-batched diagnostics from the levers resolved at load.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: a prefill attend runs the kernel `for_prefill_levers` returns under
//! `levers.dsa.prefill`; a decode attend runs the kernel resolved at load.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::attend::{DsaDecodeInputs, DsaDecodePaging, HbDiagnostics, decode_attention};
use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;

impl Glm5NextDsaLayer {
    /// 2026-09-25: The gather-attend for all `rows` query rows in one launch.
    ///
    /// Grid y is the row. Each row reads the paged latent cache through its own `q_abs` row,
    /// selection row and `seq_lens` entry, and the block-table row at stride
    /// `max_blocks_per_seq` (0 when the rows share one table).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        paging: &DsaDecodePaging,
        // 2026-10-12: A prefill sub-chunk's attend (`Glm5NextDsaDecodeKernel::for_prefill`).
        is_prefill: bool,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let w = &self.workspace;
        // 2026-09-25: `decode_attention` reads only `out_width` and `q_rows` from it.
        let geom = state.geometry(&self.cfg, rows)?;
        let paging = DsaDecodePaging {
            num_seqs: rows,
            ..*paging
        };
        let t = self.levers.profile.start();
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let kernel = if is_prefill {
            self.decode_kernel
                .for_prefill_levers(self.levers.dsa.prefill)?
        } else {
            self.decode_kernel
        };
        decode_attention(
            gpu,
            kernel,
            &self.cfg,
            &geom,
            &paging,
            &DsaDecodeInputs {
                q: w.q_abs,
                k_cache: pool,
                v_cache: pool, // 2026-09-25: absorbed NoPE MLA: K and V are the same latent
                out: w.attn_out,
                block_tables: block_table_dev,
                seq_lens: seq_lens_dev,
                sel_indices: w.select.tokens(),
                k_scale: self.kv_scale,
                v_scale: self.kv_scale,
                split_ws: w.split_ws,
            },
            HbDiagnostics {
                check_dir: self.dsa_owned.hb_check.as_deref(),
                serve_ref: self.levers.dsa.serve_ref,
            },
            stream,
        )?;
        profile::end(profile::DSA_ATTEND, t, gpu, stream);
        Ok(())
    }
}
