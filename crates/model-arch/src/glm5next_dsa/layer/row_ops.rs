// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The launches `decode_k` (rows of one sequence) and `decode_rows` (one row per
//! sequence) share: the input projections, the latent write, the device geometry, and the
//! output projection.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Each helper issues exactly the launches, arguments and order `decode_k` issued inline
//!   before they were extracted, so the single-sequence paths launch what they launched.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::Glm5NextDsaLayer;

impl Glm5NextDsaLayer {
    /// 2026-10-08: `q_a_proj`, its RMSNorm into `q_resid`, `q_absorb` into `q_abs`, and
    /// `kv_a_proj` into `kv_a`, each once for all `k` rows of `hidden`.
    pub(super) fn project_in(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        self.gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.batchm_bf16(),
            hidden,
            self.weights.q_a_proj,
            w.q_a,
            k,
            self.cfg.q_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        KernelLaunch::new(gpu, self.kernels.rms_norm)
            // 2026-09-25: `rms_norm_vanilla` runs one block per row, so one launch covers all
            // k rows.
            .grid([k as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.q_a)
            .arg_ptr(self.weights.q_a_layernorm)
            .arg_ptr(w.q_resid)
            .arg_u32(self.cfg.q_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        self.gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.batchm_bf16(),
            w.q_resid,
            self.weights.q_absorb,
            w.q_abs,
            k,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            self.cfg.q_lora_rank,
            stream,
        )?;
        self.gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.batchm_bf16(),
            hidden,
            self.weights.kv_a_proj,
            w.kv_a,
            k,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )
    }

    /// 2026-10-08: RMSNorm, FP8 quantisation and the paged write of `rows` consecutive `kv_a`
    /// rows from row `row`, row `row + i` into the KV slot at `slot_dev[i]` (i64).
    /// `glm5next_mla_latent_write_fp8` runs one block per row and the blocks share nothing,
    /// so a row's bytes do not depend on `rows`.
    pub(super) fn write_latent_rows(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        rows: usize,
        kv_cache: &PagedKvCache,
        slot_dev: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([rows as u32, 1, 1])
            .block([self.cfg.kv_lora_rank as u32, 1, 1])
            .arg_ptr(self.workspace.kv_a.offset(row * self.cfg.kv_lora_rank * 2))
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(slot_dev)
            .arg_u32(self.cfg.kv_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)
    }

    /// 2026-10-08: `dsa_write_geom` into the workspace's `geom_dev`, reading S from `d_sl`, a
    /// row's `seq_len` entry in the metadata.
    pub(super) fn write_geom(
        &self,
        gpu: &dyn GpuBackend,
        d_sl: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.select_kernels.write_geom)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(d_sl)
            .arg_ptr(self.workspace.geom_dev)
            .arg_u32(self.cfg.index_kpool as u32)
            .arg_u32(self.cfg.index_topk as u32)
            .arg_u32(super::super::select::topk_tile() as u32)
            .launch(stream)
    }

    /// 2026-10-08: `o_absorb` over the `k` rows of `attn_out`, written over `hidden`.
    pub(super) fn project_out(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let t_proj = self.levers.profile.start();
        self.gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.batchm_bf16(),
            self.workspace.attn_out,
            self.weights.o_absorb,
            hidden,
            k,
            self.cfg.hidden,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        Ok(())
    }

    /// 2026-10-09: The selector query (`wq_b` over `q_resid`) into `q_idx_rows` and the head
    /// weights (`weights_proj` over the layer input `hidden`) into `head_weights_rows` of `n`
    /// rows, one launch each, so `wq_b` is read once per group instead of once per row. Each
    /// row of `dense_gemv_bf16_batchm_fp32out` is the M = 1 `gemv_f32`'s result for that row.
    /// `wide` (more rows than `DENSE_GEMV_BATCHM_MAX_M`, `cublas_wide_proj`) takes the FP32-out
    /// cuBLASLt call instead, which sums in another order. 2026-10-11: moved here from
    /// `decode_rows` so the k-row verify (`decode_k_with`) shares it.
    pub(super) fn selector_query_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        n: usize,
        wide: bool,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let (heads, idx_row) = (
            self.cfg.index_heads,
            self.cfg.index_heads * self.cfg.index_head_dim,
        );
        let k = &self.kernels;
        for (a, wt, out, n_out, kk) in [
            (
                w.q_resid,
                self.weights.wq_b,
                w.q_idx_rows,
                idx_row,
                self.cfg.q_lora_rank,
            ),
            (
                hidden,
                self.weights.weights_proj,
                w.head_weights_rows,
                heads,
                self.cfg.hidden,
            ),
        ] {
            if wide {
                metrale_model_layers::layers::ops::cublas_bf16_proj_dense_f32_out(
                    a,
                    wt,
                    out,
                    n as u32,
                    n_out as u32,
                    kk as u32,
                    stream,
                )?;
            } else {
                self.gemm(
                    gpu,
                    k.gemm_f32,
                    k.gemv_f32,
                    k.batchm_f32(),
                    a,
                    wt,
                    out,
                    n,
                    n_out,
                    kk,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-10-09: Whether `selector_query_rows` can run for `n` rows: two rows or more (one
    /// row keeps the single-sequence launches), the batched FP32-out GEMV and the M = 1
    /// `gemv_f32` resolved, and the row buffers allocated.
    pub(super) fn selector_query_rows_ready(&self, n: usize) -> bool {
        let w = &self.workspace;
        n > 1
            && self.kernels.gemv_batchm_f32.0 != 0
            && self.kernels.gemv_f32.0 != 0
            && w.q_idx_rows.0 != 0
            && w.head_weights_rows.0 != 0
    }

    /// 2026-10-09: The indexer key projection (`wk`), its LayerNorm (`k_norm`) and the
    /// compress-gate projection of all `n` rows of `hidden`, into staging rows `0..n`
    /// (`stage_k`, `stage_gate`), for `store_indexer_row` to place row by row. Per row these are
    /// `indexer_forward_with`'s launches: the projections run in groups of at most
    /// `DENSE_GEMV_BATCHM_MAX_M` rows, so each row has the M = 1 projection's bits (the batched
    /// BF16 GEMV under `declared`, the row-invariant W8A8 family under `fp8` and `w4a16`, whose
    /// one activation quantization both projections share under a `stable_input` guard), and
    /// `nllb_layernorm_bf16` runs one block per row. Refuses more rows than the workspace.
    pub(super) fn indexer_project_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        anyhow::ensure!(
            (1..=w.max_rows).contains(&n),
            "DSA layer {}: {n} indexer rows for a workspace of {}",
            self.layer_idx,
            w.max_rows
        );
        let (d, h) = (self.cfg.index_head_dim, self.cfg.hidden);
        let group = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
        let _stable = crate::glm5next_fp8_dense::stable_input(hidden);
        let whole = self.levers.dsa_idx_proj_whole;
        for (weight, out) in [
            (self.weights.wk, w.stage_k),
            (self.weights.compress_gate, w.stage_gate),
        ] {
            // 2026-10-12: Under `METRALE_GLM_DSA_IDX_PROJ_WHOLE=1` a projection the FP8 registry
            // serves (and the W4A16 one does not) runs all `n` rows in one call: the W8A8 family
            // is row-invariant, so each row keeps its bits, and the two projections of `hidden`
            // quantize it once per W8A8 launch instead of once per 16-row group each.
            let chunks = if whole_rows_apply(
                whole,
                crate::glm5next_fp8_dense::holds(weight),
                crate::glm5next_w4a16_dense::footprint(weight).is_some(),
            ) {
                vec![(0, n)]
            } else {
                crate::glm5next_layer::multi_seq_chunks(n, group)
            };
            for (r0, m) in chunks {
                self.gemm(
                    gpu,
                    self.kernels.gemm,
                    self.kernels.gemv,
                    self.kernels.batchm_bf16(),
                    hidden.offset(r0 * h * 2),
                    weight,
                    out.offset(r0 * d * 2),
                    m,
                    d,
                    h,
                    stream,
                )?;
            }
        }
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([n as u32, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(w.stage_k)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(n as u32)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)
    }
}

/// 2026-10-12: Whether `indexer_project_rows` runs a projection over all rows at once: the
/// lever on, the FP8 registry serving it, and no W4A16 copy (whose launch it would be).
fn whole_rows_apply(lever: bool, fp8: bool, w4a16: bool) -> bool {
    lever && fp8 && !w4a16
}

#[cfg(test)]
mod whole_rows_tests {
    use super::whole_rows_apply;

    /// 2026-10-12: Only an FP8-served projection without a W4A16 copy, under the lever.
    #[test]
    fn only_fp8_served_projections_run_whole() {
        assert!(whole_rows_apply(true, true, false));
        assert!(!whole_rows_apply(false, true, false));
        assert!(!whole_rows_apply(true, false, false));
        assert!(!whole_rows_apply(true, true, true));
    }
}
