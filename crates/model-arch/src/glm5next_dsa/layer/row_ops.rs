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

use super::{Glm5NextDsaLayer, gemm};

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
        gemm(
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
        gemm(
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
        gemm(
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
        let t_proj = profile::start();
        gemm(
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
        for (weight, out) in [
            (self.weights.wk, w.stage_k),
            (self.weights.compress_gate, w.stage_gate),
        ] {
            for (r0, m) in crate::glm5next_layer::multi_seq_chunks(n, group) {
                gemm(
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
