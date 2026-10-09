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
            self.kernels.gemv_batchm,
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
            self.kernels.gemv_batchm,
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
            self.kernels.gemv_batchm,
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
            self.kernels.gemv_batchm,
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
}
