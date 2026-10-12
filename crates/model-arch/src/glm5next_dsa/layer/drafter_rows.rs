// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The MTP drafter's context rows in one pass: `m` consecutive rows' KV latents and
//! indexer rows, from `m` normed input rows, on the drafter's flat indexer cache. The one-row
//! form is `write_kv_row`; this one runs each projection once over all rows (`glm_mm` picks
//! the batched GEMV, the tile GEMM or cuBLASLt by row count) instead of once per row.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Writes nothing when the indexer cache is behind `seq_len` (rows must be dense from 0), as
//!   `write_kv_row`.
//! - Row `r` goes to KV slot `slots[r]` and to indexer row `seq_len + r`.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::{ForwardContext, LayerState};

use super::super::paged::IndexerCache;
use super::super::state::Glm5NextDsaState;
use super::{Glm5NextDsaLayer, gemm};

impl Glm5NextDsaLayer {
    /// 2026-10-11: The widest batch `write_kv_rows_flat` takes: the workspace's rows (its
    /// `kv_a` staging).
    pub fn drafter_rows_max(&self) -> usize {
        self.workspace.max_rows
    }

    /// 2026-10-11: `m` MTP drafter context rows from `normed` (`[m, hidden]` BF16, the rows
    /// after `input_norm`): the KV latents (`kv_a_proj`, then the latent write at
    /// `slots_dev[r]`, `m` i64 slots) and the indexer rows (`wk` and the key norm,
    /// `compress_gate`) written straight into the flat cache rows `seq_len..seq_len + m`, which
    /// are marked valid. Errors on a paged indexer cache or more rows than
    /// [`Self::drafter_rows_max`].
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_rows_flat(
        &self,
        normed: DevicePtr,
        m: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        slots_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        if m == 0 || m > self.workspace.max_rows || st.cache() != IndexerCache::Flat {
            bail!(
                "DSA layer {}: {m} drafter context rows for a {}-row workspace on a {:?} \
                 indexer cache (one pass needs a flat cache and at most the workspace's rows)",
                self.layer_idx,
                self.workspace.max_rows,
                st.cache()
            );
        }
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} rows but the drafter is at {seq_len} — \
                 rows are MISSING, not merely stale.",
                self.layer_idx,
                st.len()
            ),
            std::cmp::Ordering::Equal => {}
        }
        st.ensure_room(m)?;
        let (k, c) = (&self.kernels, &self.cfg);
        let proj = |a: DevicePtr, w: DevicePtr, out: DevicePtr, n: usize| {
            gemm(
                gpu,
                k.gemm,
                k.gemv,
                k.batchm_bf16(),
                a,
                w,
                out,
                m,
                n,
                c.hidden,
                stream,
            )
        };
        proj(
            normed,
            self.weights.kv_a_proj,
            self.workspace.kv_a,
            c.kv_lora_rank,
        )?;
        self.write_latent_rows(gpu, 0, m, kv_cache, slots_dev, stream)?;
        let d = c.index_head_dim;
        let off = st.row_offset(seq_len);
        let (k_rows, gate_rows) = (st.k_normed.offset(off), st.gate.offset(off));
        proj(normed, self.weights.wk, k_rows, d)?;
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([m as u32, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(k_rows)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(m as u32)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        proj(normed, self.weights.compress_gate, gate_rows, d)?;
        gpu.memset_async(st.valid.offset(seq_len), 1, m, stream)?;
        st.advance(m)
    }
}

impl Glm5NextDsaLayer {
    /// 2026-10-11: `out[r] = a[r] @ w^T` for `m` rows, `n` outputs over this layer's hidden
    /// width (`[m, hidden]` BF16 in, `[m, n]` BF16 out), through this layer's projection
    /// dispatch (`glm_mm`). For a caller outside the layer that projects over the same width
    /// (the drafter's `eh_proj` takes `kk = 2 * hidden`, so `kk` is explicit).
    #[allow(clippy::too_many_arguments)]
    pub fn proj_rows(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        a: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        let k = &self.kernels;
        gemm(
            gpu,
            k.gemm,
            k.gemv,
            k.batchm_bf16(),
            a,
            w,
            out,
            m,
            n,
            kk,
            stream,
        )
    }
}
