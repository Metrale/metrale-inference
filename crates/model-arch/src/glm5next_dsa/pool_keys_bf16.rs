// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The DSA kernels under the load's persisted-key levers:
//! `METRALE_GLM_DSA_POOL_KEYS_BF16` (the persisted pool keys stored and scored as BF16) and
//! `METRALE_GLM_DSA_DECODE_SCORES_MMA` (those keys scored on BF16 tensor cores).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: resolved once per load (ROTI); a lever whose kernels or prerequisites are
//! missing refuses the load, never a guessed fallback (PCND).

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;

use super::{DsaRowsKernels, Glm5NextDsaKernels};

/// 2026-10-23: Module of the persisted keys' BF16 store and scores: `dsa_scores_heads_bf16.cu`.
pub const DSA_SCORES_HEADS_BF16_MODULE: &str = "dsa_scores_heads_bf16";

/// 2026-10-23: Module of the BF16 persisted keys' tensor-core scores: `dsa_scores_mma_paged.cu`.
pub const DSA_SCORES_MMA_PAGED_MODULE: &str = "dsa_scores_mma_paged";

impl DsaRowsKernels {
    /// 2026-10-23: These kernels with the persisted keys stored and scored as BF16
    /// ([`Self::pool_keys_bf16`]); refused unless both lane-per-head walks resolved (the only
    /// readers with a BF16-key instance) and the BF16-key module is in the build.
    pub fn with_bf16_pool_keys(mut self, gpu: &dyn GpuBackend) -> Result<Self> {
        ensure!(
            self.scores_heads_paged.0 != 0 && self.scores_heads_window.0 != 0,
            "METRALE_GLM_DSA_POOL_KEYS_BF16: needs dsa_index_scores_heads_paged and \
             dsa_index_scores_heads_window (the BF16 keys' only readers)"
        );
        let k = |name: &str| {
            gpu.kernel(DSA_SCORES_HEADS_BF16_MODULE, name)
                .with_context(|| format!("METRALE_GLM_DSA_POOL_KEYS_BF16: {name}"))
        };
        self.pool_keys_store = k("dsa_pool_keys_store_rows_bf16")?;
        self.scores_heads_paged = k("dsa_index_scores_heads_paged_bf16")?;
        self.scores_heads_window = k("dsa_index_scores_heads_window_bf16")?;
        self.scores_mma_paged = gpu
            .kernel(
                DSA_SCORES_MMA_PAGED_MODULE,
                "dsa_index_scores_mma_paged_bf16",
            )
            .context("METRALE_GLM_DSA_POOL_KEYS_BF16: dsa_index_scores_mma_paged_bf16")?;
        self.pool_keys_bf16 = true;
        Ok(self)
    }
}

impl DsaRowsKernels {
    /// 2026-10-23: These BF16-key kernels with [`Self::scores_mma_paged`] as both lane-per-head
    /// handles (`METRALE_GLM_DSA_DECODE_SCORES_MMA`, the paired A/B's `dsa_mma` arm).
    pub fn with_scores_mma(mut self) -> Self {
        let mma = self.scores_mma_paged;
        (self.scores_heads_paged, self.scores_heads_window) = (mma, mma);
        self
    }
}

impl Glm5NextDsaKernels {
    /// 2026-10-23: [`Self::resolve`] under the load's `METRALE_GLM_DSA_POOL_KEYS_BF16`
    /// ([`DsaRowsKernels::with_bf16_pool_keys`]); refused with either lane-per-head lever off.
    pub fn resolve_for(
        gpu: &dyn GpuBackend,
        s: &metrale_settings::InferenceServerSettings,
    ) -> Result<Self> {
        let mut k = Self::resolve(gpu)?;
        let l = &s.kernels;
        if l.glm_dsa_pool_keys_bf16 {
            ensure!(
                l.glm_dsa_scores_heads && l.glm_dsa_scores_heads_window,
                "METRALE_GLM_DSA_POOL_KEYS_BF16 needs METRALE_GLM_DSA_SCORES_HEADS and \
                 METRALE_GLM_DSA_SCORES_HEADS_WINDOW on"
            );
            k.rows = k.rows.with_bf16_pool_keys(gpu)?;
            tracing::info!("GLM DSA persisted pool keys: BF16 (METRALE_GLM_DSA_POOL_KEYS_BF16)");
        }
        if l.glm_dsa_decode_scores_mma {
            ensure!(
                k.rows.pool_keys_bf16,
                "METRALE_GLM_DSA_DECODE_SCORES_MMA needs METRALE_GLM_DSA_POOL_KEYS_BF16"
            );
            // 2026-10-23: Both lane-per-head handles: one launch contract, any row count.
            k.rows = k.rows.with_scores_mma();
            tracing::info!(
                "GLM DSA decode scores: BF16 tensor cores (dsa_index_scores_mma_paged_bf16)"
            );
        }
        // 2026-10-24: R4, a command-line lever (not a setting): the CLI's published value.
        if crate::glm5next_decode_levers::glm_decode_levers().topk_seg {
            k.rows = k.rows.with_topk_seg()?;
            tracing::info!("GLM DSA multi-CTA top-k: one persistent segment (--glm-dsa-topk-seg)");
        }
        Ok(k)
    }
}
