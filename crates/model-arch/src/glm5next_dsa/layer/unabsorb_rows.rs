// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The unabsorbed-MLA prefill's two halves (`METRALE_GLM_MLA_UNABSORB`,
//! `glm5next_dsa::unabsorb`): `q_b_proj` then `glm5next_mla_head_bmm_q` into `q_abs`, and
//! `glm5next_mla_head_bmm_o` then `o_proj` into `hidden`, in place of `q_absorb` and `o_absorb`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Taken only by a pass the caller states is a prefill and only when the layer holds
//!   unabsorbed weights ([`Glm5NextDsaLayer::prefill_unabsorb`]); decode and verify passes keep
//!   the absorbed launches unchanged.
//! - Both halves refuse more rows than the scratch holds before their first launch.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::unabsorb::lever::{HEAD_DIM, O_PROJ_SITE, Q_B_SITE};
use super::super::unabsorb::{HeadBmm, UnabsorbWeights, launch_head_bmm};
use super::{Glm5NextDsaLayer, gemm};

impl Glm5NextDsaLayer {
    /// 2026-10-15: The unabsorbed weights a pass runs on: `Some` only for a prefill pass of a
    /// layer that holds them.
    pub(super) fn prefill_unabsorb(&self, is_prefill: bool) -> Option<&UnabsorbWeights> {
        self.weights.unabsorb.as_ref().filter(|_| is_prefill)
    }

    /// 2026-10-15: The unabsorbed prefill's dense projections, `(weight, n, k, name)` as
    /// [`Self::dense_projections`] lists, for `--dsa-quantization` under
    /// `METRALE_GLM_MLA_UNABSORB=dsa` (the DSA site tier table decides); empty without the
    /// lever. Apart from `dense_projections` so the decode-side consumers of that list (the L2
    /// warm plan) do not see them.
    pub fn unabsorb_projections(&self) -> Vec<(DevicePtr, usize, usize, &'static str)> {
        let Some(u) = &self.weights.unabsorb else {
            return Vec::new();
        };
        let (c, d) = (&self.cfg, self.cfg.local_heads * HEAD_DIM);
        vec![
            (u.q_b, d, c.q_lora_rank, Q_B_SITE),
            (u.o_proj, c.hidden, d, O_PROJ_SITE),
        ]
    }

    fn check_mid(&self, u: &UnabsorbWeights, k: usize) -> Result<()> {
        ensure!(
            (1..=u.mid_rows).contains(&k),
            "DSA layer {}: {k} unabsorbed prefill rows for a scratch of {}",
            self.layer_idx,
            u.mid_rows
        );
        Ok(())
    }

    /// 2026-10-15: `q_b_proj` over the `k` rows of `q_resid` into the scratch, then the
    /// per-head `W_UK^T` into `q_abs`: the latent query `q_absorb` would have written.
    pub(super) fn unabsorbed_q(
        &self,
        gpu: &dyn GpuBackend,
        u: &UnabsorbWeights,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        self.check_mid(u, k)?;
        let (c, w) = (&self.cfg, &self.workspace);
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.batchm_bf16(),
            w.q_resid,
            u.q_b,
            u.mid,
            k,
            c.local_heads * HEAD_DIM,
            c.q_lora_rank,
            stream,
        )?;
        launch_head_bmm(
            gpu,
            &u.kernels,
            HeadBmm::Q,
            u.mid,
            u.w_uk_t,
            w.q_abs,
            k,
            c.local_heads,
            u.mid_rows,
            stream,
        )
    }

    /// 2026-10-15: The per-head `W_UV` over the `k` rows of `attn_out` into the scratch, then
    /// `o_proj` into `hidden`: what `o_absorb` would have written.
    pub(super) fn unabsorbed_o(
        &self,
        gpu: &dyn GpuBackend,
        u: &UnabsorbWeights,
        hidden: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        self.check_mid(u, k)?;
        let c = &self.cfg;
        launch_head_bmm(
            gpu,
            &u.kernels,
            HeadBmm::O,
            self.workspace.attn_out,
            u.w_uv,
            u.mid,
            k,
            c.local_heads,
            u.mid_rows,
            stream,
        )?;
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.batchm_bf16(),
            u.mid,
            u.o_proj,
            hidden,
            k,
            c.hidden,
            c.local_heads * HEAD_DIM,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "unabsorb_rows_tests.rs"]
mod tests;
