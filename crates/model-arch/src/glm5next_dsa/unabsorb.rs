// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: MLA-UNABSORB (`METRALE_GLM_MLA_UNABSORB`, default off): a DSA prefill computes its
//! latent query and its output without the absorbed weights.
//!
//! ```text
//! absorbed (decode, and prefill with the lever off):
//!   q_resid ─ q_absorb [lh*512, q_lora] ─────────────────────────── q_abs  [k, lh*512]
//!   attn_out [k, lh*512] ─ o_absorb [hidden, lh*512] ────────────── hidden
//! unabsorbed (prefill with the lever on):
//!   q_resid ─ q_b_proj [lh*256, q_lora] ─ mid ─ head_bmm_q (W_UK^T) ─ q_abs
//!   attn_out ─ head_bmm_o (W_UV) ─ mid ─ o_proj [hidden, lh*256] ──── hidden
//! ```
//!
//! The absorbed products cost `lh*512` per head on both sides; unabsorbed, the dense
//! projections run at `lh*256` and the per-head `kv_b_proj` halves (`glm5next_mla_head_bmm`)
//! add `256 x 512` per head and row: about 44% fewer o-side and 33% fewer q-side FLOPs. The
//! attend kernel, the latent cache and decode are unchanged; decode keeps the absorbed weights.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Off builds nothing, allocates nothing and launches what it launched before the lever.
//! - On, the unabsorbed weights are checkpoint values moved, never products (`host`), and the
//!   scratch `mid` (`[rows, lh*256]` BF16) is the lever's own allocation, made at load: one for
//!   the stack when the DSA workspace is shared (`layer::dsa_ws_shared`), else one per layer.
//! - Only a pass its caller states is a prefill (`decode_k`'s `is_prefill`) takes the unabsorbed
//!   path; the choice never depends on the row count, so a row's bits do not depend on how the
//!   prompt was cut into passes beyond what the shared GEMM dispatch already does.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextDsaConfig;
use super::build::{LoadFn, build_dsa_weights, up_bf16};
use super::layer::Glm5NextDsaWeights;
use super::tp::DsaTpPlan;
use crate::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer};

pub mod host;
pub mod launch;
pub mod lever;

pub use launch::{HeadBmm, HeadBmmKernels, head_bmm_grid, launch_head_bmm};
pub use lever::{MlaUnabsorb, SiteRule, check_cfg, mode, site_rule};

/// 2026-10-15: One DSA layer's unabsorbed-prefill weights and the scratch they run through.
pub struct UnabsorbWeights {
    pub mode: MlaUnabsorb,
    /// 2026-10-15: `[local_heads * 256, q_lora_rank]` BF16, or the key a `--dsa-quantization`
    /// registry put in its place under `dsa`.
    pub q_b: DevicePtr,
    /// 2026-10-15: `[hidden, local_heads * 256]` BF16 (row-parallel, the caller all-reduces), or a
    /// registry key under `dsa`.
    pub o_proj: DevicePtr,
    /// 2026-10-15: `[local_heads, 512, 256]` BF16, `kv_b_proj`'s K half per head, transposed.
    pub w_uk_t: DevicePtr,
    /// 2026-10-15: `[local_heads, 256, 512]` BF16, `kv_b_proj`'s V half per head.
    pub w_uv: DevicePtr,
    /// 2026-10-15: `[mid_rows, local_heads * 256]` BF16 scratch: q_b's output, then head_bmm_o's.
    pub mid: DevicePtr,
    pub mid_rows: usize,
    pub kernels: HeadBmmKernels,
}

/// 2026-10-15: Bytes of the scratch for `rows` rows.
pub fn mid_bytes(cfg: &Glm5NextDsaConfig, rows: usize) -> usize {
    rows * cfg.local_heads * lever::HEAD_DIM * 2
}

/// 2026-10-15: Bytes of one layer's unabsorbed weights on this rank (BF16).
pub fn weight_bytes(cfg: &Glm5NextDsaConfig) -> usize {
    let lh = cfg.local_heads;
    let (d, l) = (lever::HEAD_DIM, lever::LATENT);
    2 * (lh * d * cfg.q_lora_rank + cfg.hidden * lh * d + 2 * lh * l * d)
}

/// 2026-10-15: The text stack's lever state, made once before the layer loop: the mode, the
/// kernels and (shared workspace) the one scratch.
pub struct TextUnabsorb {
    mode: MlaUnabsorb,
    kernels: Option<HeadBmmKernels>,
    shared_mid: Option<DevicePtr>,
    rows: usize,
}

impl TextUnabsorb {
    /// 2026-10-15: The loader's one call: [`Self::new`] in [`mode`] for `skeleton`'s DSA layers,
    /// the scratch one for the stack when `shared` (the shared DSA workspace) is `Some`.
    pub fn for_stack<T>(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        skeleton: &Glm5NextTextSkeleton,
        rows: usize,
        shared: &Option<T>,
    ) -> Result<Self> {
        let dsa_layers = skeleton
            .layers
            .iter()
            .filter(|l| l.mixer == Mixer::Dsa)
            .count();
        Self::new(gpu, cfg, rows, shared.is_some(), dsa_layers, mode()?)
    }

    /// 2026-10-15: One DSA layer's weights: `build::build_dsa_weights`, with
    /// [`Self::weights`] in `unabsorb`.
    pub fn dsa_weights(
        &self,
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        plan: &DsaTpPlan,
        load: LoadFn<'_>,
    ) -> Result<Glm5NextDsaWeights> {
        let mut w = build_dsa_weights(gpu, cfg, plan, load)?;
        w.unabsorb = self.weights(gpu, cfg, plan, load)?;
        Ok(w)
    }

    /// 2026-10-15: The lever in `mode`; on, check the geometry,
    /// resolve the kernels, allocate the shared scratch for `rows` rows (`shared`) and log what
    /// is active and what it costs. Off touches nothing. The loader calls [`Self::for_stack`].
    pub fn new(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        rows: usize,
        shared: bool,
        dsa_layers: usize,
        mode: MlaUnabsorb,
    ) -> Result<Self> {
        if !mode.is_on() {
            return Ok(Self {
                mode,
                kernels: None,
                shared_mid: None,
                rows,
            });
        }
        check_cfg(cfg)?;
        let kernels = HeadBmmKernels::resolve(gpu)?;
        let scratch = mid_bytes(cfg, rows);
        let shared_mid = shared.then(|| gpu.alloc(scratch)).transpose()?;
        let scratch_total = scratch * if shared { 1 } else { dsa_layers };
        let precision = match mode {
            MlaUnabsorb::Declared => {
                "follow the checkpoint-declared precision (BF16), whatever --dsa-quantization is"
                    .to_string()
            }
            _ => format!(
                "follow --dsa-quantization {:?}",
                metrale_model_layers::layers::dsa_quantization()
            ),
        };
        tracing::info!(
            "{}={}: DSA prefill runs unabsorbed MLA (q_b_proj, per-head kv_b_proj halves on \
             glm5next_mla_head_bmm, o_proj); decode stays absorbed; q_b_proj/o_proj {precision}; \
             +{:.1} MB per rank: scratch {:.1} MB ({rows} rows, {}) + weights {:.1} MB x {dsa_layers} \
             DSA layers (BF16 before any registry)",
            lever::ENV,
            mode.name(),
            (scratch_total + weight_bytes(cfg) * dsa_layers) as f64 / 1e6,
            scratch_total as f64 / 1e6,
            if shared { "shared" } else { "per layer" },
            weight_bytes(cfg) as f64 / 1e6,
        );
        Ok(Self {
            mode,
            kernels: Some(kernels),
            shared_mid,
            rows,
        })
    }

    /// 2026-10-15: One DSA layer's [`UnabsorbWeights`] on this rank when the lever is on (`None`
    /// when off; the layer's `Glm5NextDsaWeights::unabsorb`). `load` as for
    /// `build::build_dsa_weights`.
    pub fn weights(
        &self,
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        plan: &DsaTpPlan,
        load: LoadFn<'_>,
    ) -> Result<Option<UnabsorbWeights>> {
        let Some(kernels) = self.kernels else {
            return Ok(None);
        };
        let h = host::unabsorb_for_rank(cfg, plan, load)?;
        let mid = match self.shared_mid {
            Some(p) => p,
            None => gpu.alloc(mid_bytes(cfg, self.rows))?,
        };
        Ok(Some(UnabsorbWeights {
            mode: self.mode,
            q_b: up_bf16(gpu, &h.q_b)?,
            o_proj: up_bf16(gpu, &h.o_proj)?,
            w_uk_t: up_bf16(gpu, &h.w_uk_t)?,
            w_uv: up_bf16(gpu, &h.w_uv)?,
            mid,
            mid_rows: self.rows,
            kernels,
        }))
    }
}

#[cfg(test)]
#[path = "unabsorb_tests.rs"]
mod tests;
