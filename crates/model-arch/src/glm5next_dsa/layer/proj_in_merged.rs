// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_DSA_PROJ_IN_MERGED` (default on): a DSA decode or verify's
//! projections of the layer input `hidden` (q_a, kv_a, and, where the indexer's projections run
//! in the same pass, wk and the compress gate) as one quantization of `hidden` and one
//! `w8a8_gemv_rowscale_mb1_ku8_split4` launch (`glm5next_fp8_dense::proj_split`), instead of a
//! launch each and a second quantization of `hidden` after q_absorb's. Each output is the bytes
//! its own launch writes: the same tiles on the same quantized activation
//! (`examples/glm5next_dsa_proj_in_gate`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Taken only when every projection is served by the FP8 registry and none by the W4A16 one
//!   (which `glm_mm` would launch first), for 1..=8 rows; otherwise `project_in` and
//!   `indexer_project_rows` launch each projection as before.
//! - The indexer's outputs are its staging rows (`stage_k`, `stage_gate`), so with them the rows
//!   must fit the workspace, as `indexer_project_rows` requires.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextDsaLayer;
use crate::glm5next_fp8_dense::{SplitProj, proj_split, split_ready};

impl Glm5NextDsaLayer {
    /// 2026-10-21: The projections the merged launch takes: q_a and kv_a, then (`idx`) wk and
    /// the compress gate.
    fn merged_in_projs(&self, idx: bool) -> Vec<SplitProj> {
        let w = &self.workspace;
        let d = self.cfg.index_head_dim;
        let mut p = vec![
            SplitProj {
                b: self.weights.q_a_proj,
                c: w.q_a,
                n: self.cfg.q_lora_rank,
            },
            SplitProj {
                b: self.weights.kv_a_proj,
                c: w.kv_a,
                n: self.cfg.kv_lora_rank,
            },
        ];
        if idx {
            p.push(SplitProj {
                b: self.weights.wk,
                c: w.stage_k,
                n: d,
            });
            p.push(SplitProj {
                b: self.weights.compress_gate,
                c: w.stage_gate,
                n: d,
            });
        }
        p
    }

    /// 2026-10-21: Whether `project_in` runs [`Self::merged_in_projs`] of `k` rows as one launch:
    /// the lever on, the rows fit (with `idx`, the indexer's staging rows), no projection served
    /// by the W4A16 registry, and the FP8 registry's split plan served. Nothing is launched.
    pub fn proj_in_merged_ready(&self, k: usize, idx: bool) -> Result<bool> {
        if !self.levers.dsa_proj_in_merged || (idx && k > self.workspace.max_rows) {
            return Ok(false);
        }
        let projs = self.merged_in_projs(idx);
        if projs
            .iter()
            .any(|p| crate::glm5next_w4a16_dense::footprint(p.b).is_some())
        {
            return Ok(false);
        }
        split_ready(k, self.cfg.hidden, &projs)
    }

    /// 2026-10-21: Run [`Self::merged_in_projs`] of the `k` rows of `hidden` as one launch where
    /// [`Self::proj_in_merged_ready`]; `Ok(false)`, with nothing launched, otherwise.
    pub(super) fn project_in_merged(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        idx: bool,
        stream: u64,
    ) -> Result<bool> {
        if !self.proj_in_merged_ready(k, idx)? {
            return Ok(false);
        }
        let projs = self.merged_in_projs(idx);
        anyhow::ensure!(
            proj_split(gpu, hidden, k, self.cfg.hidden, &projs, stream)?,
            "DSA layer {}: the merged input projections were ready but did not run",
            self.layer_idx
        );
        Ok(true)
    }
}
