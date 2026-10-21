// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: One selection pass for a captured verify's rows
//! (`METRALE_GLM_DSA_VERIFY_SELECT_ONCE`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Runs after every row's indexer row (and the pool key it completes) is stored; row `r`
//!   only takes pools ending at or before `q_pos[r]`, so the rows stored after it do not
//!   change its selection.
//! - Row `r`'s geometry is at `geom_rows + r * GEOM_SLOT_BYTES`, written by `write_geom` in the
//!   row loop of `decode_k_with`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::paged::IndexerRowsDev;
use super::super::select::{DsaSelectInputs, DsaSelectLaunch, contiguous_pool_count, select_tokens};
use super::super::state::{Glm5NextDsaState, max_dsa_context};
use super::Glm5NextDsaLayer;

impl Glm5NextDsaLayer {
    /// 2026-10-21: The ceiling launch of a captured decode or verify pass.
    pub(super) fn ceiling_launch(&self) -> DsaSelectLaunch {
        DsaSelectLaunch::Ceiling {
            max_pools: contiguous_pool_count(self.cfg.index_kpool, max_dsa_context(&self.cfg)),
        }
    }

    /// 2026-10-21: The inputs of the verify rows' pass: the rows' precomputed selector queries
    /// and head weights (`METRALE_GLM_DSA_VERIFY_ROWS`), `q_pos` the metadata's positions and
    /// the per-row geometry.
    fn verify_select_inputs(&self, rows: IndexerRowsDev, q_pos: DevicePtr) -> DsaSelectInputs {
        let w = &self.workspace;
        DsaSelectInputs {
            rows,
            ape: self.weights.ape,
            q: w.q_idx_rows,
            weights: w.head_weights_rows,
            q_pos,
            // 2026-10-21: `dsa_expand_selection_rows` reads `q_mask[0]` for every row.
            q_mask: w.q_mask,
            first_key: 0,
            geom_dev: w.geom_rows,
        }
    }

    /// 2026-10-21: Whether a captured `k`-row verify takes [`Self::select_verify_rows`]: the
    /// lever on and the pass's kernels, cache format and scratch able to run it
    /// (`select::rows_ceiling_ready`). Otherwise the rows are selected one pass each, as before.
    pub fn verify_select_once_for(
        &self,
        st: &Glm5NextDsaState,
        rows: IndexerRowsDev,
        k: usize,
    ) -> Result<bool> {
        if !self.levers.dsa_verify_select_once {
            return Ok(false);
        }
        let DsaSelectLaunch::Ceiling { max_pools } = self.ceiling_launch() else {
            return Ok(false);
        };
        let geom = st.geometry(&self.cfg, k)?;
        let inputs = self.verify_select_inputs(rows, DevicePtr::NULL);
        Ok(super::super::select::rows_ceiling_ready(
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &self.workspace.select,
            max_pools,
        ))
    }

    /// 2026-10-21: Select the `k` verify rows in one ceiling pass into rows `0..k` of the
    /// selection output; `q_pos` is the metadata's `[k]` positions.
    pub(super) fn select_verify_rows(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        st: &Glm5NextDsaState,
        rows: IndexerRowsDev,
        q_pos: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let geom = st.geometry(&self.cfg, k)?;
        let inputs = self.verify_select_inputs(rows, q_pos);
        let t = self.levers.profile.start();
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &self.workspace.select,
            self.ceiling_launch(),
            stream,
        )?;
        use crate::glm5next_layer::profile;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        Ok(())
    }
}
