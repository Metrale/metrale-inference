// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `Glm5NextDsaLayer::decode_k_plan`: a captured decode row or a captured verify's
//! rows from a [`DsaPlan`], in `decode_k_with`'s order on that path (with metadata, under
//! capture: every row placed on the device, the selection a ceiling pass over the persisted
//! pool keys). The plan's steps are the DSA mixer's own kernels; the projections are the layer's
//! own code, called as `decode_k_with` calls them.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Reads no DSA lever: every own kernel, the attend's split count and the selection's ceiling
//!   are the plan's. Where `decode_k_with` would take another arm (no metadata, an FP8 ring, a
//!   verify whose projections or selector query did not run merged), this is an error naming
//!   it, never that arm.
//! - The profile buckets are `decode_k_with`'s, closed at the same points.

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layer::{ForwardContext, LayerState};

use super::super::attend::{DsaDecodePaging, Glm5NextDsaDecodeKernel, HbDiagnostics};
use super::super::plan::{DsaPlan, KNormStep, plan_rows};
use super::super::select::{DsaSelectInputs, GEOM_SLOT_BYTES, select_planned};
use super::super::state::Glm5NextDsaState;
use super::{Glm5NextDsaLayer, IndexerKernels, IndexerPlace};
use crate::glm5next_layer::profile;

impl Glm5NextDsaLayer {
    /// 2026-10-23: `k` rows of one sequence from `plan`, as [`Self::decode_k`] runs them on a
    /// captured pass with row metadata (`plan::plan_rows`); an error on any other pass.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k_plan(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        ctx: &ForwardContext,
        is_prefill: bool,
        plan: &DsaPlan,
        stream: u64,
    ) -> Result<()> {
        let s = &plan.steps;
        ensure!(
            k == s.rows && (1..=self.workspace.max_rows).contains(&k),
            "DSA layer {}: a plan for {} rows run at {k} (workspace {})",
            self.layer_idx,
            s.rows,
            self.workspace.max_rows
        );
        let m = plan_rows(ctx, k, is_prefill).with_context(|| {
            format!(
                "DSA layer {}: a plan runs a captured pass with metadata for its {k} rows",
                self.layer_idx
            )
        })?;
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .context("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")?;
        self.check_lockstep(st, seq_len)?;
        ensure!(
            st.fp8_ring().is_none(),
            "DSA layer {}: the plan places BF16 indexer rows; this state keeps an FP8 ring",
            self.layer_idx
        );
        let gpu = ctx.gpu;
        let w = &self.workspace;
        let t_proj = self.levers.profile.start();
        let stable = crate::glm5next_fp8_dense::stable_input(hidden);
        let own = IndexerKernels {
            k_norm: match s.k_norm {
                KNormStep::Row(n) | KNormStep::Rows(n) => n,
            },
            store: s.store,
            pool_keys: s.pool_keys,
        };
        let verify = matches!(s.k_norm, KNormStep::Rows(_));
        ensure!(
            verify == (k > 1),
            "DSA layer {}: a {k}-row plan with a {:?} key norm",
            self.layer_idx,
            s.k_norm
        );
        let projected = self.project_in(gpu, hidden, k, verify, stream)?;
        if verify {
            ensure!(
                projected && self.selector_pair(gpu, hidden, k, stream)?,
                "DSA layer {}: the plan's verify takes the indexer projections and the selector \
                 query of its {k} rows merged; the dense tier ran them apart",
                self.layer_idx
            );
            self.k_norm_rows(gpu, own.k_norm, w.stage_k, k, stream)?;
        }
        let bt_stride = m.max_blocks_per_seq as usize * 4;
        for row in 0..k {
            let slot = m.slot.offset(row * 8);
            self.write_latent_rows(gpu, s.latent_write, row, 1, kv_cache, slot, stream)?;
            profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
            let t = self.levers.profile.start();
            let place = IndexerPlace::Device {
                pos: m.positions.offset(row * 4),
                bt: m.block_table.offset(row * bt_stride),
            };
            if verify {
                st.ensure_room(1)?;
                let pos = st.len();
                self.store_indexer_row(gpu, st, kv_cache, place, pos, row, own, stream)?;
            } else {
                self.indexer_forward_with(gpu, hidden, st, kv_cache, place, true, own, stream)?;
            }
            profile::end(profile::DSA_INDEXER, t, gpu, stream);
            let (sl, geom) = match verify {
                true => (
                    m.seq_len.offset(row * 4),
                    w.geom_rows.offset(row * GEOM_SLOT_BYTES),
                ),
                false => (m.seq_len, w.geom_dev),
            };
            self.write_geom(gpu, s.write_geom, sl, geom, stream)?;
            if !verify {
                let (q, weights) = self.selector_query_row(gpu, row, stream)?;
                let at = (m.positions, m.block_table);
                self.select_plan(gpu, plan, st, kv_cache, 1, (q, weights), at, stream)?;
            }
        }
        if verify {
            let pre = (w.q_idx_rows, w.head_weights_rows);
            let at = (m.positions, m.block_table);
            self.select_plan(gpu, plan, st, kv_cache, k, pre, at, stream)?;
        }
        let paging = DsaDecodePaging {
            num_seqs: 1,
            num_q_heads: self.cfg.local_heads,
            num_kv_heads: 1,
            max_blocks_per_seq: m.max_blocks_per_seq as usize,
            block_size: kv_cache.config().block_size,
            cache_stride_bytes: (kv_cache.config().block_size * self.cfg.kv_lora_rank) as u64,
        };
        let kernel = Glm5NextDsaDecodeKernel::from_plan(s.attend)?;
        let none = HbDiagnostics {
            check_dir: None,
            serve_ref: false,
        };
        let (bt, sl) = (m.block_table, m.seq_len);
        self.attend_rows_with(gpu, kernel, none, k, st, kv_cache, bt, sl, &paging, stream)?;
        drop(stable);
        self.project_out(gpu, hidden, k, stream)
    }

    /// 2026-10-23: The plan's selection of `rows` rows through the block table `bt` (one decode
    /// row into its row of the output, or a verify's rows in one pass), from the selector queries
    /// and head weights `(q, weights)` at the metadata's positions `q_pos`, each row's geometry at
    /// its slot (`geom_dev` for one row, `geom_rows` for a verify): `select_row` and
    /// `select_verify_rows` on their replay-safe paths.
    #[allow(clippy::too_many_arguments)]
    fn select_plan(
        &self,
        gpu: &dyn GpuBackend,
        plan: &DsaPlan,
        st: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        rows: usize,
        (q, weights): (DevicePtr, DevicePtr),
        (q_pos, bt): (DevicePtr, DevicePtr),
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let geom = st.geometry(&self.cfg, rows)?;
        let one_row;
        let (geom_dev, scratch) = if rows == 1 {
            one_row = w.select.row(0, &self.cfg);
            (w.geom_dev, &one_row)
        } else {
            (w.geom_rows, &w.select)
        };
        let inputs = DsaSelectInputs {
            rows: self.indexer_rows(st, kv_cache, bt)?,
            ape: self.weights.ape,
            q,
            weights,
            q_pos,
            q_mask: w.q_mask,
            first_key: 0,
            geom_dev,
        };
        let t = self.levers.profile.start();
        select_planned(
            gpu,
            &self.cfg,
            &geom,
            &inputs,
            scratch,
            &plan.select(),
            stream,
        )?;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        Ok(())
    }
}
