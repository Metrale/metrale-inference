// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: One draft step of the batched propose (`batch.rs`): the eager body, and the same
//! body captured into a graph and replayed for a batch whose device addresses it has seen.
//!
//! A captured step bakes every address it reads: each row's target hidden, each sequence's
//! flat DSA indexer cache and KV region, and the batch's fixed scratch. The key is those
//! addresses (`BatchGraphKey`), so a graph is replayed only where every address it bakes is
//! the same; the row metadata and the embedding rows are written into fixed buffers before
//! each launch.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - A replay is followed by the host bookkeeping the eager body does (the DSA indexer length,
//!   `seq_len`), as `graph.rs` does for one row.
//! - After a failed capture the batch runs eagerly for the rest of the process.

use super::batch::{BatchScratch, ProposeBatch};
use super::*;
use metrale_gpu_runtime::gpu::GraphHandle;
use metrale_model_layers::layer::{AttnMetadataDev, LayerGraphHooks};

/// 2026-10-11: Captured steps kept per batch; the oldest is destroyed past it.
const GRAPHS_KEPT: usize = 8;

/// 2026-10-11: The device addresses a captured step bakes, per row: the target hidden, the DSA
/// indexer buffers and the KV region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BatchGraphKey(Vec<(u64, u64, u64, u64, Option<usize>)>, bool);

impl BatchGraphKey {
    /// 2026-10-11: `lp`: the step computes log-probabilities (other kernels).
    pub(super) fn of(
        sts: &[&mut Glm5NextMtpProposerState],
        hiddens_in: &[DevicePtr],
        lp: bool,
    ) -> Self {
        Self(
            sts.iter()
                .zip(hiddens_in)
                .map(|(st, h)| {
                    (
                        h.0,
                        st.dsa.k_normed.0,
                        st.dsa.gate.0,
                        st.dsa.valid.0,
                        st.kv_region,
                    )
                })
                .collect(),
            lp,
        )
    }
}

impl Glm5NextMtpHead {
    /// 2026-10-11: Whether draft 0 of a batch runs through a graph: the drafter-graph lever, no
    /// earlier capture failure, no timing arm, and a DSA block whose rows are placed and planned
    /// on the device.
    pub(super) fn batch_graph_eligible(&self, sc: &BatchScratch) -> bool {
        super::graph::graph_enabled()
            && !sc.graph_off
            && !skip_block()
            && !skip_head()
            && match &self.module.layer.mixer {
                crate::glm5next_layer::Glm5NextMixer::Dsa(l) => {
                    l.select_kernels.indexer_store.0 != 0 && l.select_kernels.write_geom.0 != 0
                }
                _ => false,
            }
    }

    /// 2026-10-11: One draft step for every row, from the embedding rows `embeds[i]` and the
    /// hidden rows `hiddens_in[i]` (target rows when `target_rows`) to each row's global draft
    /// id in `sc.arg[i]`: the input norms, one batched `eh_proj`, the block over all rows
    /// (advancing every `seq_len`), the final norm, one head sweep, one argmax, and the device
    /// exchange of every row behind one all-reduce. `bctx` carries the row metadata.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn step_body(
        &self,
        batch: &ProposeBatch,
        sc: &BatchScratch,
        sts: &mut [&mut Glm5NextMtpProposerState],
        embeds: &[Option<DevicePtr>],
        hiddens_in: &[DevicePtr],
        target_rows: bool,
        rows: &[usize],
        lp: Option<DevicePtr>,
        bctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = bctx.gpu;
        let h = self.hidden;
        let m = sts.len();
        let meta = bctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("GLM MTP batched step without row metadata"))?;
        let (n, v0, sharded) = self.head_range(bctx);
        // 2026-10-11: Row i's concat: `enorm(embed[token_i])`, then the hidden half: a target
        // row (the target's final norm under `METRALE_MTP_TARGET_POSTNORM`, staged in row i of
        // `x`, which `eh_proj` writes next) or the drafter's own output, as `propose`.
        for i in 0..m {
            let row = sc.concat.offset(i * 2 * h * 2);
            self.embed_half(gpu, embeds[i], row, stream)?;
            self.hidden_half(
                bctx,
                hiddens_in[i],
                target_rows,
                sc.x.offset(i * h * 2),
                row,
                stream,
            )?;
        }
        // 2026-10-11: `eh_proj` over every row: under `METRALE_GLM_MTP_EH_TP` with a
        // communicator, this rank's output rows into a zeroed `x` and one all-reduce of all the
        // rows, as the one-row path does (`Self::eh_proj`); each output is the same GEMV either
        // way.
        match (bctx.comm, self.eh_shard) {
            (Some(comm), Some((r0, rows_n))) => {
                gpu.memset_async(sc.x, 0, m * h * 2, stream)?;
                ops::dense_gemv_batchm(
                    gpu,
                    batch.kernels.gemv_batchm,
                    sc.concat,
                    &DenseWeight {
                        weight: self.module.eh_proj.weight.offset(r0 * 2 * h * 2),
                    },
                    sc.x.offset(r0 * 2),
                    m as u32,
                    rows_n as u32,
                    (2 * h) as u32,
                    h as u32,
                    stream,
                )?;
                comm.all_reduce_async(sc.x.0, m * h * 2, stream)?;
            }
            _ => ops::dense_gemv_batchm(
                gpu,
                batch.kernels.gemv_batchm,
                sc.concat,
                &self.module.eh_proj,
                sc.x,
                m as u32,
                h as u32,
                (2 * h) as u32,
                h as u32,
                stream,
            )?,
        }
        if !skip_block() {
            let mut kv = self.kv_cache.lock();
            let mut dsa: Vec<&mut (dyn LayerState + 'static)> = sts
                .iter_mut()
                .map(|st| &mut st.dsa as &mut (dyn LayerState + 'static))
                .collect();
            self.module
                .layer
                .decode_rows_for_drafter(sc.x, &mut dsa, &mut kv, rows, &meta, bctx, stream)?;
        }
        for st in sts.iter_mut() {
            st.seq_len += 1;
        }
        self.norm_rows(gpu, sc.x, self.module.final_norm, sc.x, h, m, stream)?;
        match (
            self.head_fp8.filter(|_| sharded && n == self.head_n),
            batch.kernels.fp8_batch16,
        ) {
            (Some(q), k) if k.0 != 0 => ops::fp8_gemv_rowscale_batch16_rt2_ldc(
                gpu, k, sc.x, &q, sc.logits, m as u32, n as u32, h as u32, n as u32, stream,
            )?,
            _ => ops::dense_gemv_batchm(
                gpu,
                batch.kernels.gemv_batchm,
                sc.x,
                &DenseWeight {
                    weight: self.lm_head.weight.offset(v0 * h * 2),
                },
                sc.logits,
                m as u32,
                n as u32,
                h as u32,
                n as u32,
                stream,
            )?,
        }
        // 2026-10-11: With `lp`, row i's log-probability at `lp + 4i` (`draft_lp`).
        match lp {
            Some(lp) => ops::argmax_bf16_batch_lp(
                gpu,
                self.lp_k.argmax_lp,
                sc.logits,
                sc.arg,
                lp,
                n as u32,
                m as u32,
                n as u32,
                stream,
            )?,
            None => ops::argmax_bf16_batch(
                gpu,
                batch.kernels.argmax_batch,
                sc.logits,
                sc.arg,
                n as u32,
                m as u32,
                n as u32,
                stream,
            )?,
        }
        if sharded {
            let (Some(k), Some(comm)) = (self.device_xchg(bctx), bctx.comm) else {
                bail!("GLM MTP batched propose: a sharded head without the device exchange");
            };
            self.exchange_rows(
                gpu,
                comm,
                k,
                sc.arg,
                sc.logits,
                n * 2,
                sc.xchg,
                m,
                lp,
                v0,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-10-11: [`Self::step_body`] for draft 0 through a graph: copy the embedding rows into
    /// `embed_in`, then replay the step captured for this batch's addresses, or capture it. A
    /// replay redoes the host bookkeeping the capture run did. The row metadata is already in
    /// `sc.meta_words`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn step_graphed(
        &self,
        batch: &ProposeBatch,
        sc: &mut BatchScratch,
        sts: &mut [&mut Glm5NextMtpProposerState],
        embeds: &[DevicePtr],
        hiddens_in: &[DevicePtr],
        rows: &[usize],
        lp: Option<DevicePtr>,
        meta: AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let staged: Vec<Option<DevicePtr>> = (0..sts.len())
            .map(|i| Some(sc.embed_in.offset(i * h * 2)))
            .collect();
        for (src, dst) in embeds.iter().zip(staged.iter().flatten()) {
            gpu.copy_d2d_async(*src, *dst, h * 2, stream)?;
        }
        let key = BatchGraphKey::of(sts, hiddens_in, lp.is_some());
        let layer = &self.module.layer;
        if let Some(&(_, graph)) = sc.graphs.iter().find(|(k, _)| *k == key) {
            for st in sts.iter() {
                layer.check_replay_room(&st.dsa, st.seq_len, 1)?;
            }
            gpu.launch_graph(graph, stream)?;
            for st in sts.iter_mut() {
                layer.sync_replayed_step(&mut st.dsa, st.seq_len, 1)?;
                st.seq_len += 1;
            }
            return Ok(());
        }
        let gctx = super::batch::drafter_ctx(ctx, meta, true);
        gpu.begin_capture(stream)?;
        let body = self.step_body(
            batch, sc, sts, &staged, hiddens_in, true, rows, lp, &gctx, stream,
        );
        let graph = gpu.end_capture(stream);
        if let (Err(e), Ok(g)) = (&body, &graph) {
            if g.0 != 0 {
                gpu.destroy_graph(*g)?;
            }
            sc.graph_off = true;
            bail!("GLM MTP batched graph: capture failed (later steps run eagerly): {e:#}");
        }
        let graph = graph?;
        if graph.0 == 0 {
            sc.graph_off = true;
            bail!("GLM MTP batched graph: the capture returned no graph");
        }
        gpu.launch_graph(graph, stream)?;
        if sc.graphs.len() >= GRAPHS_KEPT {
            let (_, old): (BatchGraphKey, GraphHandle) = sc.graphs.remove(0);
            gpu.destroy_graph(old)?;
        }
        sc.graphs.push((key, graph));
        tracing::info!(
            "GLM MTP: captured a batched drafter step for {} sequences",
            sts.len()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: Keys compare by every baked address: equal tuples match, a different hidden,
    /// region or row order does not.
    #[test]
    fn keys_match_only_on_every_address() {
        assert_ne!(
            BatchGraphKey(vec![(1, 2, 3, 4, Some(0))], false),
            BatchGraphKey(vec![(1, 2, 3, 4, Some(0))], true)
        );
        let a = BatchGraphKey(vec![(1, 2, 3, 4, Some(0)), (5, 6, 7, 8, Some(1))], false);
        assert_eq!(a, a.clone());
        assert_ne!(
            a,
            BatchGraphKey(vec![(1, 2, 3, 4, Some(0)), (9, 6, 7, 8, Some(1))], false)
        );
        assert_ne!(
            a,
            BatchGraphKey(vec![(1, 2, 3, 4, Some(0)), (5, 6, 7, 8, Some(2))], false)
        );
        assert_ne!(
            a,
            BatchGraphKey(vec![(5, 6, 7, 8, Some(1)), (1, 2, 3, 4, Some(0))], false)
        );
    }
}
