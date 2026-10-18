// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The batched propose (`METRALE_GLM_MTP_PROPOSE_BATCH=1`): one drafter pass over
//! one row per sequence for each draft step, instead of one pass per sequence.
//!
//! Per draft step, for `m` sequences: the input norms of each row, one `eh_proj` over all rows,
//! the block over all rows (`Glm5NextLayer::decode_rows_for_drafter`), one `shared_head.norm`,
//! one draft-head sweep over all rows, one argmax, the cross-rank exchange of every row behind
//! one all-reduce, and one device-to-host read of the `m` ids.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Row `i` of every buffer is sequence `i` of the call.
//! - A sequence's drafter rows, KV slots and `seq_len` advance as `propose` advances them, so
//!   `after_verify` and the next propose (batched or not) see the same state.
//! - Under the BF16 head every row's draft is the per-sequence propose's draft (the batched
//!   GEMVs follow the one-row GEMVs' order per row). The FP8 batch head is not bit-identical
//!   to the one-row FP8 sweep (it accumulates in another order): it changes drafts, never an
//!   emitted token.

use super::*;
use metrale_gpu_runtime::gpu::GraphHandle;
use metrale_model_layers::layer::AttnMetadataDev;

/// 2026-10-11: The widest batch: the batched BF16 GEMV's and the FP8 batch head's row limit.
pub const PROPOSE_BATCH_MAX: usize = 16;

/// 2026-10-11: The kernels the batched pass needs beyond the one-row ones.
pub(super) struct BatchKernels {
    pub(super) gemv_batchm: KernelHandle,
    pub(super) argmax_batch: KernelHandle,
    /// 2026-10-11: `fp8_gemv_rowscale_batch16_rt2`; `0` keeps the BF16 batched head.
    pub(super) fp8_batch16: KernelHandle,
}

/// 2026-10-11: The batched pass's device buffers, `PROPOSE_BATCH_MAX` rows each.
pub(super) struct BatchScratch {
    pub(super) concat: DevicePtr,
    pub(super) x: DevicePtr,
    pub(super) logits: DevicePtr,
    pub(super) arg: DevicePtr,
    pub(super) xchg: DevicePtr,
    /// 2026-10-11: Row metadata in one buffer, written by one copy per draft step: `[MAX]` i64
    /// slots, then `[MAX]` u32 positions, then `[MAX]` i32 lengths.
    pub(super) meta_words: DevicePtr,
    /// 2026-10-11: `[MAX, per_seq]` i32 block tables; row `i` is the region in
    /// `table_regions[i]`.
    pub(super) block_table: DevicePtr,
    pub(super) per_seq: usize,
    pub(super) table_regions: Vec<Option<usize>>,
    /// 2026-10-11: `[MAX, hidden]` BF16: the embedding rows a graphed step reads, copied in
    /// before each launch.
    pub(super) embed_in: DevicePtr,
    /// 2026-10-11: Captured steps by the device addresses they bake (`batch_graph.rs`).
    pub(super) graphs: Vec<(super::batch_graph::BatchGraphKey, GraphHandle)>,
    /// 2026-10-11: Set after a capture failed: every later step runs eagerly.
    pub(super) graph_off: bool,
}

/// 2026-10-11: The batched pass, present when the lever is on and its kernels resolve.
pub(super) struct ProposeBatch {
    pub(super) kernels: BatchKernels,
    pub(super) scratch: Mutex<BatchScratch>,
}

impl ProposeBatch {
    /// 2026-10-11: `None` with the lever (`on`, `METRALE_GLM_MTP_PROPOSE_BATCH`) off or a kernel
    /// missing (logged once).
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        on: bool,
        hidden: usize,
        vocab: usize,
        per_seq: usize,
    ) -> Result<Option<Self>> {
        if !on {
            return Ok(None);
        }
        let k = |m, f| metrale_model_layers::layers::try_kernel(gpu, m, f);
        let kernels = BatchKernels {
            gemv_batchm: k("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
            argmax_batch: k("argmax", "argmax_bf16_batch"),
            fp8_batch16: k("fp8_gemv_rt", "fp8_gemv_rowscale_batch16_rt2"),
        };
        if kernels.gemv_batchm.0 == 0 || kernels.argmax_batch.0 == 0 {
            tracing::warn!(
                "GLM MTP: METRALE_GLM_MTP_PROPOSE_BATCH set but dense_gemv_bf16_batchm or \
                 argmax_bf16_batch is not in this build; proposes stay per sequence"
            );
            return Ok(None);
        }
        let n = PROPOSE_BATCH_MAX;
        let scratch = BatchScratch {
            concat: gpu.alloc(n * 2 * hidden * 2)?,
            x: gpu.alloc(n * hidden * 2)?,
            logits: gpu.alloc(n * vocab * 2)?,
            // 2026-10-11: `[MAX]` u32 ids, then `[MAX]` f32 log-probabilities (`draft_lp`).
            arg: gpu.alloc(n * 8)?,
            xchg: gpu.alloc(n * super::head_xchg::BUFFER_BYTES)?,
            meta_words: gpu.alloc(n * (8 + 4 + 4))?,
            block_table: gpu.alloc(n * per_seq * 4)?,
            per_seq,
            table_regions: vec![None; n],
            embed_in: gpu.alloc(n * hidden * 2)?,
            graphs: Vec::new(),
            graph_off: false,
        };
        tracing::info!(
            "GLM MTP: batched propose ON, up to {n} sequences per drafter pass ({} head)",
            if kernels.fp8_batch16.0 != 0 {
                "FP8 batch16"
            } else {
                "BF16 batched"
            }
        );
        Ok(Some(Self {
            kernels,
            scratch: Mutex::new(scratch),
        }))
    }
}

/// 2026-10-11: The drafter row a propose at target position `position` starts writing at: the
/// position when the drafter is ahead of it (rows past it are from rejected drafts), else the
/// drafter's own length. A drafter can lag the target: its prompt context is skipped when
/// another sequence's prefill overwrote the shared hidden capture first
/// (`ensure_drafter_context`), so its rows start at 0.
pub(super) fn start_row(drafter_rows: usize, position: usize) -> usize {
    drafter_rows.min(position)
}

/// 2026-10-11: The packed row metadata of one draft step: the bytes `meta_words` holds for
/// rows at drafter rows `rows[i]` with block tables `tables[i]`, padded to `max` rows.
pub(super) fn meta_words_bytes(rows: &[usize], tables: &[&[u32]], max: usize) -> Result<Vec<u8>> {
    let mut slot = vec![0u8; max * 8];
    let mut pos = vec![0u8; max * 4];
    let mut len = vec![0u8; max * 4];
    for (i, (&row, table)) in rows.iter().zip(tables).enumerate() {
        let w = super::graph::staged_words(row, table)?;
        slot[i * 8..][..8].copy_from_slice(&w.slot.to_le_bytes());
        pos[i * 4..][..4].copy_from_slice(&w.position.to_le_bytes());
        len[i * 4..][..4].copy_from_slice(&w.seq_len.to_le_bytes());
    }
    slot.extend_from_slice(&pos);
    slot.extend_from_slice(&len);
    Ok(slot)
}

/// 2026-10-11: The drafter's `ForwardContext` around `meta`: the caller's buffers, switches and
/// communicator, one decode step, captured or not.
pub(super) fn drafter_ctx<'a>(
    ctx: &ForwardContext<'a>,
    meta: AttnMetadataDev,
    graph_capture: bool,
) -> ForwardContext<'a> {
    ForwardContext {
        buffers: ctx.buffers,
        hc_row_offset: ctx.hc_row_offset,
        gpu: ctx.gpu,
        config: ctx.config,
        dispatch: ctx.dispatch,
        derived: ctx.derived,
        levers: ctx.levers,
        settings: ctx.settings,
        stats: ctx.stats,
        attn_metadata: Some(meta),
        profile: false,
        comm: ctx.comm,
        graph_capture,
        decode_step: true,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: ctx.moe_lora_route,
    }
}

impl Glm5NextMtpHead {
    /// 2026-10-11: The widest batch `propose_batch` takes: 1 (per sequence) without the batched
    /// pass, or with a sharded head whose exchange is not on the device (the batched pass has
    /// no host exchange).
    pub(super) fn batch_max(&self) -> usize {
        let sharded_host_xchg = self.head_n != self.vocab && self.xchg_k.is_none();
        if self.batch.is_some() && !sharded_host_xchg {
            PROPOSE_BATCH_MAX
        } else {
            1
        }
    }

    /// 2026-10-11: `num_drafts` drafts for each of `states.len()` sequences, one drafter pass
    /// per draft step over all of them. `Ok(None)` when the batch is not supported
    /// (`batch_max`); a supported batch never returns `None`, so a caller that announced it to
    /// other ranks gets drafts or an error.
    ///
    /// 2026-10-11: With `out_conf`, or under `--draft-confidence-stop`, each draft's top-1
    /// log-probability (`draft_lp`; NaN where it cannot be computed) goes to `out_conf[i]`, and
    /// under the stop the steps end once every sequence's chain has stopped
    /// (`draft_stop::ChainStop::continues`). Every sequence keeps the drafts the steps made; the
    /// scheduler truncates each chain to its own depth (`draft_stop::chain_depth` in
    /// `mtp_dcut::plan`), as for the Qwen drafter.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_rows(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
        mut out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let m = states.len();
        let Some(batch) = self.batch.as_ref() else {
            return Ok(None);
        };
        if m == 0 || m > self.batch_max() {
            return Ok(None);
        }
        if last_tokens.len() != m || target_hiddens.len() != m || positions.len() != m {
            bail!(
                "GLM MTP batched propose: {m} states, {} tokens, {} hiddens, {} positions",
                last_tokens.len(),
                target_hiddens.len(),
                positions.len()
            );
        }
        let mut sts: Vec<&mut Glm5NextMtpProposerState> = Vec::with_capacity(m);
        for s in states.iter_mut() {
            sts.push(
                s.as_any_mut()
                    .downcast_mut::<Glm5NextMtpProposerState>()
                    .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?,
            );
        }
        // 2026-10-11: As `propose`: a drafter ahead of the target's position is rewound to it
        // (`start_row`); one behind it (no prompt context) starts where it is. 2026-10-13: One
        // whose drafts would pass its row space restarts at row 0 (`propose_start`).
        for (st, &p) in sts.iter_mut().zip(positions) {
            let start =
                super::context_cap::propose_start(st.seq_len, p, num_drafts, self.max_seq_len);
            if start < st.seq_len {
                st.dsa.rewind_to(start)?;
                st.seq_len = start;
            }
        }
        let starts: Vec<usize> = sts.iter().map(|st| st.seq_len).collect();
        let gpu = ctx.gpu;
        let h = self.hidden;
        let mut sc = batch.scratch.lock();
        // 2026-10-11: Block-table rows are uploaded only when a row's region changes. Every
        // state holds a region unless the pool leaked one (`alloc_state`).
        for (i, st) in sts.iter().enumerate() {
            if st.kv_region.is_none() {
                bail!("GLM MTP batched propose: sequence {i} has no drafter KV region");
            }
            if sc.table_regions[i] != st.kv_region {
                if st.block_table.len() != sc.per_seq {
                    bail!(
                        "GLM MTP batched propose: a {}-block table for {}-block rows",
                        st.block_table.len(),
                        sc.per_seq
                    );
                }
                let bytes: Vec<u8> = st
                    .block_table
                    .iter()
                    .flat_map(|&b| (b as i32).to_le_bytes())
                    .collect();
                gpu.copy_h2d(&bytes, sc.block_table.offset(i * sc.per_seq * 4))?;
                sc.table_regions[i] = st.kv_region;
            }
        }
        let max = PROPOSE_BATCH_MAX;
        let meta = AttnMetadataDev {
            positions: sc.meta_words.offset(max * 8),
            positions_h: sc.meta_words.offset(max * 8),
            positions_w: sc.meta_words.offset(max * 8),
            slot: sc.meta_words,
            seq_len: sc.meta_words.offset(max * 12),
            block_table: sc.block_table,
            max_blocks_per_seq: sc.per_seq as u32,
            num_seqs: m as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let bctx = drafter_ctx(ctx, meta, false);
        let mut drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(num_drafts); m];
        let mut tokens: Vec<u32> = last_tokens.to_vec();
        // 2026-10-11: One stop state per sequence, under the published rule (either form).
        let stop_rule = metrale_model_layers::speculative::draft_stop::draft_stop_rule();
        let mut chains = stop_rule.map(|r| vec![r.chain(); m]);
        // 2026-10-11: Log-probabilities whenever computable, at `arg + 4 * MAX`. NOT "when this
        // caller wants them": the choice sets the exchange's kernels and all-reduce size, and a
        // worker's propose (`ep_worker_mtp_propose_batch`) passes no `out_conf` while rank 0's
        // passes one under D-Cut. Keyed on `out_conf`, rank 0 all-reduced 36 bytes per row and
        // the workers 24 (world 3): the drafts were garbage and the communicator died (nightO
        // w2/w8). It must be the same on every rank, so it depends only on the build and the
        // serve's levers.
        let lp = self.lp_ready(ctx, true).then_some(sc.arg.offset(max * 4));
        if let Some(c) = out_conf.as_deref_mut() {
            c.clear();
            c.resize(m, Vec::with_capacity(num_drafts));
        }
        let mut stopped = vec![false; m];
        let mut steps = 0usize;
        for d in 0..num_drafts {
            let rows: Vec<usize> = sts.iter().map(|st| st.seq_len).collect();
            if let Some(&r) = rows.iter().find(|&&r| r >= self.max_seq_len) {
                bail!(
                    "GLM MTP drafter: row {r} is past the {} it was sized for",
                    self.max_seq_len
                );
            }
            let tables: Vec<&[u32]> = sts.iter().map(|st| st.block_table.as_slice()).collect();
            gpu.copy_h2d(&meta_words_bytes(&rows, &tables, max)?, sc.meta_words)?;
            // 2026-10-11: A row's embedding half is zeroed at pair key 0 under
            // `METRALE_MTP_ZERO_POS0`, as `forward_one` does: its key is the TARGET position of
            // this draft minus one (`positions[i] + d - 1`), not the drafter row, which lags the
            // target for a drafter without its prompt context.
            let embeds: Vec<Option<DevicePtr>> = tokens
                .iter()
                .zip(positions)
                .map(|(&t, &p)| {
                    let zero = (p + d).checked_sub(1).is_some_and(|key| {
                        super::hidden_half::zeroes_embedding(ctx.levers.mtp_zero_pos0, key)
                    });
                    (!zero).then(|| self.embed_tokens.weight.offset(t as usize * h * 2))
                })
                .collect();
            let hiddens_in: Vec<DevicePtr> = (0..m)
                .map(|i| {
                    if d == 0 {
                        target_hiddens[i]
                    } else {
                        sc.x.offset(i * h * 2)
                    }
                })
                .collect();
            // 2026-10-11: Draft 0 replays a graph of the step when `METRALE_GLM_MTP_GRAPH` is on
            // (`batch_graph.rs`); later drafts read the drafter's own rows and run eagerly, as
            // `propose` does.
            // 2026-10-11: A zeroed embedding row has no row to stage, so that step runs eagerly.
            let all_embedded: Option<Vec<DevicePtr>> = embeds.iter().copied().collect();
            match all_embedded {
                Some(e) if d == 0 && self.batch_graph_eligible(&sc) => self.step_graphed(
                    batch,
                    &mut sc,
                    &mut sts,
                    &e,
                    &hiddens_in,
                    &rows,
                    lp,
                    meta,
                    ctx,
                    stream,
                )?,
                _ => self.step_body(
                    batch,
                    &sc,
                    &mut sts,
                    &embeds,
                    &hiddens_in,
                    d == 0,
                    &rows,
                    lp,
                    &bctx,
                    stream,
                )?,
            }
            steps = d + 1;
            // 2026-10-11: One read per draft step: every row's id, and with `lp` every row's
            // log-probability after the `MAX` id slots.
            let mut out = vec![0u8; if lp.is_some() { max * 8 } else { m * 4 }];
            gpu.copy_d2h(sc.arg, &mut out)?;
            let word = |o: usize| [out[o], out[o + 1], out[o + 2], out[o + 3]];
            for i in 0..m {
                // 2026-10-11: A global id: the exchange's pick when sharded, else the argmax over
                // the whole vocab (`v0` is 0).
                let id = u32::from_le_bytes(word(i * 4));
                let l = if lp.is_some() {
                    f32::from_le_bytes(word(max * 4 + i * 4))
                } else {
                    f32::NAN
                };
                drafts[i].push(id);
                tokens[i] = id;
                if let Some(c) = out_conf.as_deref_mut() {
                    c[i].push(l);
                }
                if let Some(c) = chains.as_mut()
                    && !stopped[i]
                    && !l.is_nan()
                    && !c[i].continues(l)
                {
                    stopped[i] = true;
                }
            }
            if chains.is_some() && stopped.iter().all(|&s| s) {
                break;
            }
        }
        // 2026-10-11: The rows each drafter wrote, which `after_verify` trims from.
        for st in sts.iter_mut() {
            st.last_drafted = steps;
        }
        drop(sc);
        if self.levers.propose_batch_check {
            self.check_against_rows(
                &drafts,
                last_tokens,
                target_hiddens,
                positions,
                &starts,
                num_drafts,
                &mut sts,
                ctx,
                stream,
            )?;
        }
        Ok(Some(drafts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: Row i's slot, position and length land at index i of their arrays, the
    /// arrays at `[0, max*8)`, `[max*8, max*12)` and `[max*12, max*16)`, unused rows zero.
    #[test]
    fn meta_words_place_each_row_in_its_array() {
        let a: Vec<u32> = super::super::kv_pool::region_table(0, 4);
        let b: Vec<u32> = super::super::kv_pool::region_table(3, 4);
        let bytes = meta_words_bytes(&[5, 17], &[&a, &b], 3).unwrap();
        assert_eq!(bytes.len(), 3 * 16);
        let i64_at = |i: usize| i64::from_le_bytes(bytes[i * 8..][..8].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..][..4].try_into().unwrap());
        assert_eq!(i64_at(0), 5);
        assert_eq!(i64_at(1), (13 * 16 + 1) as i64);
        assert_eq!(i64_at(2), 0);
        assert_eq!(u32_at(24), 5);
        assert_eq!(u32_at(28), 17);
        assert_eq!(u32_at(36), 6);
        assert_eq!(u32_at(40), 18);
        assert_eq!(u32_at(44), 0);
    }

    /// 2026-10-11: A drafter ahead of the target rewinds to the target's position; one behind
    /// it (no prompt context: rows 2 at position 101, the case nightM's parity check hit when
    /// it rewound "forward" to the position) starts at its own rows.
    #[test]
    fn a_propose_starts_at_the_drafter_rows_or_the_position_whichever_is_lower() {
        assert_eq!(start_row(110, 101), 101);
        assert_eq!(start_row(101, 101), 101);
        assert_eq!(start_row(2, 101), 2);
        assert_eq!(start_row(0, 76), 0);
    }

    /// 2026-10-11: A row past its table is refused, not written to another region's slot.
    #[test]
    fn a_row_past_its_table_is_refused() {
        let t = super::super::kv_pool::region_table(1, 2);
        assert!(meta_words_bytes(&[32], &[&t], 1).is_err());
    }
}
