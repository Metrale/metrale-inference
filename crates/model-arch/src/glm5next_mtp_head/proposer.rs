// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl DraftProposer for Glm5NextMtpHead`: state lifetime, context prefill, propose and rollback.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - `after_verify` never trims the first row of a propose.
//! - A second `free_state` on the same state is a no-op.

use super::*;

impl DraftProposer for Glm5NextMtpHead {
    /// 2026-09-25: `self.max_seq_len` is already capped at `max_dsa_context` by `new`, so the
    /// model's capture buffer is no longer than the drafter's own row space.
    fn prefill_hidden_rows(&self, max_seq_len: usize) -> usize {
        max_seq_len.min(self.max_seq_len)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        let dsa = match &self.module.layer.mixer {
            crate::glm5next_layer::Glm5NextMixer::Dsa(l) => Glm5NextDsaState::alloc(gpu, &l.cfg)?,
            _ => bail!("GLM MTP block is not a DSA layer"),
        };
        let h = self.hidden;
        // 2026-10-11: A region of the pool of its own (`kv_pool`); before, every state claimed
        // the same single-sequence pool and live sequences overwrote each other's rows. The
        // pool holds one region per `--max-batch-size` sequence, so a miss means a state was
        // not freed; the sequence then drafts nothing instead of failing.
        let (kv_region, block_table) = match self.kv_regions.claim() {
            Ok((r, t)) => (Some(r), t),
            Err(e) => {
                tracing::warn!("{e:#}; this sequence proposes no drafts");
                (None, Vec::new())
            }
        };
        let graph = if super::graph::graph_enabled() {
            Some(super::graph::DrafterGraph::alloc(gpu, h, &block_table)?)
        } else {
            None
        };
        Ok(Box::new(Glm5NextMtpProposerState {
            dsa,
            seq_len: 0,
            block_table,
            last_drafted: 0,
            concat: gpu.alloc(2 * h * 2)?,
            x: gpu.alloc(h * 2)?,
            logits: gpu.alloc(self.vocab * 2)?,
            // 2026-10-11: The draft id, then its log-probability (`draft_lp`).
            arg: gpu.alloc(8)?,
            head_xchg: gpu.alloc(super::head_xchg::BUFFER_BYTES)?,
            released: false,
            kv_region,
            graph,
        }))
    }

    /// 2026-09-25: Release everything `alloc_state` allocated: the DSA indexer cache and the
    /// scratch buffers. `DevicePtr` has no `Drop`, so without this override they would leak.
    /// A second call is a no-op (`released`).
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid GLM MTP proposer state"))?;
        if st.released {
            return Ok(());
        }
        st.released = true;
        st.dsa.free(gpu)?;
        for p in [st.concat, st.x, st.logits, st.arg, st.head_xchg] {
            gpu.free(p)?;
        }
        if let Some(mut g) = st.graph.take() {
            g.free(gpu)?;
        }
        // 2026-10-11: The region goes back to the pool; clearing the table stops a freed state
        // from looking live.
        if let Some(r) = st.kv_region.take() {
            self.kv_regions.release(r)?;
        }
        st.block_table.clear();
        st.seq_len = 0;
        Ok(())
    }

    /// 2026-09-25: The block's routed experts are split across EP ranks and its DSA `o_proj`
    /// is row-parallel, so its forward needs the communicator. On unless
    /// `METRALE_NO_MTP_EP_PROPOSE=1` (`mtp_ep_propose_enabled`).
    fn needs_comm(&self) -> bool {
        metrale_model_layers::speculative::mtp_ep_propose_enabled()
    }

    /// 2026-10-11: A context row normalises into the state's own `concat`
    /// (`drafter_write_kv_row`), so the drafter prefill can run at the end of each sequence's
    /// target prefill, while the shared capture is still its own; at the first propose instead,
    /// at C >= 2 every sequence but the last-prefilled drafts without its prompt. Opt-in
    /// (`METRALE_GLM_MTP_EAGER_PREFILL=1`) until the context rows run as one pass: one at a time
    /// they cost ~0.35 ms per prompt row, which would land in every request's TTFT. The
    /// diagnostic full-block arm (`METRALE_GLM_MTP_PREFILL_FULL=1`, `decode_one_for_drafter`)
    /// writes the shared scratch, so it always waits for the first propose.
    fn prefill_uses_shared_buffers(&self) -> bool {
        std::env::var("METRALE_GLM_MTP_PREFILL_FULL").as_deref() == Ok("1")
            || std::env::var("METRALE_GLM_MTP_EAGER_PREFILL").as_deref() != Ok("1")
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .map_or(0, |st| st.seq_len)
    }

    /// 2026-09-25: Dense row space: the newest row's slot is its pair key.
    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .and_then(|st| st.seq_len.checked_sub(1))
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let t0 = std::time::Instant::now();
        let rows = self.rows_impl(prompt_tokens, hiddens, 0, state, ctx, stream)?;
        // 2026-09-25: `rows_impl` returns 0 without work once the drafter has rows, so only a
        // call that wrote rows logs.
        if rows > 0 {
            tracing::info!(
                "GLM MTP drafter prefill: {rows} rows ({} prompt tokens) in {:.1} ms",
                prompt_tokens.len(),
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(rows)
    }

    /// 2026-09-25: `pos_base` is unused: this drafter's RoPE position is its KV slot (see
    /// `rows_impl`). A feed that does not start exactly at `drafter_rows()` writes nothing.
    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        _pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.rows_impl(tokens, hiddens, row_base, state, ctx, stream)
    }

    /// 2026-10-11: The batched-verify catch-up (`METRALE_MTP_ACCEPTED_ROWS` or `_KV_EXACT`): each
    /// sequence's accepted drafts as context rows (`catchup.rs`).
    fn catchup_batch(
        &self,
        tokens: &[Vec<u32>],
        hiddens: &[Vec<DevicePtr>],
        first_pos: &[usize],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.catchup_rows(tokens, hiddens, first_pos, states, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        if !self.begin_propose(st, last_token, target_hidden, position, ctx)? {
            return Ok(Vec::new());
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        // 2026-10-11: `--draft-confidence-stop`: the chain ends with the first draft whose top-1
        // probability is below tau (`speculative::draft_stop`); an unmeasured draft (NaN) never
        // ends it.
        // 2026-10-11: The published rule, either form (`draft_stop::StopRule`).
        let mut stop =
            metrale_model_layers::speculative::draft_stop::draft_stop_rule().map(|r| r.chain());
        for i in 0..num_drafts {
            let (d, lp) = self.forward_one(token, hidden, i == 0, position + i, st, ctx, stream)?;
            drafts.push(d);
            if let Some(stop) = stop.as_mut()
                && !lp.is_nan()
                && !stop.continues(lp)
            {
                break;
            }
            token = d;
            // 2026-09-25: Draft 1 consumes the target's verified hidden; every later draft
            // consumes the drafter's own block output.
            hidden = st.x;
        }
        st.last_drafted = drafts.len();
        Ok(drafts)
    }

    /// 2026-10-11: One draft left in `st.arg` (`DraftProposer::propose_device`), when the id
    /// ends there on every rank (`draft_stays_on_device`): the same launches as `propose` with one
    /// draft, without the readback.
    fn propose_device(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !self.draft_stays_on_device(ctx) {
            return Ok(false);
        }
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        if !self.begin_propose(st, last_token, target_hidden, position, ctx)? {
            return Ok(false);
        }
        self.draft_on_device(last_token, target_hidden, true, position, st, ctx, stream)?;
        st.last_drafted = 1;
        Ok(true)
    }

    /// 2026-10-11: `st.arg`, where `propose_device` leaves the draft.
    fn device_draft(&self, state: &mut dyn ProposerState) -> Option<DevicePtr> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .map(|st| st.arg)
    }

    /// 2026-10-11: The batched propose (`batch.rs`); `Ok(None)` without it. Each draft's top-1
    /// log-probability goes to `out_conf` when asked for (`draft_lp`).
    #[allow(clippy::too_many_arguments)]
    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.propose_rows(
            last_tokens,
            target_hiddens,
            positions,
            num_drafts,
            states,
            ctx,
            stream,
            out_conf,
        )
    }

    fn propose_batch_max(
        &self,
        _buffers: &metrale_gpu_runtime::buffers::BufferArena,
        _config: &metrale_config::ModelConfig,
    ) -> usize {
        self.batch_max()
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        // 2026-09-25: Rejected rows are unreachable: the indexer reads `[0, len)` and the next
        // propose writes from `seq_len`, so rolling the counters back is the whole rollback.
        //
        // The first row of a propose is always kept. It pairs the last committed token with the
        // target's own hidden, both known at propose time, so a rejected draft does not make it
        // wrong. Only later rows depend on a draft having been accepted, and trimming the first
        // would drop a real row from the dense row space and shift every later RoPE position.
        // At `num_drafts = 1` nothing is trimmed.
        let keep = st.last_drafted.min(num_accepted + 1);
        let trim = st.last_drafted - keep;
        if trim > 0 {
            st.seq_len = st.seq_len.saturating_sub(trim);
            st.dsa.rewind_to(st.seq_len)?;
        }
        Ok(())
    }
}

impl Glm5NextMtpHead {
    /// 2026-10-11: The start of every propose: `false` (no drafts) without a KV region
    /// (`alloc_state`); a drafter ahead of the target's `position` is rewound to it, so its
    /// indexer does not select over rows past the target's context.
    fn begin_propose(
        &self,
        st: &mut Glm5NextMtpProposerState,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        ctx: &ForwardContext,
    ) -> Result<bool> {
        if st.kv_region.is_none() {
            return Ok(false);
        }
        let start = super::batch::start_row(st.seq_len, position);
        if start < st.seq_len {
            st.dsa.rewind_to(start)?;
            st.seq_len = start;
        }
        if metrale_model_layers::speculative::mtp_refeed_debug() {
            let fp = metrale_model_layers::speculative::hidden_fingerprint(
                ctx.gpu,
                target_hidden,
                self.hidden,
            );
            tracing::info!(
                "GLM_MTP_DBG propose position={position} drafter_rows={} tok={last_token} \
                 fp_target={fp:016x}",
                st.seq_len,
            );
        }
        Ok(true)
    }
}
