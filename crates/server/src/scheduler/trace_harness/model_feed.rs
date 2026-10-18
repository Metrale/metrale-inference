// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The recording fake's device token feed: the `Model` methods the asynchronous router calls, recorded and answered like every other model call.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The feed cells are a plain `Vec<u32>` in the shared state and the
//! "pinned" slot is heap memory. An event completes on its first query
//! unless `ModelCfg::event_lag` or `reversed_completion` holds it back, and
//! the n-th query can be scripted to fail (`event_query_fault_at`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    FedAheadPolicy, FeedSource, ModelDeviceFeed, RowMask, SequenceState,
};

use super::model::{RecordingModel, fmt_ids};
use super::model_forward::sid;

macro_rules! rec {
    ($m:expr, $($arg:tt)*) => { $m.rec(format!($($arg)*)) };
}

impl RecordingModel {
    fn feed_cell(&self, row: u32) -> u32 {
        let st = self.state.lock().unwrap();
        st.feed_cells.get(row as usize).copied().unwrap_or_else(|| {
            panic!(
                "feed cell {row} read, but the last feed argmax wrote {}",
                st.feed_cells.len()
            )
        })
    }

    /// 2026-09-25: Grow `seq.block_table` to what the sync decode of the same step would
    /// leave it at (`blocks_for(seq_len + 1)`), returning the blocks added.
    fn grow_for_next_position(&self, seq: &mut SequenceState) -> usize {
        let want = self.blocks_for(seq.seq_len + 1);
        let have = seq.block_table.len();
        if want > have {
            self.take_blocks(want - have);
            seq.block_table.extend((have..want).map(|b| b as u32));
        }
        want.saturating_sub(have)
    }

    pub(super) fn fx_decode_batch_fed(
        &self,
        sources: &[FeedSource],
        seqs: &mut [&mut SequenceState],
        st: u64,
    ) -> Result<DevicePtr> {
        let call = self.next_decode_batch_call();
        let inputs: Vec<u32> = sources
            .iter()
            .map(|s| match *s {
                FeedSource::Feed { from_row } => self.feed_cell(from_row),
                FeedSource::Host(id) => id,
            })
            .collect();
        let ids: Vec<String> = seqs.iter().map(|s| sid(s)).collect();
        let head = format!(
            "decode_batch_fed(inputs={}, seqs={ids:?}, stream={st})",
            fmt_ids(&inputs)
        );
        if self.cfg.kv_exhaust_at.contains(&call) {
            rec!(self, "{head} -> ERR KV cache exhausted");
            anyhow::bail!("KV cache exhausted (scripted at decode_batch #{call})");
        }
        let mut rows = Vec::with_capacity(seqs.len());
        for s in seqs.iter_mut() {
            self.maybe_cancel(s);
            // 2026-09-25: the input sits at position `seq_len`; the
            // script's next token follows it. No push: per the
            // `decode_batch_fed` contract the core applies that
            // bookkeeping.
            self.grow_for_next_position(s);
            let clean = self.recurrent_consume(s, s.seq_len, 1);
            rows.push(self.recurrent_out(clean, self.out_after(s, s.seq_len)));
        }
        self.set_rows(rows);
        self.rec(head);
        Ok(self.row_ptr(0))
    }

    /// 2026-10-10: The masked pick of row `row` of the last forward (`argmax_feed.cu`'s rule
    /// over the fake's one-hot logits).
    fn masked_pick(&self, row: usize, mask: RowMask) -> u32 {
        let hit = self.row(row);
        if hit == mask[0] || hit == mask[1] {
            (0..self.cfg.vocab as u32)
                .rev()
                .find(|t| *t != mask[0] && *t != mask[1])
                .unwrap_or(0)
        } else {
            hit
        }
    }

    pub(super) fn fx_arm_device_argmax(&self, masks: &[RowMask]) -> Result<bool> {
        let mut st = self.state.lock().unwrap();
        st.armed = None;
        if !self.cfg.device_argmax || masks.len() != 1 {
            return Ok(false);
        }
        st.armed = Some(masks[0]);
        drop(st);
        rec!(self, "arm_device_argmax(masks={masks:?}) -> true");
        Ok(true)
    }

    pub(super) fn fx_device_argmax_to_host(
        &self,
        dst: &mut [u8],
        event: Option<u64>,
    ) -> Result<()> {
        let mask = self
            .state
            .lock()
            .unwrap()
            .armed
            .take()
            .ok_or_else(|| anyhow::anyhow!("device_argmax_to_host without an armed step"))?;
        let tok = self.masked_pick(0, mask);
        self.state.lock().unwrap().feed_cells = vec![tok];
        dst[..4].copy_from_slice(&tok.to_le_bytes());
        if let Some(e) = event {
            self.fx_record_event(e);
        }
        rec!(self, "device_argmax_to_host(event={event:?}) -> {tok}");
        Ok(())
    }

    pub(super) fn fx_argmax_batch_to_feed(
        &self,
        logits_ptr: DevicePtr,
        masks: &[RowMask],
        dst: *mut u32,
        event: u64,
        st: u64,
    ) -> Result<()> {
        let (row, _) = self.locate(logits_ptr);
        let vocab = self.cfg.vocab as u32;
        let out: Vec<u32> = masks
            .iter()
            .enumerate()
            .map(|(i, mask)| {
                // 2026-09-25: the fake's logits are one-hot at the
                // scripted token: the pick is that token, and when a
                // row mask hits it the pick is the highest unmasked id.
                let hit = self.row(row + i);
                if hit == mask[0] || hit == mask[1] {
                    (0..vocab)
                        .rev()
                        .find(|t| *t != mask[0] && *t != mask[1])
                        .unwrap_or(0)
                } else {
                    hit
                }
            })
            .collect();
        self.state.lock().unwrap().feed_cells = out.clone();
        // 2026-09-25: per the trait contract, `event` is recorded after the D2H.
        self.fx_record_event(event);
        // 2026-09-25: SAFETY: `dst` is one of the router's readback ring
        // slots, `max_rows` cells wide, and the router refuses more than
        // `max_rows` masks; the ring lives until the router's teardown.
        let slot = unsafe { std::slice::from_raw_parts_mut(dst, out.len()) };
        slot.copy_from_slice(&out);
        rec!(
            self,
            "argmax_batch_to_feed(row={row}, n={}, masks={masks:?}, event={event}, stream={st}) -> {out:?}",
            masks.len()
        );
        Ok(())
    }

    pub(super) fn fx_reserve_decode_block(&self, seq: &mut SequenceState) -> Result<usize> {
        if self.free_blocks() == 0 && self.blocks_for(seq.seq_len + 1) > seq.block_table.len() {
            rec!(self, "reserve_decode_block({}) -> ERR exhausted", sid(seq));
            anyhow::bail!("KV cache exhausted: no free blocks (scripted reserve)");
        }
        let added = self.grow_for_next_position(seq);
        rec!(self, "reserve_decode_block({}) -> {added}", sid(seq));
        Ok(added)
    }

    pub(super) fn fx_release_decode_blocks(
        &self,
        seq: &mut SequenceState,
        blocks: usize,
    ) -> Result<()> {
        let n = blocks.min(seq.block_table.len());
        seq.block_table.truncate(seq.block_table.len() - n);
        self.give_blocks(n);
        rec!(self, "release_decode_blocks({}, n={blocks})", sid(seq));
        Ok(())
    }
}

// 2026-09-25: events, which the router polls before it reads a readback slot.

impl RecordingModel {
    pub(super) fn fx_create_event(&self) -> u64 {
        let mut st = self.state.lock().unwrap();
        st.events_created += 1;
        st.events_created + 1
    }

    pub(super) fn fx_record_event(&self, event: u64) {
        let mut st = self.state.lock().unwrap();
        st.pending_events.retain(|(e, _)| *e != event);
        st.pending_events.push((event, 0));
    }

    /// 2026-09-25: Complete `event` and everything recorded before it.
    pub(super) fn fx_event_synchronize(&self, event: u64) {
        let mut st = self.state.lock().unwrap();
        if let Some(pos) = st.pending_events.iter().position(|(e, _)| *e == event) {
            st.pending_events.drain(..=pos);
        }
    }

    pub(super) fn fx_event_query(&self, event: u64) -> Result<bool> {
        let mut st = self.state.lock().unwrap();
        st.event_queries += 1;
        if self.cfg.event_query_fault_at == Some(st.event_queries) {
            drop(st);
            rec!(
                self,
                "event_query(event={event}) -> ERR scripted device fault"
            );
            anyhow::bail!("scripted device fault");
        }
        let Some(pos) = st.pending_events.iter().position(|(e, _)| *e == event) else {
            return Ok(true);
        };
        st.pending_events[pos].1 += 1;
        let lagging = st.pending_events[pos].1 <= self.cfg.event_lag;
        let newer_pending = self.cfg.reversed_completion && pos + 1 < st.pending_events.len();
        if lagging || newer_pending {
            return Ok(false);
        }
        st.pending_events.remove(pos);
        Ok(true)
    }
}

// 2026-10-10: `impl ModelDeviceFeed`, moved here from `model_impl.rs` (size cap) with its
// macro-generated methods written out; the trace lines are unchanged.
impl ModelDeviceFeed for RecordingModel {
    fn fed_ahead_policy(&self) -> FedAheadPolicy {
        match self.cfg.recurrent_fed_rows {
            Some(max_rows) => FedAheadPolicy::RepickOnly { max_rows },
            None => FedAheadPolicy::Redo,
        }
    }
    fn arm_device_argmax(&self, masks: &[RowMask]) -> Result<bool> {
        self.fx_arm_device_argmax(masks)
    }
    fn device_argmax_to_host(&self, dst: &mut [u8], event: Option<u64>) -> Result<()> {
        self.fx_device_argmax_to_host(dst, event)
    }
    fn decode_batch_fed(
        &self,
        sources: &[FeedSource],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.fx_decode_batch_fed(sources, seqs, stream)
    }
    fn argmax_batch_to_feed(
        &self,
        logits_ptr: DevicePtr,
        masks: &[RowMask],
        dst: *mut u32,
        event: u64,
        stream: u64,
    ) -> Result<()> {
        self.fx_argmax_batch_to_feed(logits_ptr, masks, dst, event, stream)
    }
    fn reserve_decode_block(&self, seq: &mut SequenceState) -> Result<usize> {
        self.fx_reserve_decode_block(seq)
    }
    fn release_decode_blocks(&self, seq: &mut SequenceState, blocks: usize) -> Result<()> {
        self.fx_release_decode_blocks(seq, blocks)
    }
    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        metrale_gpu_runtime::host_heap::alloc_zeroed(bytes)
    }
    fn free_host_pinned(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        metrale_gpu_runtime::host_heap::free(ptr, bytes)
    }
    fn supports_device_token_feed(&self) -> bool {
        self.cfg.device_token_feed
    }
    fn event_query(&self, event: u64) -> Result<bool> {
        self.fx_event_query(event)
    }
    fn event_synchronize(&self, event: u64) -> Result<()> {
        rec!(self, "event_synchronize(event={event})");
        self.fx_event_synchronize(event);
        Ok(())
    }
    fn destroy_event(&self, event: u64) -> Result<()> {
        rec!(self, "destroy_event(event={event})");
        Ok(())
    }
    fn fed_step_settled(&self) {
        rec!(self, "fed_step_settled()");
    }
}
