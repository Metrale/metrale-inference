// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The standalone folds of the carried GDN state (`gdn_carry_flush` and
//! `gdn_carry_conv_flush`): every pending slot before a call that reads the state, or the
//! slots a verify must not carry pending into an eager launch.
//!
//! Owner: model-engine speculative decoding.
//! Invariants: see the parent module.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;

use super::super::types::TransformerModel;
use super::{CarryInner, SlotStates};

impl TransformerModel {
    /// 2026-09-26: Fold every pending row into H. A no-op when nothing is pending.
    pub(crate) fn gdn_carry_flush_pending(&self) -> Result<()> {
        let mut inner = self.gdn_carry.inner.lock();
        if inner.pending.is_empty() {
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        self.gdn_carry_flush_locked(&mut inner, stream, |_| true)
    }

    /// 2026-09-26: Fold the pending slots `which` selects; the rest stay pending.
    pub(super) fn gdn_carry_flush_locked(
        &self,
        inner: &mut CarryInner,
        stream: u64,
        which: impl Fn(&SlotStates) -> bool,
    ) -> Result<()> {
        let (pending, keep): (Vec<SlotStates>, Vec<SlotStates>) =
            std::mem::take(&mut inner.pending)
                .into_iter()
                .partition(|r| which(r));
        inner.pending = keep;
        let b = inner.bufs.as_ref().expect("bound");
        let (layers, slots_n) = (b.layers, b.slots);
        for chunk in pending.chunks(VERIFY_WY_TABLE_SEQS) {
            let slot_bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|r| (r.slot as u32).to_le_bytes())
                .collect();
            self.gpu
                .copy_h2d_async(&state_table(chunk, layers, |r| &r.h), b.flush_tab, stream)?;
            self.gpu.copy_h2d_async(
                &state_table(chunk, layers, |r| &r.conv),
                b.flush_conv_tab,
                stream,
            )?;
            self.gpu
                .copy_h2d_async(&slot_bytes, b.flush_slots, stream)?;
            metrale_model_layers::layers::ops::gdn_carry_flush(
                self.gpu.as_ref(),
                b.flush_k,
                b.flush_tab,
                VERIFY_WY_TABLE_SEQS as u64,
                b.stash,
                (slots_n * b.seq_floats) as u64,
                b.flush_slots,
                b.pend,
                slots_n as u32,
                b.seq_floats as u32,
                chunk.len() as u32,
                b.nv as u32,
                layers as u32,
                stream,
            )?;
            metrale_model_layers::layers::ops::gdn_carry_conv_flush(
                self.gpu.as_ref(),
                b.conv_flush_k,
                b.flush_conv_tab,
                VERIFY_WY_TABLE_SEQS as u64,
                b.conv_stash,
                (slots_n * b.conv_seq_elems) as u64,
                b.flush_slots,
                b.pend,
                slots_n as u32,
                b.conv_seq_elems as u32,
                chunk.len() as u32,
                b.conv_dim as u32,
                b.d_conv as u32,
                layers as u32,
                stream,
            )?;
        }
        for r in &pending {
            for l in 0..layers {
                inner.mirror[l * slots_n + r.slot] = 0;
            }
        }
        self.upload_mirror(inner, stream)
    }
}

/// 2026-09-26: `[layer][VERIFY_WY_TABLE_SEQS]` u64 pointer table of `rows` (row `j` at
/// entry `j` of each layer), from `pick`; rows without an entry for a layer leave 0.
pub(super) fn state_table(
    rows: &[SlotStates],
    layers: usize,
    pick: impl Fn(&SlotStates) -> &Vec<DevicePtr>,
) -> Vec<u8> {
    let mut tab = vec![0u64; layers * VERIFY_WY_TABLE_SEQS];
    for (j, r) in rows.iter().enumerate() {
        for (l, p) in pick(r).iter().enumerate() {
            tab[l * VERIFY_WY_TABLE_SEQS + j] = p.0;
        }
    }
    tab.iter().flat_map(|v| v.to_le_bytes()).collect()
}
