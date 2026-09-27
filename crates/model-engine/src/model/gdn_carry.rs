// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Model side of the carried-state GDN verify (`qwen3_ssm/carry.rs`): the
//! stash and pending-count buffers, the per-verify slot table and engaged words, and the
//! commit of a verdict as pending rows.
//!
//! A batched MTP verify that asks for write-on-accept runs, in every GDN layer that
//! engages, a carry kernel: it folds the rows the previous verdict accepted into H, then
//! verifies without writing any h-state and stashes its rows. The verdict then only
//! records how many of them each slot accepted (`pend`); they stay pending until the next
//! carry verify of that slot folds them, or until any other call that reads the state
//! folds every pending slot first (`gdn_carry_flush_pending`, called at the top of those
//! `Model` methods).
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - Once bound, the buffers never move (captured graphs bake their addresses).
//! - `CarryInner::mirror` is the host copy of the device `pend` array; every change is
//!   uploaded on the default stream before the next launch that reads it.
//! - A slot listed in `CarryInner::pending` has rows that only the stash holds: H must be
//!   folded (`gdn_carry_flush_pending`) before anything but a carry verify reads it.

use anyhow::Result;
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{GdnCarryBinding, VERIFY_WY_TABLE_SEQS};
use parking_lot::Mutex;

use super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-09-26: Presence of `METRALE_NO_GDN_CARRY` (any value, `0` included) turns the
/// carried-state verify off. Read once per process.
fn gdn_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_GDN_CARRY").is_none())
}

struct CarryBufs {
    stash: DevicePtr,
    pend: DevicePtr,
    slot_tab: DevicePtr,
    flags: DevicePtr,
    flush_tab: DevicePtr,
    flush_slots: DevicePtr,
    flush_k: metrale_gpu_runtime::gpu::KernelHandle,
    layers: usize,
    slots: usize,
    seq_floats: usize,
    nv: usize,
}

/// 2026-09-26: The last carry verify, consumed by its verdict.
struct LastCarry {
    slots: Vec<usize>,
    ks: Vec<usize>,
    h_ptrs: Vec<Vec<DevicePtr>>,
    /// 2026-09-26: `[layer][VERIFY_WY_TABLE_SEQS]`: whether that layer's carry kernel
    /// verified that batch position.
    engaged: Vec<bool>,
}

#[derive(Default)]
struct CarryInner {
    probed: bool,
    bufs: Option<CarryBufs>,
    mirror: Vec<u32>,
    /// 2026-09-26: `(slot, h pointer per GDN layer)` of every slot with pending rows.
    pending: Vec<(usize, Vec<DevicePtr>)>,
    last: Option<LastCarry>,
}

/// 2026-09-26: The carry state of one model; see the module header.
#[derive(Default)]
pub(crate) struct GdnCarry {
    inner: Mutex<CarryInner>,
}

impl TransformerModel {
    fn gdn_layer_indices(&self) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&i| self.config.layer_type(i) == LayerType::LinearAttention)
            .collect()
    }

    /// 2026-09-26: Allocate and bind the carry buffers on the first call. `false` when the
    /// model cannot carry (a GDN layer without the kernels, an FP16 h pool, no MTP pools).
    fn gdn_carry_bind(&self, inner: &mut CarryInner) -> Result<bool> {
        if inner.probed {
            return Ok(inner.bufs.is_some());
        }
        inner.probed = true;
        let gdn = self.gdn_layer_indices();
        let per_layer: Vec<Option<usize>> = gdn
            .iter()
            .map(|&i| self.layers[i].gdn_carry_seq_floats())
            .collect();
        let seq_floats = match per_layer.first() {
            Some(Some(f)) if per_layer.iter().all(|p| *p == Some(*f)) => *f,
            _ => return Ok(false),
        };
        let flush_k = metrale_model_layers::layers::try_kernel(
            self.gpu.as_ref(),
            "gated_delta_rule_carry",
            "gdn_carry_flush",
        );
        if flush_k.0 == 0
            || self.ssm_pool.mtp_slots == 0
            || self.ssm_pool.h_stored_bytes != self.ssm_pool.h_bytes
        {
            return Ok(false);
        }
        let layers = gdn.len();
        let slots = self.ssm_pool.mtp_slots + 1;
        let stash = self.gpu.alloc(layers * slots * seq_floats * 4)?;
        let pend = self.gpu.alloc(layers * slots * 4)?;
        self.gpu.memset(pend, 0, layers * slots * 4)?;
        let slot_tab = self.gpu.alloc(VERIFY_WY_TABLE_SEQS * 4)?;
        let flags = self.gpu.alloc(layers * VERIFY_WY_TABLE_SEQS * 4)?;
        self.gpu
            .memset(flags, 0, layers * VERIFY_WY_TABLE_SEQS * 4)?;
        let flush_tab = self.gpu.alloc(layers * VERIFY_WY_TABLE_SEQS * 8)?;
        let flush_slots = self.gpu.alloc(VERIFY_WY_TABLE_SEQS * 4)?;
        for (l, &i) in gdn.iter().enumerate() {
            self.layers[i].gdn_carry_bind(GdnCarryBinding {
                flag: flags.offset(l * VERIFY_WY_TABLE_SEQS * 4),
                stash: stash.offset(l * slots * seq_floats * 4),
                pend: pend.offset(l * slots * 4),
                slot_tab,
                seq_floats,
            });
        }
        let nv = self.config.linear_num_value_heads;
        tracing::info!(
            "GDN carry: bound {:.1} MB stash ({layers} GDN layers x {slots} slots)",
            (layers * slots * seq_floats * 4) as f64 / 1e6
        );
        inner.mirror = vec![0; layers * slots];
        inner.bufs = Some(CarryBufs {
            stash,
            pend,
            slot_tab,
            flags,
            flush_tab,
            flush_slots,
            flush_k,
            layers,
            slots,
            seq_floats,
            nv,
        });
        Ok(true)
    }

    fn carry_slot(&self, seq: &SequenceState) -> usize {
        seq.slot_idx.min(self.ssm_pool.mtp_slots)
    }

    /// 2026-09-26: Whether this batched verify runs carried. Called before the graph
    /// decision; clears the engaged words on the default stream.
    pub(super) fn gdn_carry_begin(&self, requested: bool, stream: u64) -> Result<bool> {
        let mut inner = self.gdn_carry.inner.lock();
        inner.last = None;
        if !requested || !gdn_carry_enabled() || !self.gdn_carry_bind(&mut inner)? {
            // 2026-09-26: This verify's parent kernels read H: nothing may stay pending.
            if !inner.pending.is_empty() {
                self.gdn_carry_flush_locked(&mut inner, stream)?;
            }
            return Ok(false);
        }
        let b = inner.bufs.as_ref().expect("bound");
        self.gpu
            .memset_async(b.flags, 0, b.layers * VERIFY_WY_TABLE_SEQS * 4, stream)?;
        Ok(true)
    }

    /// 2026-09-26: Stage the slot table of a carried verify: the batch's slots, then the
    /// ghost rows' slots of a borrowed graph, in table order.
    pub(super) fn gdn_carry_stage(
        &self,
        seqs: &[&mut SequenceState],
        ghosts: &[(u32, u32)],
        stream: u64,
    ) -> Result<()> {
        let inner = self.gdn_carry.inner.lock();
        let b = inner.bufs.as_ref().expect("bound");
        let slots: Vec<u8> = seqs
            .iter()
            .map(|s| self.carry_slot(s) as u32)
            .chain(
                ghosts
                    .iter()
                    .map(|&(g, _)| (g as usize).min(self.ssm_pool.mtp_slots) as u32),
            )
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.gpu.copy_h2d_async(&slots, b.slot_tab, stream)
    }

    fn upload_mirror(&self, inner: &CarryInner, stream: u64) -> Result<()> {
        let b = inner.bufs.as_ref().expect("bound");
        let bytes: Vec<u8> = inner.mirror.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, b.pend, stream)
    }

    /// 2026-09-26: After a carried verify's forward completed (its argmax was read back):
    /// every layer consumed the batch's pending rows, in its carry kernel or in the fold a
    /// declining layer runs. Records which layers engaged for the verdict.
    pub(super) fn gdn_carry_end(
        &self,
        seqs: &[&mut SequenceState],
        ks: &[usize],
        stream: u64,
    ) -> Result<()> {
        let mut inner = self.gdn_carry.inner.lock();
        let (layers, slots_n, flags) = {
            let b = inner.bufs.as_ref().expect("bound");
            (b.layers, b.slots, b.flags)
        };
        let mut raw = vec![0u8; layers * VERIFY_WY_TABLE_SEQS * 4];
        self.gpu.copy_d2h_on_stream(flags, &mut raw, stream)?;
        let engaged: Vec<bool> = raw.chunks(4).map(|c| c != [0, 0, 0, 0]).collect();
        let gdn = self.gdn_layer_indices();
        let mut slots = Vec::with_capacity(seqs.len());
        let mut h_ptrs = Vec::with_capacity(seqs.len());
        for seq in seqs {
            let slot = self.carry_slot(seq);
            for l in 0..layers {
                inner.mirror[l * slots_n + slot] = 0;
            }
            inner.pending.retain(|(s, _)| *s != slot);
            let mut hp = Vec::with_capacity(layers);
            for &i in &gdn {
                let st = seq.layer_states[i]
                    .as_any()
                    .downcast_ref::<metrale_model_layers::layer::SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("gdn_carry_end: layer {i} is not SSM"))?;
                hp.push(st.h_state);
            }
            slots.push(slot);
            h_ptrs.push(hp);
        }
        self.upload_mirror(&inner, stream)?;
        inner.last = Some(LastCarry {
            slots,
            ks: ks.to_vec(),
            h_ptrs,
            engaged,
        });
        Ok(())
    }

    /// 2026-09-26: Commit a carried verify's verdict: `accepted_rows[i]` rows (anchor
    /// included) of the sequence on `slots[i]`. Engaged layers get them as pending rows; a
    /// layer that declined ran its parent kernels, and a partial accept restores its H from
    /// Hi here. `Ok(false)` when the last verify was not carried.
    pub(super) fn gdn_carry_commit(&self, slots: &[usize], accepted_rows: &[u32]) -> Result<bool> {
        let mut inner = self.gdn_carry.inner.lock();
        let Some(last) = inner.last.take() else {
            return Ok(false);
        };
        let stream = self.gpu.default_stream();
        anyhow::ensure!(
            accepted_rows.len() == last.slots.len()
                && slots.len() == last.slots.len()
                && slots
                    .iter()
                    .zip(&last.slots)
                    .all(|(&s, &c)| s.min(self.ssm_pool.mtp_slots) == c),
            "gdn_carry_commit: verdict for slots {slots:?} after a carried verify of {:?}",
            last.slots
        );
        let slots_n = inner.bufs.as_ref().expect("bound").slots;
        let mut restores = Vec::new();
        for (b, (&slot, &rows)) in last.slots.iter().zip(accepted_rows).enumerate() {
            anyhow::ensure!(
                rows >= 1 && rows as usize <= last.ks[b] && rows <= 4,
                "gdn_carry_commit: {rows} accepted rows of a {}-row verify",
                last.ks[b]
            );
            for l in 0..last.h_ptrs[b].len() {
                if last.engaged[l * VERIFY_WY_TABLE_SEQS + b] {
                    inner.mirror[l * slots_n + slot] = rows;
                } else if (rows as usize) < last.ks[b] {
                    restores.push(StateCopy {
                        src: self.ssm_pool.h_intermediate(l, slot, rows as usize - 1),
                        dst: last.h_ptrs[b][l],
                        bytes: self.ssm_pool.h_stored_bytes,
                    });
                }
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &restores, &[], stream)?;
        self.upload_mirror(&inner, stream)?;
        for (b, &slot) in last.slots.iter().enumerate() {
            inner.pending.push((slot, last.h_ptrs[b].clone()));
        }
        Ok(true)
    }

    /// 2026-09-26: Fold every pending row into H. A no-op when nothing is pending.
    pub(crate) fn gdn_carry_flush_pending(&self) -> Result<()> {
        let mut inner = self.gdn_carry.inner.lock();
        if inner.pending.is_empty() {
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        self.gdn_carry_flush_locked(&mut inner, stream)
    }

    fn gdn_carry_flush_locked(&self, inner: &mut CarryInner, stream: u64) -> Result<()> {
        let pending = std::mem::take(&mut inner.pending);
        let b = inner.bufs.as_ref().expect("bound");
        let (layers, slots_n) = (b.layers, b.slots);
        for chunk in pending.chunks(VERIFY_WY_TABLE_SEQS) {
            let mut tab = vec![0u64; layers * VERIFY_WY_TABLE_SEQS];
            for (j, (_, hp)) in chunk.iter().enumerate() {
                for (l, p) in hp.iter().enumerate() {
                    tab[l * VERIFY_WY_TABLE_SEQS + j] = p.0;
                }
            }
            let tab_bytes: Vec<u8> = tab.iter().flat_map(|v| v.to_le_bytes()).collect();
            let slot_bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|(s, _)| (*s as u32).to_le_bytes())
                .collect();
            self.gpu.copy_h2d_async(&tab_bytes, b.flush_tab, stream)?;
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
        }
        for (slot, _) in &pending {
            for l in 0..layers {
                inner.mirror[l * slots_n + slot] = 0;
            }
        }
        self.upload_mirror(inner, stream)
    }
}
