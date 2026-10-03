// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Model side of the carried-state GDN verify (`qwen3_ssm/carry.rs`): the
//! stash and pending-count buffers, the per-verify slot table and engaged words, and the
//! commit of a verdict as pending rows.
//!
//! A batched MTP verify that asks for write-on-accept runs, in every GDN layer that
//! engages, a carry kernel: it folds the rows the previous verdicts accepted into its copy
//! of H, verifies without writing any intermediate state and stashes its rows. The verdict
//! then only records how many of them each slot accepted (`pend`). They stay pending,
//! stacked up to `GDN_CARRY_CAP` rows, until a carry verify of that slot writes the state
//! back (every verify below `GDN_CARRY_LAZY_MIN_SEQS` sequences; from there only when its
//! own rows might not fit behind them), or until any other call that reads the state folds every pending slot
//! first (`gdn_carry_flush_pending`, called at the top of those `Model` methods).
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
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layer::{GdnCarryBinding, VERIFY_WY_TABLE_SEQS};
use metrale_model_layers::layers::ops::{GDN_CARRY_EAGER_MAX_PENDING, GDN_CARRY_LAZY_MIN_SEQS};
use parking_lot::Mutex;

use super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::types::TransformerModel;
use crate::traits::SequenceState;

mod flush;
use flush::state_table;

/// 2026-09-26: Presence of `METRALE_NO_GDN_CARRY` (any value, `0` included) turns the
/// carried-state verify off. Read once per process.
fn gdn_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_GDN_CARRY").is_none())
}

struct CarryBufs {
    stash: DevicePtr,
    conv_stash: DevicePtr,
    pend: DevicePtr,
    slot_tab: DevicePtr,
    conv_tab: DevicePtr,
    flags: DevicePtr,
    flush_tab: DevicePtr,
    flush_conv_tab: DevicePtr,
    flush_slots: DevicePtr,
    flush_k: KernelHandle,
    conv_flush_k: KernelHandle,
    layers: usize,
    slots: usize,
    seq_floats: usize,
    conv_seq_elems: usize,
    nv: usize,
    conv_dim: usize,
    d_conv: usize,
}

/// 2026-09-26: One sequence's GDN state pointers, per GDN layer.
#[derive(Clone)]
struct SlotStates {
    slot: usize,
    h: Vec<DevicePtr>,
    conv: Vec<DevicePtr>,
}

/// 2026-09-26: The last carry verify, consumed by its verdict.
struct LastCarry {
    seqs: Vec<SlotStates>,
    ks: Vec<usize>,
    /// 2026-09-26: `[layer][VERIFY_WY_TABLE_SEQS]`: whether that layer's carry kernel
    /// verified that batch position.
    engaged: Vec<bool>,
}

#[derive(Default)]
struct CarryInner {
    probed: bool,
    bufs: Option<CarryBufs>,
    mirror: Vec<u32>,
    /// 2026-09-26: Every slot with pending rows.
    pending: Vec<SlotStates>,
    last: Option<LastCarry>,
    /// 2026-09-26: Slots whose last verdict `gdn_carry_commit` committed (pending rows and
    /// any restore); `commit_accepted_prefix` copies nothing for them.
    committed: Vec<usize>,
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
        let kernel = |name| {
            metrale_model_layers::layers::try_kernel(
                self.gpu.as_ref(),
                "gated_delta_rule_carry",
                name,
            )
        };
        let (flush_k, conv_flush_k) = (
            metrale_model_layers::layers::qwen3_ssm::carry_flush_kernel(self.gpu.as_ref()),
            kernel("gdn_carry_conv_flush"),
        );
        if flush_k.0 == 0
            || conv_flush_k.0 == 0
            || self.ssm_pool.mtp_slots == 0
            || self.ssm_pool.h_stored_bytes != self.ssm_pool.h_bytes
        {
            return Ok(false);
        }
        let layers = gdn.len();
        let slots = self.ssm_pool.mtp_slots + 1;
        let c = &self.config;
        let conv_dim = c.linear_num_key_heads * c.linear_key_head_dim * 2
            + c.linear_num_value_heads * c.linear_value_head_dim;
        let conv_seq_elems = metrale_model_layers::layers::ops::gdn_carry_conv_seq_elems(conv_dim);
        // 2026-10-01: The sizes the preflight reserve plans (`GdnCarrySizes`, one source).
        let sz = metrale_model_layers::layers::ops::GdnCarrySizes::new(
            layers,
            slots,
            seq_floats,
            conv_seq_elems,
            VERIFY_WY_TABLE_SEQS,
        );
        let stash = self.gpu.alloc(sz.stash)?;
        let conv_stash = self.gpu.alloc(sz.conv_stash)?;
        let conv_tab = self.gpu.alloc(sz.conv_tab)?;
        let flush_conv_tab = self.gpu.alloc(sz.flush_conv_tab)?;
        let pend = self.gpu.alloc(sz.pend)?;
        self.gpu.memset(pend, 0, sz.pend)?;
        let slot_tab = self.gpu.alloc(sz.slot_tab)?;
        let flags = self.gpu.alloc(sz.flags)?;
        self.gpu.memset(flags, 0, sz.flags)?;
        let flush_tab = self.gpu.alloc(sz.flush_tab)?;
        let flush_slots = self.gpu.alloc(sz.flush_slots)?;
        for (l, &i) in gdn.iter().enumerate() {
            self.layers[i].gdn_carry_bind(GdnCarryBinding {
                flag: flags.offset(l * VERIFY_WY_TABLE_SEQS * 4),
                stash: stash.offset(l * slots * seq_floats * 4),
                pend: pend.offset(l * slots * 4),
                slot_tab,
                seq_floats,
                conv_stash: conv_stash.offset(l * slots * conv_seq_elems * 2),
                conv_seq_elems,
                conv_tab: conv_tab.offset(l * VERIFY_WY_TABLE_SEQS * 8),
            });
        }
        let nv = self.config.linear_num_value_heads;
        tracing::info!(
            "GDN carry: bound {:.1} MB stash ({layers} GDN layers x {slots} slots)",
            (layers * slots * (seq_floats * 4 + conv_seq_elems * 2)) as f64 / 1e6
        );
        inner.mirror = vec![0; layers * slots];
        inner.bufs = Some(CarryBufs {
            stash,
            conv_stash,
            pend,
            slot_tab,
            conv_tab,
            flags,
            flush_tab,
            flush_conv_tab,
            flush_slots,
            flush_k,
            conv_flush_k,
            layers,
            slots,
            seq_floats,
            conv_seq_elems,
            nv,
            conv_dim,
            d_conv: c.linear_conv_kernel_dim,
        });
        Ok(true)
    }

    /// 2026-09-30: Bind the carry buffers now, if the model can carry; the circuit executor's
    /// build reads the bindings from the layers.
    pub(super) fn gdn_carry_bind_now(&self) -> Result<bool> {
        let mut inner = self.gdn_carry.inner.lock();
        self.gdn_carry_bind(&mut inner)
    }

    fn carry_slot(&self, seq: &SequenceState) -> usize {
        seq.slot_idx.min(self.ssm_pool.mtp_slots)
    }

    /// 2026-09-26: Whether this batched verify runs carried. Called before the graph
    /// decision; clears the engaged words on the default stream. `ks` gives the runs the
    /// layers launch (maximal groups of adjacent sequences with equal `ks`, as
    /// `batched_conv_gdn_route` forms them): a slot in a run narrower than
    /// `GDN_CARRY_LAZY_MIN_SEQS` goes to an eager kernel, which folds at most
    /// `GDN_CARRY_EAGER_MAX_PENDING` rows, so a slot holding more is folded here first.
    pub(super) fn gdn_carry_begin(
        &self,
        requested: bool,
        seqs: &[&mut SequenceState],
        ks: &[usize],
        stream: u64,
    ) -> Result<bool> {
        let mut inner = self.gdn_carry.inner.lock();
        inner.last = None;
        if !requested || !gdn_carry_enabled() || !self.gdn_carry_bind(&mut inner)? {
            // 2026-09-26: This verify's parent kernels read H: nothing may stay pending.
            if !inner.pending.is_empty() {
                self.gdn_carry_flush_locked(&mut inner, stream, |_| true)?;
            }
            return Ok(false);
        }
        let (layers, slots_n, flags) = {
            let b = inner.bufs.as_ref().expect("bound");
            (b.layers, b.slots, b.flags)
        };
        let mut eager_deep = Vec::new();
        let mut g0 = 0;
        while g0 < ks.len() {
            let g1 = (g0..ks.len())
                .find(|&i| ks[i] != ks[g0])
                .unwrap_or(ks.len());
            if g1 - g0 < GDN_CARRY_LAZY_MIN_SEQS {
                for seq in &seqs[g0..g1] {
                    let slot = self.carry_slot(seq);
                    if (0..layers).any(|l| {
                        inner.mirror[l * slots_n + slot] as usize > GDN_CARRY_EAGER_MAX_PENDING
                    }) {
                        eager_deep.push(slot);
                    }
                }
            }
            g0 = g1;
        }
        if !eager_deep.is_empty() {
            self.gdn_carry_flush_locked(&mut inner, stream, |r| eager_deep.contains(&r.slot))?;
        }
        self.gpu
            .memset_async(flags, 0, layers * VERIFY_WY_TABLE_SEQS * 4, stream)?;
        Ok(true)
    }

    /// 2026-09-26: This sequence's h and conv state per GDN layer.
    fn slot_states(&self, seq: &SequenceState) -> Result<SlotStates> {
        let gdn = self.gdn_layer_indices();
        let mut out = SlotStates {
            slot: self.carry_slot(seq),
            h: Vec::with_capacity(gdn.len()),
            conv: Vec::with_capacity(gdn.len()),
        };
        for &i in &gdn {
            let st = seq.layer_states[i]
                .as_any()
                .downcast_ref::<metrale_model_layers::layer::SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GDN carry: layer {i} is not SSM"))?;
            out.h.push(st.h_state);
            out.conv.push(st.conv_state);
        }
        Ok(out)
    }

    /// 2026-09-26: Stage the slot table and the conv-state table of a carried verify: the
    /// batch's entries, then the ghost rows' entries of a borrowed graph, in table order.
    pub(super) fn gdn_carry_stage(
        &self,
        seqs: &[&mut SequenceState],
        ghosts: &[(u32, u32)],
        stream: u64,
    ) -> Result<()> {
        let inner = self.gdn_carry.inner.lock();
        let b = inner.bufs.as_ref().expect("bound");
        let mut rows: Vec<SlotStates> = seqs
            .iter()
            .map(|s| self.slot_states(s))
            .collect::<Result<_>>()?;
        for &(g, _) in ghosts {
            let slot = g as usize;
            rows.push(SlotStates {
                slot: slot.min(self.ssm_pool.mtp_slots),
                h: Vec::new(),
                conv: (0..b.layers)
                    .map(|l| self.ssm_pool.conv_state(l, slot))
                    .collect(),
            });
        }
        let slots: Vec<u8> = rows
            .iter()
            .flat_map(|r| (r.slot as u32).to_le_bytes())
            .collect();
        self.gpu.copy_h2d_async(&slots, b.slot_tab, stream)?;
        let tab = state_table(&rows, b.layers, |r| &r.conv);
        self.gpu.copy_h2d_async(&tab, b.conv_tab, stream)
    }

    fn upload_mirror(&self, inner: &CarryInner, stream: u64) -> Result<()> {
        let b = inner.bufs.as_ref().expect("bound");
        let bytes: Vec<u8> = inner.mirror.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, b.pend, stream)
    }

    /// 2026-09-26: After a carried verify's forward completed (its argmax was read back).
    /// A layer that engaged kept the batch's pending rows unless it wrote the state back
    /// (its engaged word says which); a declining layer folded them. Records which layers
    /// engaged for the verdict.
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
        // 2026-09-26: 0 = the layer declined this position, 1 = carried with the rows kept
        // pending, 2 = carried and written back.
        let words: Vec<u32> = raw
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let engaged: Vec<bool> = words.iter().map(|&w| w != 0).collect();
        let mut rows = Vec::with_capacity(seqs.len());
        for (b, seq) in seqs.iter().enumerate() {
            let r = self.slot_states(seq)?;
            for l in 0..layers {
                if words[l * VERIFY_WY_TABLE_SEQS + b] != 1 {
                    inner.mirror[l * slots_n + r.slot] = 0;
                }
            }
            // 2026-09-26: A slot still holding rows stays listed, so a later fold reaches
            // it even if its verdict never commits.
            inner.pending.retain(|p| p.slot != r.slot);
            if (0..layers).any(|l| inner.mirror[l * slots_n + r.slot] != 0) {
                inner.pending.push(r.clone());
            }
            rows.push(r);
        }
        self.upload_mirror(&inner, stream)?;
        inner.last = Some(LastCarry {
            seqs: rows,
            ks: ks.to_vec(),
            engaged,
        });
        Ok(())
    }

    /// 2026-09-26: Commit a carried verify's verdict: `accepted_rows[i]` rows (anchor
    /// included) of the sequence on `slots[i]`. Engaged layers get them as pending rows; a
    /// layer that declined ran its parent kernels, and a partial accept restores its h and
    /// conv state from the intermediates here. `commit_accepted_prefix` then copies nothing
    /// for these slots. `Ok(false)` when the last verify was not carried.
    pub(super) fn gdn_carry_commit(&self, slots: &[usize], accepted_rows: &[u32]) -> Result<bool> {
        let mut inner = self.gdn_carry.inner.lock();
        inner.committed.clear();
        let Some(last) = inner.last.take() else {
            return Ok(false);
        };
        let stream = self.gpu.default_stream();
        anyhow::ensure!(
            accepted_rows.len() == last.seqs.len()
                && slots.len() == last.seqs.len()
                && slots
                    .iter()
                    .zip(&last.seqs)
                    .all(|(&s, r)| s.min(self.ssm_pool.mtp_slots) == r.slot),
            "gdn_carry_commit: verdict for slots {slots:?} after a carried verify of {:?}",
            last.seqs.iter().map(|r| r.slot).collect::<Vec<_>>()
        );
        let (slots_n, d_conv, conv_dim) = {
            let b = inner.bufs.as_ref().expect("bound");
            (b.slots, b.d_conv, b.conv_dim)
        };
        let (mut h_restores, mut conv_restores) = (Vec::new(), Vec::new());
        for (b, (r, &rows)) in last.seqs.iter().zip(accepted_rows).enumerate() {
            anyhow::ensure!(
                rows >= 1 && rows as usize <= last.ks[b] && rows <= 4,
                "gdn_carry_commit: {rows} accepted rows of a {}-row verify",
                last.ks[b]
            );
            for l in 0..r.h.len() {
                if last.engaged[l * VERIFY_WY_TABLE_SEQS + b] {
                    inner.mirror[l * slots_n + r.slot] += rows;
                } else if (rows as usize) < last.ks[b] {
                    let t = rows as usize - 1;
                    h_restores.push(StateCopy {
                        src: self.ssm_pool.h_intermediate(l, r.slot, t),
                        dst: r.h[l],
                        bytes: self.ssm_pool.h_stored_bytes,
                    });
                    conv_restores.push(StateCopy {
                        src: self.ssm_pool.conv_intermediate(l, r.slot, t),
                        dst: r.conv[l],
                        bytes: conv_dim * d_conv * 4,
                    });
                }
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_restores, &conv_restores, stream)?;
        self.upload_mirror(&inner, stream)?;
        inner.committed = slots.to_vec();
        inner
            .pending
            .retain(|p| last.seqs.iter().all(|r| r.slot != p.slot));
        inner.pending.extend(last.seqs);
        Ok(true)
    }

    /// 2026-09-26: Whether `slot`'s last verdict was committed by `gdn_carry_commit`; the
    /// entry is consumed.
    pub(super) fn gdn_carry_take_committed(&self, slot: usize) -> bool {
        let mut inner = self.gdn_carry.inner.lock();
        match inner.committed.iter().position(|&s| s == slot) {
            Some(p) => {
                inner.committed.swap_remove(p);
                true
            }
            None => false,
        }
    }
}
