// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: SNAPZ (`METRALE_GLM_KDA_SNAPZ=1`), the model side: a batched speculative verify
//! whose verdict keeps only a sequence's first row skips the commit's copy of that row's
//! snapshot into the live recurrent and conv state (two state blobs per KDA layer). The state
//! stays in the sequence's first snapshot slot, and the next batched verify's KDA launches read
//! it there: the flag word of the sequence's batch position says so (GLM layer
//! `steps/kda_snapz.rs`, kernel `kda_seqs_src`). Every other reader lands the skipped copy
//! first (`kda_snapz_flush_all`, inside `gdn_carry_flush_pending`, which the `Model` methods
//! that read the state already call), as does a batched verify that holds the slot at a
//! position the flags cannot name.
//!
//! The bookkeeping is pure ([`book::Book`]); this module does the I/O.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - Bound once, before the first batched verify's graph decision; the flag buffer never
//!   moves (captured graphs bake its address).
//! - The device flag table is the book's last upload, written on the default stream before
//!   the verify that reads it.
//! - A deferred slot's live state is wrong until its copy lands: nothing but a flagged batched
//!   verify reads it before that ([`book`] invariants).

use anyhow::{Result, bail, ensure};
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{KdaSnapzBinding, SsmLayerState, VERIFY_WY_TABLE_SEQS};
use parking_lot::Mutex;

use super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::types::TransformerModel;
use crate::traits::SequenceState;

mod book;
use book::Book;

/// 2026-10-16: `METRALE_GLM_KDA_SNAPZ=1` turns SNAPZ on; read once per process. Unset (or any
/// other value) keeps every commit eager.
pub(crate) fn kda_snapz() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_KDA_SNAPZ").as_deref() == Ok("1"))
}

/// 2026-10-16: A skipped commit: the copies `commit_accepted_prefix` would have issued.
struct Plan {
    h: Vec<StateCopy>,
    conv: Vec<StateCopy>,
}

struct SnapzInner {
    /// 2026-10-16: The flag buffer once bound, or why binding failed (sticky).
    bound: Option<std::result::Result<DevicePtr, String>>,
    book: Book<Plan>,
    said_defer: bool,
    said_flush: bool,
}

/// 2026-10-16: SNAPZ state of one model; see the module header.
pub(crate) struct KdaSnapz {
    inner: Mutex<SnapzInner>,
}

impl Default for KdaSnapz {
    fn default() -> Self {
        Self {
            inner: Mutex::new(SnapzInner {
                bound: None,
                book: Book::new(VERIFY_WY_TABLE_SEQS),
                said_defer: false,
                said_flush: false,
            }),
        }
    }
}

impl TransformerModel {
    /// 2026-10-16: The flag buffer, bound on the first call. With the lever on, a model whose
    /// batched verify cannot read a deferred state fails here, at its first batched verify.
    fn kda_snapz_bound(&self, inner: &mut SnapzInner) -> Result<DevicePtr> {
        if inner.bound.is_none() {
            inner.bound = Some(self.kda_snapz_try_bind().map_err(|e| format!("{e:#}")));
        }
        match inner.bound.as_ref().expect("set above") {
            Ok(p) => Ok(*p),
            Err(e) => bail!("METRALE_GLM_KDA_SNAPZ=1: {e}"),
        }
    }

    fn kda_snapz_try_bind(&self) -> Result<DevicePtr> {
        use metrale_model_layers::ssm_reserve::SsmRollbackMode;
        ensure!(
            self.ssm_pool.rollback_mode != SsmRollbackMode::Replay,
            "replay rollback keeps no snapshot to defer to"
        );
        ensure!(self.ssm_pool.mtp_slots > 0, "no MTP snapshot pools");
        ensure!(
            self.dflash_hidden_save.is_none(),
            "a DFlash verify is not supported"
        );
        let kda: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.config.layer_type(i) == LayerType::LinearAttention)
            .collect();
        ensure!(!kda.is_empty(), "the model has no recurrent layer");
        let bytes = VERIFY_WY_TABLE_SEQS * 4;
        let flags = self.gpu.alloc(bytes)?;
        self.gpu.memset(flags, 0, bytes)?;
        let b = KdaSnapzBinding {
            flags,
            h_bytes: self.ssm_pool.h_stored_bytes,
            conv_bytes: self.ssm_pool.conv_bytes,
        };
        let refused: Vec<usize> = kda
            .iter()
            .copied()
            .filter(|&i| !self.layers[i].kda_snapz_bind(b))
            .collect();
        ensure!(
            refused.is_empty(),
            "layers {refused:?} refuse the binding (no batched verify reading a deferred \
             state, no resolve kernel, or a state size other than the pool's)"
        );
        tracing::warn!(
            "METRALE_GLM_KDA_SNAPZ: bound on {} recurrent layers (rank-local, first batched \
             verify)",
            kda.len()
        );
        Ok(flags)
    }

    /// 2026-10-16: Issue the copies of `plans` on `stream`, h then conv.
    fn kda_snapz_land(&self, plans: &[Plan], stream: u64) -> Result<()> {
        let h: Vec<StateCopy> = plans.iter().flat_map(|p| p.h.iter().copied()).collect();
        let conv: Vec<StateCopy> = plans.iter().flat_map(|p| p.conv.iter().copied()).collect();
        run_ssm_state_copies(self.gpu.as_ref(), &h, &conv, stream)
    }

    /// 2026-10-16: Before a batched verify of `seqs` (and its graph decision): land the
    /// deferred commits this verify cannot read through a flag, and upload the flag table when
    /// it changed. Both on the default stream, ahead of the verify's launches.
    pub(crate) fn kda_snapz_begin(&self, seqs: &[&mut SequenceState]) -> Result<()> {
        if !kda_snapz() {
            return Ok(());
        }
        ensure!(
            self.circuit.read().is_none(),
            "METRALE_GLM_KDA_SNAPZ=1: the circuit executor's verify does not read deferred \
             states"
        );
        let mut inner = self.gdn_carry.snapz.inner.lock();
        let flags = self.kda_snapz_bound(&mut inner)?;
        let slots: Vec<usize> = seqs.iter().map(|s| s.slot_idx).collect();
        let begin = inner.book.begin(&slots);
        let stream = self.gpu.default_stream();
        if !begin.flush.is_empty() {
            self.kda_snapz_land(&begin.flush, stream)?;
        }
        if let Some(table) = begin.upload {
            let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.gpu.copy_h2d_async(&bytes, flags, stream)?;
        }
        Ok(())
    }

    /// 2026-10-16: After that verify (`ok`: it returned `Ok`): its flagged slots no longer
    /// have a deferred commit, and its slots at flaggable positions may defer their commit.
    pub(crate) fn kda_snapz_end(&self, seqs: &[&mut SequenceState], ok: bool) {
        if !kda_snapz() {
            return;
        }
        let slots: Vec<usize> = seqs.iter().map(|s| s.slot_idx).collect();
        self.gdn_carry.snapz.inner.lock().book.end(ok, &slots);
    }

    /// 2026-10-16: Consume `slot`'s arm from the last batched verify: `true` when its commit
    /// may be deferred.
    pub(crate) fn kda_snapz_take_armed(&self, slot: usize) -> bool {
        kda_snapz() && self.gdn_carry.snapz.inner.lock().book.take_armed(slot)
    }

    /// 2026-10-16: Defer `seq`'s commit of its first verify row (an armed slot). `Ok(false)`:
    /// it cannot (a slot on the shared snapshot dummy, or a layer without a first snapshot),
    /// and the caller commits eagerly. A snapshot other than the commit's source is an error:
    /// the next verify would read the wrong state.
    pub(crate) fn kda_snapz_defer(&self, seq: &mut SequenceState) -> Result<bool> {
        if seq.slot_idx >= self.ssm_pool.mtp_slots {
            return Ok(false);
        }
        let (h, conv) = self.ssm_commit_plan(seq, 0, true)?;
        let mut li = 0usize;
        for (i, st) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = st
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("SNAPZ: layer {i} is not SSM"))?;
            let (Some(&hs), Some(&cs)) = (
                ssm.h_state_intermediates.first(),
                ssm.conv_state_intermediates.first(),
            ) else {
                return Ok(false);
            };
            ensure!(
                h.get(li).is_some_and(|c| c.src == hs) && conv.get(li).is_some_and(|c| c.src == cs),
                "SNAPZ: layer {i} of slot {} snapshots at {hs:?}/{cs:?}, not at the commit's source",
                seq.slot_idx
            );
            li += 1;
        }
        let mut inner = self.gdn_carry.snapz.inner.lock();
        inner
            .book
            .defer(seq.slot_idx, Plan { h, conv })
            .map_err(anyhow::Error::msg)?;
        if !std::mem::replace(&mut inner.said_defer, true) {
            tracing::warn!(
                "METRALE_GLM_KDA_SNAPZ: first commit deferred (rank-local, slot {})",
                seq.slot_idx
            );
        }
        Ok(true)
    }

    /// 2026-10-16: Land every deferred commit and wait for it: the state is then live for any
    /// reader, on any stream. A no-op (no lock) with the lever off, and with nothing pending.
    pub(crate) fn kda_snapz_flush_all(&self) -> Result<()> {
        if !kda_snapz() {
            return Ok(());
        }
        let mut inner = self.gdn_carry.snapz.inner.lock();
        if inner.book.is_idle() {
            return Ok(());
        }
        let plans = inner.book.flush_all();
        if plans.is_empty() {
            return Ok(());
        }
        if !std::mem::replace(&mut inner.said_flush, true) {
            tracing::warn!(
                "METRALE_GLM_KDA_SNAPZ: first deferred commit landed outside a batched verify \
                 (rank-local, {} slots)",
                plans.len()
            );
        }
        let stream = self.gpu.default_stream();
        self.kda_snapz_land(&plans, stream)?;
        self.gpu.synchronize(stream)
    }
}
