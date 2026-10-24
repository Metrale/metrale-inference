// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Prepare one MoE layer's tp-layout routed experts on the host in parallel, a wave
//! at a time, from `O_DIRECT` shard reads (`METRALE_GLM_EXPERT_FAST_LOAD`, default on).
//!
//! **WHY.** Under `--moe-expert-layout tp` every routed-expert tensor is deferred, and the
//! binder read each one with its own buffered `open + seek + read_exact` into a fresh `Vec` on
//! the loading thread, then sliced it (and quantized the MTP layer's 16-bit experts) before the
//! upload. GLM-5.3 TP=3 boots spent 213-243 s in the 45-layer build at ~0.3 GB/s of expert reads,
//! against 1.85 GB/s for the fast weight loader (DEV-LOOP-SPEED.md item 3). Here a worker pool
//! does the reads, slices and quantizations of a wave of [`WAVE`] experts before their uploads;
//! the uploads stay on the loading thread, in expert order, into the same slab.
//!
//! The bytes are the same: [`ExpertBytes`] is the one place the two paths differ (buffered or
//! `O_DIRECT` reads of the same ranges), and `expert_tp::sliced_proj` is shared, so every
//! slice, quantization and pack is the code the sequential path runs. `expert_prepare_tests`
//! proves the prepared projections equal to the sequential ones, and `METRALE_GLM_EXPERT_DIGEST`
//! (`expert_digest`) compares the uploaded device bytes of two real loads.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - The `O_DIRECT` source reads deferred tensors only; a resident routed-expert tensor (which
//!   the tp layout's defer hook never leaves) is an error, not a device read-back.
//! - A prepared expert is taken once; a second take, or an id outside the layer, is an error.
//! - At most one wave's host projections are held at a time.

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use super::expert_tp::{HostProj, host_expert, read_all, read_range};
use super::*;
use crate::glm5next_mlp::expert_tp::ExpertSlice;

/// 2026-10-24: Where `expert_tp::sliced_proj` gets a routed expert's tensor bytes.
pub(super) trait ExpertBytes {
    /// 2026-10-24: Bytes `[start, start + len)` of tensor `name`.
    fn range(&self, name: &str, start: usize, len: usize) -> Result<Vec<u8>>;
    /// 2026-10-24: All of tensor `name`.
    fn all(&self, name: &str) -> Result<Vec<u8>>;
    /// 2026-10-24: The activation scale of projection `base` of `layer` (`act_scale`).
    fn input_scale(&self, layer: usize, base: &str) -> Result<Option<f32>>;
}

/// 2026-10-24: The sequential source: a buffered shard read per deferred tensor, a device
/// read-back per resident one (the reads before `METRALE_GLM_EXPERT_FAST_LOAD`).
pub(super) struct Buffered<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub store: &'a WeightStore,
}

impl ExpertBytes for Buffered<'_> {
    fn range(&self, name: &str, start: usize, len: usize) -> Result<Vec<u8>> {
        read_range(self.gpu, self.store, name, start, len)
    }
    fn all(&self, name: &str) -> Result<Vec<u8>> {
        read_all(self.gpu, self.store, name)
    }
    fn input_scale(&self, layer: usize, base: &str) -> Result<Option<f32>> {
        input_scale(self.gpu, self.store, layer, base)
    }
}

/// 2026-10-24: The parallel source: `O_DIRECT` reads of deferred tensors, safe to share across
/// worker threads; counts the bytes it reads.
pub(super) struct Direct<'a> {
    pub store: &'a WeightStore,
    pub bytes: AtomicU64,
}

impl Direct<'_> {
    fn deferred(&self, name: &str) -> Result<&metrale_model_weights::weights::DeferredTensor> {
        self.store.deferred(name).with_context(|| {
            format!(
                "{name}: the O_DIRECT expert reader reads deferred tensors only; this one is \
                 resident (METRALE_GLM_EXPERT_FAST_LOAD=0 reads it back from the device)"
            )
        })
    }
    fn direct(
        &self,
        d: &metrale_model_weights::weights::DeferredTensor,
        name: &str,
        start: usize,
        len: usize,
    ) -> Result<Vec<u8>> {
        let b = d
            .read_host_range_direct(start, len)
            .with_context(|| format!("{name}: O_DIRECT read of the deferred expert"))?;
        self.bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
        Ok(b)
    }
}

impl ExpertBytes for Direct<'_> {
    fn range(&self, name: &str, start: usize, len: usize) -> Result<Vec<u8>> {
        self.direct(self.deferred(name)?, name, start, len)
    }
    fn all(&self, name: &str) -> Result<Vec<u8>> {
        let d = self.deferred(name)?;
        self.direct(d, name, 0, d.byte_size())
    }
    fn input_scale(&self, layer: usize, base: &str) -> Result<Option<f32>> {
        let name = qualify(layer, &format!("{base}.input_scale"));
        act_scale::input_scale_by(
            self.store,
            layer,
            base,
            |d| self.direct(d, &name, 0, d.byte_size()),
            |_, n| bail!("{n}: resident; the O_DIRECT expert reader reads deferred tensors only"),
        )
    }
}

/// 2026-10-24: How the tp-layout binder reads routed experts, resolved once from the settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpertLoad {
    /// 2026-10-24: `METRALE_GLM_EXPERT_FAST_LOAD`: prepare each layer's experts with [`Prepared`].
    pub fast: bool,
    /// 2026-10-24: `METRALE_GLM_EXPERT_DIGEST`: the file `expert_digest` appends to.
    pub digest: Option<std::path::PathBuf>,
}

impl ExpertLoad {
    /// 2026-10-24: From the startup settings.
    pub(crate) fn from_settings(s: &metrale_settings::InferenceServerSettings) -> Self {
        Self {
            fast: s.storage.glm_expert_fast_load,
            digest: s
                .telemetry
                .glm_expert_digest
                .as_deref()
                .map(std::path::PathBuf::from),
        }
    }
}

/// 2026-10-24: The experts one [`Prepared`] wave reads at once: enough to keep every rayon
/// worker on an `O_DIRECT` read, few enough that a wave's host copies stay ~0.3 GB (a whole
/// GLM-5.3 layer is ~1.35 GB per rank, held at once and then kept by the allocator's per-thread
/// arenas after the free).
pub(super) const WAVE: usize = 64;

/// 2026-10-24: One layer's routed experts `ids`, prepared on the host for the tp layout, a
/// [`WAVE`] at a time as the binder takes them.
pub(super) struct Prepared<'a> {
    store: &'a WeightStore,
    layer: usize,
    slice: ExpertSlice,
    ids: Range<usize>,
    state: Mutex<Waves>,
}

/// 2026-10-24: The current wave's projections and the layer's running totals.
struct Waves {
    wave: Range<usize>,
    slots: Vec<Option<[HostProj; 3]>>,
    taken: Vec<bool>,
    bytes: u64,
    secs: f64,
}

impl<'a> Prepared<'a> {
    /// 2026-10-24: Nothing is read until the first [`Self::take`].
    pub(super) fn new(
        store: &'a WeightStore,
        layer: usize,
        ids: Range<usize>,
        s: &ExpertSlice,
    ) -> Self {
        let state = Waves {
            wave: ids.start..ids.start,
            slots: Vec::new(),
            taken: vec![false; ids.len()],
            bytes: 0,
            secs: 0.0,
        };
        Self {
            store,
            layer,
            slice: *s,
            ids,
            state: Mutex::new(state),
        }
    }

    /// 2026-10-24: Expert `id`'s projections (gate, up, down), handed over once. An id outside
    /// the current wave starts the next one at `id`: read, slice and (for 16-bit experts)
    /// quantize up to [`WAVE`] experts on the rayon pool from [`Direct`] reads. The first
    /// error by id is the error.
    pub(super) fn take(&self, id: usize) -> Result<[HostProj; 3]> {
        if !self.ids.contains(&id) {
            bail!(
                "prepared experts: {id} is outside this layer's {:?}",
                self.ids
            );
        }
        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("prepared experts: lock poisoned"))?;
        if std::mem::replace(&mut st.taken[id - self.ids.start], true) {
            bail!("prepared experts: expert {id} was taken twice");
        }
        if !st.wave.contains(&id) {
            self.read_wave(&mut st, id..(id + WAVE).min(self.ids.end))?;
        }
        let at = id - st.wave.start;
        st.slots[at]
            .take()
            .with_context(|| format!("prepared experts: expert {id} was not prepared"))
    }

    fn read_wave(&self, st: &mut Waves, wave: Range<usize>) -> Result<()> {
        let t = std::time::Instant::now();
        let src = Direct {
            store: self.store,
            bytes: AtomicU64::new(0),
        };
        let (layer, s) = (self.layer, &self.slice);
        st.slots = wave
            .clone()
            .into_par_iter()
            .map(|id| host_expert(self.store, &src, layer, id, s).map(Some))
            .collect::<Result<Vec<_>>>()?;
        st.wave = wave;
        st.bytes += src.bytes.load(Ordering::Relaxed);
        st.secs += t.elapsed().as_secs_f64();
        if st.wave.end == self.ids.end {
            let gb = st.bytes as f64 / 1e9;
            tracing::info!(
                "glm5_next layer {layer}: {} routed experts prepared on the host in {:.2}s \
                 ({gb:.2} GB O_DIRECT, {:.2} GB/s, waves of {WAVE})",
                self.ids.len(),
                st.secs,
                gb / st.secs.max(1e-9),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "expert_prepare_tests.rs"]
mod tests;
