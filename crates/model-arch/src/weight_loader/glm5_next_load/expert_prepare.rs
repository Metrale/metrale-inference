// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Prepare one MoE layer's tp-layout routed experts on the host, all at once and in
//! parallel, from `O_DIRECT` shard reads (`METRALE_GLM_EXPERT_FAST_LOAD`, default on).
//!
//! **WHY.** Under `--moe-expert-layout tp` every routed-expert tensor is deferred, and the
//! binder read each one with its own buffered `open + seek + read_exact` into a fresh `Vec` on
//! the loading thread, then sliced it (and quantized the MTP layer's 16-bit experts) before the
//! upload. GLM-5.3 TP=3 boots spent 213-243 s in the 45-layer build at ~0.3 GB/s of expert reads,
//! against 1.85 GB/s for the fast weight loader (DEV-LOOP-SPEED.md item 3). Here a worker pool
//! does the reads, slices and quantizations of every expert of the layer before the first
//! upload; the upload stays on the loading thread, in expert order, into the same slab.
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
    /// 2026-10-24: `METRALE_GLM_EXPERT_FAST_LOAD`: prepare each layer with [`Prepared::new`].
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

/// 2026-10-24: One layer's routed experts `ids`, prepared on the host for the tp layout.
pub(super) struct Prepared {
    ids: Range<usize>,
    slots: Mutex<Vec<Option<[HostProj; 3]>>>,
}

impl Prepared {
    /// 2026-10-24: Read, slice and (for 16-bit experts) quantize every expert of `ids` of
    /// `layer` under slice `s` on the rayon pool, from [`Direct`] reads. The first expert's
    /// error, by id, is the error.
    pub(super) fn new(
        store: &WeightStore,
        layer: usize,
        ids: Range<usize>,
        s: &ExpertSlice,
    ) -> Result<Self> {
        let t = std::time::Instant::now();
        let src = Direct {
            store,
            bytes: AtomicU64::new(0),
        };
        let slots = ids
            .clone()
            .into_par_iter()
            .map(|id| host_expert(store, &src, layer, id, s).map(Some))
            .collect::<Result<Vec<_>>>()?;
        let (gb, secs) = (
            src.bytes.load(Ordering::Relaxed) as f64 / 1e9,
            t.elapsed().as_secs_f64(),
        );
        tracing::info!(
            "glm5_next layer {layer}: {} routed experts prepared on the host in {secs:.2}s \
             ({gb:.2} GB O_DIRECT, {:.2} GB/s)",
            ids.len(),
            gb / secs.max(1e-9),
        );
        Ok(Self {
            ids,
            slots: Mutex::new(slots),
        })
    }

    /// 2026-10-24: Expert `id`'s projections (gate, up, down), handed over once.
    pub(super) fn take(&self, id: usize) -> Result<[HostProj; 3]> {
        if !self.ids.contains(&id) {
            bail!(
                "prepared experts: {id} is outside this layer's {:?}",
                self.ids
            );
        }
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| anyhow::anyhow!("prepared experts: lock poisoned"))?;
        slots[id - self.ids.start]
            .take()
            .with_context(|| format!("prepared experts: expert {id} was taken twice"))
    }
}

#[cfg(test)]
#[path = "expert_prepare_tests.rs"]
mod tests;
