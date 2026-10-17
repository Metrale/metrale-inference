// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: One device allocation per MoE layer for the routed experts of the `tp` layout, in
//! place of two per projection.
//!
//! **WHY.** Under `--moe-expert-layout tp` every rank binds all 288 experts of each of the 43 MoE
//! layers as six buffers (codes and block scales of gate, up and down): 74,304 `cuMemAlloc`s of
//! 1408 KiB and 176 KiB on ranks 0 and 1 of GLM-5.3 at TP=3 (slice width 704). GB10 places an
//! allocation under 2 MiB inside a 2 MiB chunk that stays resident whole
//! (`metrale_gpu_runtime::cuda_backend::SMALL_ALLOC_CHUNK`), so each 1408 KiB code buffer took a
//! chunk of its own and one 176 KiB scale buffer filled part of the rest: 464 KiB idle per chunk,
//! 16.53 GiB per rank (21.33 GiB on rank 2) that the KV budget charged as chunk slack and nothing
//! read (K65536/K262144/K524288 serve logs, 2026-10-14). A slab of one layer (~1.3 GiB) is one
//! large allocation, which the driver does not pack: no slack.
//!
//! Placement only. Every buffer holds the bytes its own allocation held, at an offset aligned to
//! [`SLAB_ALIGN`]; kernels reach the experts through per-expert pointer tables
//! (`glm5next_mlp::build::build_expert_ptr_table`), never through the allocation.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - A slab is adopted by the store's derived-weight ledger once, whole; no buffer inside it is
//!   adopted or freed on its own.
//! - Every expert of a slab has the buffer sizes of the first one placed; another is an error.
//! - Expert `id` is placed at most once, in slot `id - ids.start`.

use std::cell::RefCell;
use std::ops::Range;

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

/// 2026-10-14: The alignment of every buffer's offset in a slab. `cuMemAlloc` guarantees 256 B
/// and GB10 placed the separate allocations this replaces at 512 B or coarser, so no kernel sees
/// a less-aligned pointer than before. GLM-5.3's buffer sizes are multiples of 16 KiB, so it adds
/// no padding there.
pub(super) const SLAB_ALIGN: usize = 4096;

/// 2026-10-14: The ledger label of a `tp`-layout slab: every routed-expert buffer of a layer.
pub(super) const TP_SLAB_LABEL: &str = "glm5_next routed experts, one slab per layer (tp layout)";
/// 2026-10-17: The ledger label of a whole-layout slab: the block scales decoded from csf planes
/// (`expert_scales`), the only routed-expert bytes that layout uploads rather than binds.
pub(super) const CSF_SCALE_SLAB_LABEL: &str =
    "glm5_next decoded csf block scales, one slab per layer (whole layout)";

/// 2026-10-14: Where each of one expert's buffers sits in its slot, and the slot's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SlotLayout {
    /// 2026-10-14: Bytes of each buffer, in placement order.
    pub sizes: Vec<usize>,
    /// 2026-10-14: Offset of each buffer from the slot's start, each a multiple of the alignment.
    pub offsets: Vec<usize>,
    /// 2026-10-14: Bytes from one slot's start to the next's, a multiple of the alignment.
    pub stride: usize,
}

/// 2026-10-14: The slot of buffers of `sizes`, each started at the next multiple of `align`.
/// Pure. An error for a zero `align` or a slot past `usize`.
pub(super) fn slot_layout(sizes: &[usize], align: usize) -> Result<SlotLayout> {
    if align == 0 {
        bail!("expert slab: alignment must be positive");
    }
    let up = |n: usize| -> Result<usize> {
        n.div_ceil(align)
            .checked_mul(align)
            .ok_or_else(|| anyhow::anyhow!("expert slab: {n} B does not align to {align}"))
    };
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut end = 0usize;
    for &s in sizes {
        offsets.push(end);
        let next = end.checked_add(s);
        end = up(next.ok_or_else(|| anyhow::anyhow!("expert slab: slot size overflows"))?)?;
    }
    Ok(SlotLayout {
        sizes: sizes.to_vec(),
        offsets,
        stride: end,
    })
}

/// 2026-10-14: The slab of one layer's experts, allocated when the first is placed.
struct Placed {
    base: DevicePtr,
    layout: SlotLayout,
    taken: Vec<bool>,
}

/// 2026-10-14: The slab of the experts `ids` of one MoE layer: one slot per id, in id order.
pub(super) struct ExpertSlab {
    ids: Range<usize>,
    label: &'static str,
    placed: RefCell<Option<Placed>>,
}

impl ExpertSlab {
    /// 2026-10-14: An empty slab for the experts `ids`; nothing is allocated yet. 2026-10-17: Its
    /// allocation is adopted under `label`.
    pub(super) fn new(ids: Range<usize>, label: &'static str) -> Self {
        Self {
            ids,
            label,
            placed: RefCell::new(None),
        }
    }

    /// 2026-10-14: Upload `blobs` into expert `id`'s slot and return their device pointers, in
    /// order. The first call allocates the slab (`ids.len()` slots of this expert's layout) and
    /// gives it to the store's derived-weight ledger. An error for an id outside `ids` or placed
    /// before, or for buffer sizes other than the first expert's.
    pub(super) fn place<const N: usize>(
        &self,
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        id: usize,
        blobs: [&[u8]; N],
    ) -> Result<[DevicePtr; N]> {
        if !self.ids.contains(&id) {
            bail!(
                "expert slab: expert {id} is outside this slab's {:?}",
                self.ids
            );
        }
        let slot = id - self.ids.start;
        let sizes: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
        let mut placed = self.placed.borrow_mut();
        if placed.is_none() {
            let layout = slot_layout(&sizes, SLAB_ALIGN)?;
            let total = layout
                .stride
                .checked_mul(self.ids.len())
                .ok_or_else(|| anyhow::anyhow!("expert slab: {} slots overflow", self.ids.len()))?;
            let base = gpu.alloc(total.max(1))?;
            store.derived().adopt(self.label, base, total);
            *placed = Some(Placed {
                base,
                layout,
                taken: vec![false; self.ids.len()],
            });
        }
        let Some(p) = placed.as_mut() else {
            bail!("expert slab: no slab after allocating one");
        };
        if sizes != p.layout.sizes {
            bail!(
                "expert slab: expert {id} has buffers of {sizes:?} B; this layer's first expert \
                 had {:?}",
                p.layout.sizes
            );
        }
        if std::mem::replace(&mut p.taken[slot], true) {
            bail!("expert slab: expert {id} was placed twice");
        }
        let start = p.base.0 + (slot * p.layout.stride) as u64;
        let mut out = [DevicePtr::NULL; N];
        for (i, b) in blobs.iter().enumerate() {
            out[i] = DevicePtr(start + p.layout.offsets[i] as u64);
            gpu.copy_h2d(b, out[i])?;
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "expert_slab_tests.rs"]
mod tests;
