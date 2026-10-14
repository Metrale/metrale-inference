// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Slab allocation: many buffers carved from one `cuMemAlloc_v2`.
//!
//! The driver rounds every allocation to its own granule. Measured on an H100 SXM
//! (2,000 allocations per size): 4 B costs 1,049 B, 819,200 B (one NVFP4 expert projection
//! of a 512-expert, hidden-2560 MoE) costs 1 MiB, 5 MiB costs 6 MiB. A loader that gives each
//! of a checkpoint's ~220k tensors its own allocation pays ~9 GiB above the tensors' bytes on
//! such a model. One slab per shard pays at most `SLAB_ALIGN - 1` bytes per tensor.
//!
//! [`SlabTable`] is the pure bookkeeping: which pointers are slab members, and when the last
//! live member of a slab is freed, its base. The backend frees the base then, and only then:
//! a member freed earlier leaves its bytes resident until its slab's last member goes.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - every member pointer maps to exactly one live slab base;
//! - a slab's base is returned by [`SlabTable::release`] exactly once, with its last member.

use std::collections::HashMap;

use anyhow::{Result, bail};

use super::{MetraleCudaBackend, cuMemAlloc_v2, cuMemFree_v2};
use crate::gpu::DevicePtr;

/// 2026-10-13: Member alignment inside a slab: the 256 B `cudaMalloc` guarantees, which is
/// what every kernel reading a weight may assume of its pointer.
pub const SLAB_ALIGN: usize = 256;

/// 2026-10-13: Offsets of `sizes` packed at [`SLAB_ALIGN`], and the slab's total bytes. A
/// zero-size entry still gets its own aligned slot, so every member pointer is distinct.
pub fn slab_layout(sizes: &[usize]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut cursor = 0usize;
    for &bytes in sizes {
        offsets.push(cursor);
        cursor += bytes.max(1).div_ceil(SLAB_ALIGN) * SLAB_ALIGN;
    }
    (offsets, cursor)
}

/// 2026-10-13: What freeing one pointer means for the slabs.
#[derive(Debug, PartialEq, Eq)]
pub enum SlabRelease {
    /// 2026-10-13: Not a slab member: the caller frees the pointer itself.
    NotMember,
    /// 2026-10-13: A member, and its slab still has live members: nothing to free.
    StillLive,
    /// 2026-10-13: The slab's last live member: the caller frees this base.
    LastMember(u64),
}

/// 2026-10-13: Live slabs: member pointer to base, base to live-member count.
/// 2026-10-14: And base to the slab's bytes, the allocation the driver placed.
#[derive(Default)]
pub struct SlabTable {
    member_base: HashMap<u64, u64>,
    live: HashMap<u64, usize>,
    bytes: HashMap<u64, usize>,
}

impl SlabTable {
    /// 2026-10-13: Register a slab at `base` whose members start at `base + offsets[i]`.
    /// Returns the member pointers in `offsets` order.
    /// 2026-10-14: `total` is the slab's allocated bytes (`slab_layout`'s second value).
    pub fn insert(&mut self, base: u64, offsets: &[usize], total: usize) -> Vec<u64> {
        let members: Vec<u64> = offsets.iter().map(|&o| base + o as u64).collect();
        for &m in &members {
            self.member_base.insert(m, base);
        }
        self.live.insert(base, members.len());
        self.bytes.insert(base, total);
        members
    }

    /// 2026-10-13: Take `ptr` off the table; see [`SlabRelease`].
    pub fn release(&mut self, ptr: u64) -> SlabRelease {
        let Some(base) = self.member_base.remove(&ptr) else {
            return SlabRelease::NotMember;
        };
        let count = self
            .live
            .get_mut(&base)
            .expect("slab member maps to a registered base");
        *count -= 1;
        if *count == 0 {
            self.live.remove(&base);
            self.bytes.remove(&base);
            SlabRelease::LastMember(base)
        } else {
            SlabRelease::StillLive
        }
    }

    /// 2026-10-13: Number of slabs with a live member.
    pub fn live_slabs(&self) -> usize {
        self.live.len()
    }

    /// 2026-10-14: `allocs` (address, bytes) as the driver placed them: every slab member
    /// dropped, and each live slab as one allocation of its full bytes. A member below the 2 MiB
    /// chunk read on its own looks like a small allocation the driver packs into a chunk, and a
    /// chunk it shares with a larger member read as slack: 340 MiB of 166,823 members on
    /// Qwen3.8-Flash-Next with its BF16 tensors in the slabs, against 0 for the slabs whole.
    pub fn driver_allocations(
        &self,
        allocs: impl Iterator<Item = (u64, usize)>,
    ) -> Vec<(u64, usize)> {
        allocs
            .filter(|(addr, _)| !self.member_base.contains_key(addr))
            .chain(self.bytes.iter().map(|(&base, &total)| (base, total)))
            .collect()
    }
}

impl MetraleCudaBackend {
    /// 2026-10-13: One `cuMemAlloc_v2` for all of `sizes`, each member on the ledger with its
    /// own bytes and `site`. An empty `sizes` allocates nothing.
    pub(crate) fn alloc_slab_impl(
        &self,
        sizes: &[usize],
        site: &'static std::panic::Location<'static>,
    ) -> Result<Vec<DevicePtr>> {
        if sizes.is_empty() {
            return Ok(Vec::new());
        }
        let (offsets, total) = slab_layout(sizes);
        let mut base: u64 = 0;
        let status = unsafe { cuMemAlloc_v2(&mut base, total) };
        if status != 0 {
            bail!(
                "cuMemAlloc_v2 failed for a slab of {} buffers ({total} bytes): status {status}",
                sizes.len()
            );
        }
        let members = self.slabs.lock().insert(base, &offsets, total);
        for (&m, &bytes) in members.iter().zip(sizes) {
            self.record_alloc(DevicePtr(m), bytes, site);
        }
        Ok(members.into_iter().map(DevicePtr).collect())
    }

    /// 2026-10-13: Free `raw` as `free` and the sweep do: a slab member frees its slab's base
    /// with the last member, anything else is freed directly. Returns the driver status.
    pub(crate) fn free_raw_or_member(&self, raw: u64) -> i32 {
        let target = match self.slabs.lock().release(raw) {
            SlabRelease::NotMember => raw,
            SlabRelease::StillLive => return 0,
            SlabRelease::LastMember(base) => base,
        };
        unsafe { cuMemFree_v2(target) }
    }
}

#[cfg(test)]
#[path = "slab_tests.rs"]
mod tests;
