// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The mapped argmax blob of the batched verify. Split from `verify_e.rs`.
//!
//! Owner: model-engine (MTP verify).
//! Invariants:
//! - The blob is allocated once and never freed, so a captured graph keeps a valid address.

/// 2026-09-25: A 65_536-byte page-locked host blob and its device alias
/// (`host_ptr_to_device`) for the mapped argmax. After the first success the
/// same pair is returned for the rest of the process, so a captured graph
/// keeps a valid address. Returns `None` when `off` (the settings'
/// `METRALE_NO_MAPPED_ARGMAX`), or when the allocation or the mapping fails;
/// callers then use scratch and a copy.
fn mapped_argmax_host_dev(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    off: bool,
) -> Option<(*mut u8, metrale_gpu_runtime::gpu::DevicePtr)> {
    use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
    static HOST: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
    static DEV: AtomicU64 = AtomicU64::new(0);
    if off {
        return None;
    }
    let mut h = HOST.load(Ordering::Acquire);
    if h.is_null() {
        // 2026-09-25: Only the scheduler thread calls this. A failure stores
        // nothing, so the next call tries again.
        h = gpu.alloc_host_pinned(65_536).ok()?;
        let d = gpu.host_ptr_to_device(h).ok()?;
        DEV.store(d.0, Ordering::Release);
        HOST.store(h, Ordering::Release);
    }
    let d = DEV.load(Ordering::Acquire);
    if d == 0 {
        return None;
    }
    Some((h, metrale_gpu_runtime::gpu::DevicePtr(d)))
}

impl super::super::super::types::TransformerModel {
    /// 2026-10-17: [`mapped_argmax_host_dev`] on this model's GPU, off when the settings say so.
    pub(in super::super) fn mapped_argmax(
        &self,
    ) -> Option<(*mut u8, metrale_gpu_runtime::gpu::DevicePtr)> {
        mapped_argmax_host_dev(self.gpu.as_ref(), self.settings.runtime.no_mapped_argmax)
    }
}
