// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Bodies the `GpuBackend` impl in `gpu_impl.rs` calls: the red-zone
//! poisoning of a new `alloc`, and the pitched D2D copy of `copy_d2d_2d_async`.
//! 2026-10-01: Also `alloc_arena` and `free_arena`, moved here unchanged from `gpu_impl.rs`.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - Each `unsafe` block is one call: `cuMemsetD8Async`, `cuMemAlloc_v2`, `cuMemGetInfo_v2`
//!   or `cuMemFree_v2`, declared in `cuda_backend.rs`, or `cudaMemcpy2DAsync`. The caller
//!   provides a current context and device pointers into live allocations with byte counts
//!   that fit them.

use std::ffi::c_void;

use anyhow::{Result, bail};

use super::super::MetraleCudaBackend;
use crate::gpu::DevicePtr;

impl MetraleCudaBackend {
    /// 2026-09-26: Fill the `pad`-byte guard band after a new `bytes`-byte
    /// allocation at `dptr`, record it, and log it. An error when the fill fails.
    pub(super) fn alloc_poison_redzone(
        &self,
        dptr: u64,
        bytes: usize,
        pad: usize,
        seq: usize,
    ) -> Result<()> {
        let st = unsafe {
            super::super::cuMemsetD8Async(dptr + bytes as u64, super::super::redzone_fill(), pad, 0)
        };
        if st != 0 {
            bail!("METRALE_REDZONE: poisoning the guard band failed: status {st}");
        }
        self.record_redzone(dptr, bytes, pad, seq);
        // 2026-09-25: One line per guarded allocation, with its creation
        // index and size; `METRALE_REDZONE_TRACE_IDX` adds a backtrace for
        // one index.
        tracing::info!(target: "metrale_gpu_runtime::cuda_backend::gpu_impl", "redzone: alloc#{seq} bytes={bytes} ptr={dptr:#x}");
        if super::super::redzone_trace_idx() == Some(seq) {
            tracing::error!(target: "metrale_gpu_runtime::cuda_backend::gpu_impl", "redzone: alloc#{seq} bytes={bytes} backtrace:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
        Ok(())
    }
}

/// 2026-09-26: `copy_d2d_2d_async`: `height` rows of `width_bytes` from `src`
/// (pitch `src_pitch`) to `dst` (pitch `dst_pitch`), enqueued on `stream`.
pub(super) fn memcpy_2d_async(
    src: DevicePtr,
    src_pitch: usize,
    dst: DevicePtr,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: One pitched copy on the caller's stream through the
    // runtime API; `kind` 3 is cudaMemcpyDeviceToDevice.
    unsafe extern "C" {
        fn cudaMemcpy2DAsync(
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: i32,
            stream: u64,
        ) -> i32;
    }
    let status = unsafe {
        cudaMemcpy2DAsync(
            dst.0 as *mut c_void,
            dst_pitch,
            src.0 as *const c_void,
            src_pitch,
            width_bytes,
            height,
            3,
            stream,
        )
    };
    if status != 0 {
        bail!("cudaMemcpy2DAsync failed: status {status}");
    }
    Ok(())
}

/// 2026-10-01: `GpuBackend::alloc_arena`: `bytes` of device memory off the allocation ledger.
pub(super) fn alloc_arena(bytes: usize) -> Result<DevicePtr> {
    use super::super::{cuMemAlloc_v2, cuMemGetInfo_v2};
    let mut dptr: u64 = 0;
    let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
    if status != 0 {
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        bail!(
            "cuMemAlloc_v2 (arena) failed: status {status}, requested {bytes} bytes \
             (device reports {:.1} GB free / {:.1} GB total)",
            free as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    tracing::info!(
        "arena: {:.2} GiB of device memory at {dptr:#x}, off the allocation ledger",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(DevicePtr(dptr))
}

/// 2026-10-01: `GpuBackend::free_arena`.
pub(super) fn free_arena(ptr: DevicePtr) -> Result<()> {
    use super::super::cuMemFree_v2;
    if ptr.is_null() {
        return Ok(());
    }
    let status = unsafe { cuMemFree_v2(ptr.0) };
    if status != 0 {
        // 2026-09-25: Logged, not returned, for any nonzero status.
        tracing::warn!("cuMemFree_v2 (arena) returned status {status}");
    }
    Ok(())
}
