// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Bodies the `GpuBackend` impl in `gpu_impl.rs` calls: the red-zone
//! poisoning of a new `alloc`, and the pitched D2D copy of `copy_d2d_2d_async`.
//! 2026-10-17: That copy is now `memcpy_2d_async`, on the driver API (`cuMemcpy2DAsync_v2`).
//! 2026-10-01: Also `alloc_arena` and `free_arena`, moved here unchanged from `gpu_impl.rs`.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - Each `unsafe` block is one call: `cuMemsetD8Async`, `cuMemAlloc_v2`, `cuMemGetInfo_v2`,
//!   `cuMemFree_v2`, declared in `cuda_backend.rs`, or `cuMemcpy2DAsync_v2`, through cudarc's
//!   dynamically loaded driver table (the CUDA 13 link stub does not export it). The caller
//!   provides a current context and device pointers into live allocations with byte counts
//!   that fit them.

use anyhow::{Result, bail};
use metrale_error::{ErrorCode, error};

use super::super::MetraleCudaBackend;
use crate::gpu::{DevicePtr, Memcpy2d};

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
            super::super::cuMemsetD8Async(dptr + bytes as u64, self.device.redzone.fill, pad, 0)
        };
        if st != 0 {
            bail!("METRALE_REDZONE: poisoning the guard band failed: status {st}");
        }
        self.record_redzone(dptr, bytes, pad, seq);
        // 2026-09-25: One line per guarded allocation, with its creation
        // index and size; `METRALE_REDZONE_TRACE_IDX` adds a backtrace for
        // one index.
        tracing::info!(target: "metrale_gpu_runtime::cuda_backend::gpu_impl", "redzone: alloc#{seq} bytes={bytes} ptr={dptr:#x}");
        if self.device.redzone.trace_idx == Some(seq) {
            tracing::error!(target: "metrale_gpu_runtime::cuda_backend::gpu_impl", "redzone: alloc#{seq} bytes={bytes} backtrace:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
        Ok(())
    }
}

/// 2026-10-17: `GpuBackend::memcpy_2d_async`: one `cuMemcpy2DAsync_v2` of `copy`, both sides
/// device memory, enqueued on `stream`.
#[cfg(not(metrale_scale))]
pub(super) fn memcpy_2d_async(copy: Memcpy2d, stream: u64) -> metrale_error::Result<()> {
    use cudarc::driver::sys::{self, CUDA_MEMCPY2D, CUmemorytype};
    let desc = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: copy.src.0,
        srcArray: std::ptr::null_mut(),
        srcPitch: copy.src_pitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: copy.dst.0,
        dstArray: std::ptr::null_mut(),
        dstPitch: copy.dst_pitch,
        WidthInBytes: copy.width_bytes,
        Height: copy.height,
    };
    let status = unsafe { sys::cuMemcpy2DAsync_v2(&desc, stream as sys::CUstream) };
    if status != sys::CUresult::CUDA_SUCCESS {
        return Err(error!(
            ErrorCode::DeviceDriver,
            "cuMemcpy2DAsync_v2", status as i32
        ));
    }
    Ok(())
}

/// 2026-10-17: `cuMemcpy2DAsync_v2` is not verified on SCALE (its loader stubs missing symbols);
/// `copy_d2d_2d_async` keeps the trait's per-row loop on SCALE.
#[cfg(metrale_scale)]
pub(super) fn memcpy_2d_async(_copy: Memcpy2d, _stream: u64) -> metrale_error::Result<()> {
    Err(error!(
        ErrorCode::NotImplemented,
        "memcpy_2d_async", "SCALE"
    ))
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
