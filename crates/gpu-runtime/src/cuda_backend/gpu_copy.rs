// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The blocking copies (`copy_h2d`, `copy_d2h`,
//! `copy_d2h_on_stream`, `copy_d2d`) as inherent `_impl` methods that the
//! `GpuBackend` impl in `gpu_impl.rs` delegates to. Each enqueues one async
//! copy and then waits for its stream.
//!
//! Nothing here checks a bound. The host end is a slice, so its length is
//! the copy length; the device end is unchecked, and the caller must size it.
//! `live_allocs` is keyed by base pointer, while callers pass interior
//! pointers (`DevicePtr::offset`), so a lookup by pointer would not find the
//! allocation.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - On `Ok`, the copy has completed: each method synchronises the stream
//!   it enqueued on before returning `Ok`.

use std::ffi::c_void;

use crate::registry::cuda_error_text;
use anyhow::{Result, bail};

use super::{
    MetraleCudaBackend, cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
    cuStreamQuery, cuStreamSynchronize,
};
use crate::gpu::DevicePtr;

impl MetraleCudaBackend {
    /// 2026-10-10: Refuse a host-blocking `what` on `stream` while it captures a CUDA graph.
    /// The driver would invalidate the capture; on a multi-rank serve the collectives already
    /// captured on the other ranks then never pair up, and the serve hangs instead of failing.
    /// Refusing first names the call and leaves the capture to the caller's error path; the
    /// message carries `STREAM_CAPTURE`, which the decode step's capture fallback recognises as
    /// a capture fault, as it does the driver's own status.
    pub(crate) fn refuse_host_sync_in_capture(&self, stream: u64, what: &str) -> Result<()> {
        #[cfg(not(metrale_scale))]
        {
            // 2026-10-10: CU_STREAM_CAPTURE_STATUS_ACTIVE = 1. A failed query is not a refusal.
            let mut status: u32 = 0;
            let rc = unsafe { super::cuStreamIsCapturing(stream, &mut status) };
            if rc == 0 && status == 1 {
                bail!(
                    "{what}: a host synchronisation on a stream that is capturing a CUDA graph \
                     (STREAM_CAPTURE: the captured path must not read device results back)"
                );
            }
        }
        #[cfg(metrale_scale)]
        let _ = (stream, what);
        Ok(())
    }

    pub(crate) fn copy_h2d_impl(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.refuse_host_sync_in_capture(self.default_stream, "copy_h2d")?;
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.0,
                src.as_ptr() as *const c_void,
                src.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        // 2026-09-25: Wait, so the caller may drop `src` on return.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after H2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2h_impl(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.refuse_host_sync_in_capture(self.default_stream, "copy_d2h")?;
        metrale_telemetry::global().blocking_d2h();
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut c_void,
                src.0,
                dst.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2h_on_stream_impl(
        &self,
        src: DevicePtr,
        dst: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        self.refuse_host_sync_in_capture(stream, "copy_d2h_on_stream")?;
        metrale_telemetry::global().blocking_d2h();
        // 2026-09-25: On the caller's `stream`, so the copy is ordered after
        // the work already queued there.
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (on_stream) failed: status {status}");
        }
        // 2026-09-25: METRALE_D2H_SPIN_SYNC=1 (read once) busy-polls
        // `cuStreamQuery` until it stops returning CUDA_ERROR_NOT_READY,
        // instead of blocking in `cuStreamSynchronize`.
        let spin = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("METRALE_D2H_SPIN_SYNC").as_deref() == Ok("1"))
        };
        let sync = if spin {
            const CUDA_ERROR_NOT_READY: i32 = 600;
            loop {
                let q = unsafe { cuStreamQuery(stream) };
                if q != CUDA_ERROR_NOT_READY {
                    break q;
                }
                std::hint::spin_loop();
            }
        } else {
            unsafe { cuStreamSynchronize(stream) }
        };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H on_stream failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2d_impl(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.default_stream) };
        if status != 0 {
            // 2026-09-25: Log a backtrace: under stream capture the status can
            // be 901 (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED), which reports an
            // earlier error in the capture, so the backtrace of this copy is
            // what locates the captured code.
            tracing::error!(
                "sync copy_d2d failed (status {status}) at:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
            bail!("cuMemcpyDtoDAsync_v2 (sync copy_d2d) failed: status {status}");
        }
        // 2026-09-25: Wait, so work on other streams sees the copied bytes.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }
}
