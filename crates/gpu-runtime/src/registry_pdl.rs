// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Programmatic dependent launch (PDL) on the driver API: a kernel launched through
//! `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` may begin while
//! the previous kernel on its stream is still running (once every block of that kernel has run
//! `griddepcontrol.launch_dependents` or exited), and its `griddepcontrol.wait` blocks until that
//! kernel has completed and its memory is visible. Under stream capture the stream edge becomes a
//! programmatic graph edge, so a replayed graph keeps the overlap.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Only an entry listed by `ptx_pdl::waiting_entries` is launched here
//!   ([`MetraleRegistry::entry_waits`] is the test the backend applies).
//! - The dynamic shared memory opt-in above 48 KiB is the same for both launch paths
//!   ([`ensure_dynamic_smem`]).

use std::ffi::c_void;

use cudarc::driver::LaunchConfig;
use metrale_core::error::{MetraleError, Result};

use crate::registry::{MetraleRegistry, RawCudaFunc, cuda_error_text};

// 2026-10-21: `cuLaunchKernelEx` goes through cudarc's dynamically loaded driver table, not a
// link-time `extern` block: the CUDA 13 link stub (`lib64/stubs/libcuda.so`) on these boxes does
// not export it (as `graph_hoist_cu.rs` notes for the graph queries).
#[cfg(not(metrale_scale))]
use cudarc::driver::sys;

/// 2026-10-21: Opt `raw_func` in to `bytes` of dynamic shared memory when that is more than the
/// default 48 KiB (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`).
///
/// # Safety
/// `raw_func` must be a valid CUfunction of a loaded module.
pub(crate) unsafe fn ensure_dynamic_smem(raw_func: RawCudaFunc, bytes: u32) -> Result<()> {
    if bytes <= 48 * 1024 {
        return Ok(());
    }
    const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
    let status = unsafe {
        crate::registry::cuFuncSetAttribute(
            raw_func.0,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    };
    if status != 0 {
        return Err(MetraleError::KernelLaunch(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={bytes}) failed: {}",
            cuda_error_text(status)
        )));
    }
    Ok(())
}

impl MetraleRegistry {
    /// 2026-10-21: Whether entry `func_name` of PTX module `module_name` waits on its grid
    /// dependencies (`ptx_pdl::waiting_entries`); false for a binary module or an unknown name.
    pub fn entry_waits(&self, module_name: &str, func_name: &str) -> bool {
        self.waiting
            .get(module_name)
            .is_some_and(|w| w.contains(func_name))
    }

    /// 2026-10-21: [`MetraleRegistry::launch_on_stream`] with programmatic stream serialization
    /// allowed (`cuLaunchKernelEx`).
    ///
    /// # Safety
    /// As `launch_on_stream`; and `raw_func` must be an entry for which
    /// [`Self::entry_waits`] holds, or the kernel may read its predecessor's output early.
    #[cfg(not(metrale_scale))]
    pub unsafe fn launch_on_stream_programmatic(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe { ensure_dynamic_smem(raw_func, cfg.shared_mem_bytes) }?;
        // 2026-10-21: SAFETY: an all-zero `CUlaunchAttributeValue` is a valid union value (every
        // member is plain data); the one member written is the one the id selects.
        let mut value: sys::CUlaunchAttributeValue = unsafe { std::mem::zeroed() };
        value.programmaticStreamSerializationAllowed = 1;
        let mut attr = sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value,
        };
        let config = sys::CUlaunchConfig {
            gridDimX: cfg.grid_dim.0,
            gridDimY: cfg.grid_dim.1,
            gridDimZ: cfg.grid_dim.2,
            blockDimX: cfg.block_dim.0,
            blockDimY: cfg.block_dim.1,
            blockDimZ: cfg.block_dim.2,
            sharedMemBytes: cfg.shared_mem_bytes,
            hStream: stream_ptr as sys::CUstream,
            attrs: &mut attr,
            numAttrs: 1,
        };
        let status = unsafe {
            sys::cuLaunchKernelEx(
                &config,
                raw_func.0 as sys::CUfunction,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        } as i32;
        if status != 0 {
            return Err(MetraleError::KernelLaunch(format!(
                "cuLaunchKernelEx (programmatic) failed: {} (grid={:?}, block={:?}, shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }
}
