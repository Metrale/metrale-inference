// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Which thread of this process the CUDA driver runs host functions on (NCCL's
//! proxy-upload callbacks, `cuLaunchHostFunc` nodes of captured graphs): one host function that
//! records `gettid`, run to completion. The wake keeper pins that thread
//! (`crate::host_wake`).
//!
//! Owner: gpu-runtime.
//! Invariants: the stream is synchronised before the recorded id is read; one probe at a time.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};

use anyhow::{Result, ensure};
use cudarc::driver::sys;

static TID: AtomicI32 = AtomicI32::new(0);
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

unsafe extern "C" fn record(_: *mut std::ffi::c_void) {
    TID.store(crate::host_wake::os::gettid(), Ordering::Release);
}

/// 2026-10-21: The kernel thread id that ran a host function queued on `stream`.
///
/// # Errors
/// The launch or the synchronisation fails, or the function did not run.
pub fn host_task_tid(stream: u64) -> Result<i32> {
    let _one = ONE_AT_A_TIME
        .lock()
        .map_err(|_| anyhow::anyhow!("host task probe lock poisoned"))?;
    TID.store(0, Ordering::Release);
    // SAFETY: `record` touches only a static; it takes no user data.
    let rc = unsafe {
        sys::cuLaunchHostFunc(stream as sys::CUstream, Some(record), std::ptr::null_mut())
    };
    ensure!(rc == sys::CUresult::CUDA_SUCCESS, "cuLaunchHostFunc: {rc:?}");
    // SAFETY: plain driver call on a live stream.
    let rc = unsafe { sys::cuStreamSynchronize(stream as sys::CUstream) };
    ensure!(rc == sys::CUresult::CUDA_SUCCESS, "cuStreamSynchronize: {rc:?}");
    let tid = TID.load(Ordering::Acquire);
    ensure!(tid != 0, "the host function did not run");
    Ok(tid)
}
