// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The driver calls the probe needs and the backend does not expose: timing events
//! (the backend's events disable timing). Streams come from the backend, as the serve makes
//! them.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: every call runs on the thread whose context the backend made current.

use anyhow::{Result, ensure};

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
}

fn ok(rc: i32, what: &str) -> Result<()> {
    ensure!(rc == 0, "{what} failed with status {rc}");
    Ok(())
}

/// 2026-10-15: A timing event.
pub fn event() -> Result<u64> {
    let mut e = 0u64;
    // 2026-10-15: SAFETY: a valid out-pointer; flags 0 keeps timing on.
    ok(unsafe { cuEventCreate(&mut e, 0) }, "cuEventCreate")?;
    Ok(e)
}

pub fn record(event: u64, stream: u64) -> Result<()> {
    // 2026-10-15: SAFETY: both handles were created by this module.
    ok(unsafe { cuEventRecord(event, stream) }, "cuEventRecord")
}

pub fn sync(event: u64) -> Result<()> {
    // 2026-10-15: SAFETY: the handle was created by this module.
    ok(unsafe { cuEventSynchronize(event) }, "cuEventSynchronize")
}

/// 2026-10-15: Microseconds from `start` to `end`; both must have completed.
pub fn elapsed_us(start: u64, end: u64) -> Result<f64> {
    let mut ms = 0f32;
    // 2026-10-15: SAFETY: a valid out-pointer and two completed timing events.
    ok(
        unsafe { cuEventElapsedTime(&mut ms, start, end) },
        "cuEventElapsedTime",
    )?;
    Ok(f64::from(ms) * 1e3)
}
