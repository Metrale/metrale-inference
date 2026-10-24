// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The `METRALE_D2H_TRACE` counter of the D2H copies, split from `gpu_impl.rs`
//! unchanged.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: none beyond the types.

/// 2026-09-25: D2H copy counter for `METRALE_D2H_TRACE=<N>`: a backtrace at
/// call N and the running count at every 10,000th call. Advanced by
/// `copy_d2h`, `copy_d2h_on_stream` and `copy_d2h_async` only while the
/// variable is set.
static D2H_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 2026-10-17: `target` is the backend's `DeviceSettings::d2h_trace`. `None` (unset) returns
/// before the counter; 0 counts without a backtrace.
pub(super) fn d2h_trace_tick(target: Option<u64>) {
    use std::sync::atomic::Ordering;
    let Some(target) = target else {
        return;
    };
    let n = D2H_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if target != 0 && n == target {
        tracing::warn!(
            "METRALE_D2H_TRACE: call #{n} backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
    if n.is_multiple_of(10_000) {
        tracing::warn!("METRALE_D2H_TRACE: {n} D2H copies so far (each forces a stream sync)");
    }
}
