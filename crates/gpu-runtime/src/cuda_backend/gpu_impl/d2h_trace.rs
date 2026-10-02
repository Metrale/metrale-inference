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

pub(super) fn d2h_trace_tick() {
    use std::sync::atomic::Ordering;
    // 2026-09-25: The variable is read once. Unset, this returns before the
    // counter. A value that does not parse as `u64` reads as 0: no backtrace,
    // only the every-10,000th count.
    static TARGET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let Some(target) = *TARGET.get_or_init(|| {
        std::env::var("METRALE_D2H_TRACE")
            .ok()
            .map(|v| v.parse().unwrap_or(0))
    }) else {
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
