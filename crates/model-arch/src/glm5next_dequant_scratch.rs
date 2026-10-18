// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The one BF16 scratch the GLM-5.3 dequantize + cuBLASLt prefill levers share:
//! `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS` (`glm5next_w4a16_wide`, NVFP4 -> BF16) and
//! `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS` (`glm5next_fp8_wide`, E4M3 x row scale -> BF16). Each
//! lever reserves the `[n, k]` of every weight it will serve as that weight registers at load;
//! the scratch grows to the largest and is never allocated during a forward.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - One projection at a time: every user dequantizes and runs its GEMM on the model's forward
//!   stream, so the next projection's dequantize is ordered after this GEMM's read.
//! - [`reserve`] runs at load only; [`get`] errors when nothing was reserved.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-10-12: The scratch: its address and capacity in BF16 elements.
#[derive(Clone, Copy, Debug)]
pub struct Scratch {
    pub ptr: DevicePtr,
    pub elems: usize,
}

fn scratch() -> &'static Mutex<Option<Scratch>> {
    static S: OnceLock<Mutex<Option<Scratch>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// 2026-10-12: Grow the scratch to hold an `[n, k]` BF16 weight; a no-op when it already does.
/// The caller has checked its lever is on.
pub fn reserve(gpu: &dyn GpuBackend, n: usize, k: usize, what: &str) -> Result<()> {
    let elems = n * k;
    let mut g = scratch().lock().unwrap_or_else(|e| e.into_inner());
    if g.is_some_and(|s| s.elems >= elems) {
        return Ok(());
    }
    if let Some(old) = g.take() {
        gpu.free(old.ptr)?;
    }
    let ptr = gpu
        .alloc(elems * 2)
        .with_context(|| format!("{what}: dequant GEMM scratch of {n} x {k} BF16"))?;
    *g = Some(Scratch { ptr, elems });
    Ok(())
}

/// 2026-10-12: The scratch reserved at load, or an error when there is none.
pub fn get(what: &str) -> Result<Scratch> {
    scratch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .with_context(|| format!("{what}: no dequant GEMM scratch was reserved at load"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

    /// 2026-10-12: The scratch grows to the largest reservation and keeps it: a smaller or equal
    /// one after it allocates nothing, a larger one replaces it. One test owns the process-wide
    /// scratch.
    #[test]
    fn the_scratch_grows_to_the_largest_reservation_only() {
        let gpu = MockGpuBackend::new();
        reserve(&gpu, 4096, 768, "a").unwrap();
        let first = get("a").unwrap();
        assert!(first.elems >= 4096 * 768);
        reserve(&gpu, 512, 4096, "b").unwrap();
        assert_eq!(
            get("b").unwrap().ptr,
            first.ptr,
            "a smaller weight keeps the scratch"
        );
        reserve(&gpu, 4096, 11264, "c").unwrap();
        let grown = get("c").unwrap();
        assert_eq!(grown.elems, 4096 * 11264);
        assert_ne!(grown.ptr, first.ptr, "a larger weight replaces it");
        assert!(
            gpu.read_alloc(first.ptr).is_none(),
            "the old scratch is freed"
        );
        assert_eq!(
            gpu.read_alloc(grown.ptr).map(|b| b.len()),
            Some(4096 * 11264 * 2)
        );
    }
}
