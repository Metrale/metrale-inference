// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Debug-build range check on the activations an unscaled E4M3 cast reads.
//!
//! Owner: model-layers ops.
//! Invariants: release builds compile the check out; it never changes a launch.
//!
//! Several W4A16 and FP8 prefill kernels cast their BF16 A operand to E4M3 with no scale
//! (`bf16x4_to_e4m3x4`, `cvt.rn.satfinite`), and the `bf16_to_fp8` op casts a buffer the
//! same way (the FP8 activation path, and some weights at load). E4M3's largest finite value is 448, so a larger
//! activation saturates without an error. A kernel declares the cast with
//! `<entry>_a_e4m3` (`GpuBackend::kernel_casts_a_to_e4m3`). Measured on the certified
//! dense 27B models (Qwen3.8-27B-NVFP4 and Qwen3.6-27B-NVFP4; short, 5k, 27.5k-token,
//! tool-call and code prompts), the largest activation at these sites was 134, a margin of
//! 3.3x. In debug builds every such launch, and every `bf16_to_fp8`, checks its source and
//! panics past 448.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// 2026-09-29: E4M3's largest finite magnitude; an unscaled cast saturates above it.
pub const E4M3_MAX: f32 = 448.0;

/// 2026-09-29: When `kernel` casts its A operand to E4M3 with no scale, check the
/// `[m, k]` BF16 `input` it reads (see [`check_e4m3_range`]). A no-op otherwise, and in
/// release builds.
pub fn check_e4m3_activation_range(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if cfg!(debug_assertions) && gpu.kernel_casts_a_to_e4m3(kernel) {
        check_e4m3_range(gpu, input, m as usize * k as usize, stream)?;
    }
    Ok(())
}

/// 2026-09-29: Debug builds: copy the `elements` BF16 values at `input` to the host on
/// `stream` and panic when one exceeds [`E4M3_MAX`] in magnitude. NaN is not checked
/// here. Skipped under graph capture, where a copy to the host cannot run. Release
/// builds: a no-op.
pub fn check_e4m3_range(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    elements: usize,
    stream: u64,
) -> Result<()> {
    if !cfg!(debug_assertions) || elements == 0 || gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    let mut buf = vec![0u8; elements * 2];
    gpu.copy_d2h_on_stream(input, &mut buf, stream)?;
    let amax = buf
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).abs())
        .filter(|v| !v.is_nan())
        .fold(0.0f32, f32::max);
    assert!(
        amax <= E4M3_MAX,
        "unscaled E4M3 cast would saturate: activation |x| = {amax} > {E4M3_MAX}"
    );
    Ok(())
}

#[cfg(test)]
#[path = "e4m3_range_tests.rs"]
mod tests;
