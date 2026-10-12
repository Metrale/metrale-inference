// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: FP32 projection accumulator plus BF16 bias, then one BF16 rounding.
//! Owner: model-layers ops. Named LKB residual: `projection_bias_before_cast`.
//! Pair with `gemv::dense_gemv_bf16_fp32out` for Q/K/V/O/router projections.
//! GPT-OSS expert bmm-plus-bias has a separate intermediate rounding contract;
//! this operator must not be substituted there without reference parity.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: `out[r,c] = bf16(accum[r,c] + float(bias[c]))`.
/// Inputs are contiguous FP32 `[rows, cols]` and BF16 `[cols]`; output is
/// contiguous BF16 `[rows, cols]`. Output may not overlap either input. The caller
/// owns allocations covering these byte ranges and supplies the exact kernel
/// `projection_bias::projection_bias_bf16`; a null kernel is refused. This only
/// launches the epilogue, not the preceding GEMV. No model target is registered.
pub fn projection_bias_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    accum: DevicePtr,
    bias: DevicePtr,
    output: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "projection bias: kernel is missing");
    ensure!(rows > 0 && cols > 0, "projection bias: empty geometry");
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow::anyhow!("projection bias: element count overflow"))?;
    let a = range(accum, u64::from(count) * 4, 4)?;
    let b = range(bias, u64::from(cols) * 2, 2)?;
    let o = range(output, u64::from(count) * 2, 2)?;
    for input in [a, b] {
        ensure!(
            o.1 <= input.0 || input.1 <= o.0,
            "projection bias: output overlaps input"
        );
    }
    KernelLaunch::new(gpu, kernel)
        .grid([count.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(accum)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(count)
        .arg_u32(cols)
        .launch(stream)
}

fn range(ptr: DevicePtr, bytes: u64, alignment: u64) -> Result<(u64, u64)> {
    ensure!(
        ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
        "projection bias: null or misaligned pointer"
    );
    let end = ptr
        .0
        .checked_add(bytes)
        .ok_or_else(|| anyhow::anyhow!("projection bias: address overflow"))?;
    Ok((ptr.0, end))
}
