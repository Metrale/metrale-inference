// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit packed MXFP4 expert GEMV launcher, with BF16 output before
//! bias. Correctness residual; no optimized/model-target registration is implied.

use crate::weight_map::Mxfp4ExpertView;
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: Launch row-major E2M1/E8M0-group32 `[N,K]` times BF16 `[K]`.
/// Output is BF16 `[N]`, materialized before any separate expert bias addition.
/// Caller supplies `gpt_oss_mxfp4_gemv::gpt_oss_mxfp4_gemv_bf16` and owns buffers.
/// Allocation extents are a caller obligation; arithmetic extents and output
/// overlap are checked. Input and weight storage are left unchanged.
pub fn gpt_oss_mxfp4_gemv_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Mxfp4ExpertView<'_>,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "GPT-OSS MXFP4 GEMV: missing kernel");
    let rows = u32::try_from(weight.rows()).context("MXFP4 rows exceed kernel ABI")?;
    let cols = u32::try_from(weight.cols()).context("MXFP4 columns exceed kernel ABI")?;
    ensure!(
        cols <= u32::MAX - 31,
        "MXFP4 column iteration would overflow"
    );
    let out = extent(output, u64::from(rows) * 2, 2)?;
    for (ptr, bytes, alignment) in [
        (input, u64::from(cols) * 2, 2),
        (weight.weight(), u64::try_from(weight.packed_bytes())?, 1),
        (weight.scales(), u64::try_from(weight.scale_bytes())?, 1),
    ] {
        let source = extent(ptr, bytes, alignment)?;
        ensure!(
            out.1 <= source.0 || source.1 <= out.0,
            "MXFP4 output overlaps input"
        );
    }
    KernelLaunch::new(gpu, kernel)
        .grid([rows.div_ceil(4), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(weight.weight())
        .arg_ptr(weight.scales())
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

fn extent(ptr: DevicePtr, bytes: u64, alignment: u64) -> Result<(u64, u64)> {
    ensure!(
        ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
        "MXFP4 null or misaligned buffer"
    );
    Ok((
        ptr.0,
        ptr.0.checked_add(bytes).context("MXFP4 address overflow")?,
    ))
}
