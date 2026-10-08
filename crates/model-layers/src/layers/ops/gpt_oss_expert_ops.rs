// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: GPT-OSS staged expert operators. BF16 bias reuses nllb_encoder;
//! activation and weighted reduction keep their explicit BF16 tensor boundaries.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: In-place BF16 `[rows,cols]` += BF16 `[cols]`. Requires the existing
/// `nllb_encoder::nllb_bias_bf16` kernel; the bmm result has already rounded.
pub fn gpt_oss_expert_bias_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    values: DevicePtr,
    bias: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    let count = geometry(kernel, rows, cols)?;
    separate(
        range(values, u64::from(count) * 2, 2)?,
        range(bias, u64::from(cols) * 2, 2)?,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([count.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(values)
        .arg_ptr(bias)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

/// 2026-10-07: Interleaved BF16 `[count,2]` -> BF16 `[count]`, distinct buffers.
/// Requires `gpt_oss_expert_ops::gpt_oss_swiglu_bf16`.
pub fn gpt_oss_swiglu_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_up: DevicePtr,
    output: DevicePtr,
    count: u32,
    stream: u64,
) -> Result<()> {
    geometry(kernel, 1, count)?;
    separate(
        range(gate_up, u64::from(count) * 4, 2)?,
        range(output, u64::from(count) * 2, 2)?,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([count.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_up)
        .arg_ptr(output)
        .arg_u32(count)
        .launch(stream)
}

/// 2026-10-07: Selected BF16 `[4,tokens,hidden]`, dense BF16 scores `[tokens,32]`,
/// U32 IDs `[tokens,4]` -> BF16 `[tokens,hidden]`. Requires distinct output and
/// `gpt_oss_expert_ops::gpt_oss_expert_reduce_bf16`. IDs must be unique in0..32;
/// invalid IDs produce NaN. Finite unselected outputs are assumed (sparse path).
#[allow(clippy::too_many_arguments)]
pub fn gpt_oss_expert_reduce_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    selected: DevicePtr,
    scores: DevicePtr,
    ids: DevicePtr,
    output: DevicePtr,
    tokens: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    let count = geometry(kernel, tokens, hidden)?;
    let out = range(output, u64::from(count) * 2, 2)?;
    for input in [
        range(selected, u64::from(count) * 8, 2)?,
        range(scores, u64::from(tokens) * 64, 2)?,
        range(ids, u64::from(tokens) * 16, 4)?,
    ] {
        separate(out, input)?;
    }
    KernelLaunch::new(gpu, kernel)
        .grid([count.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(selected)
        .arg_ptr(scores)
        .arg_ptr(ids)
        .arg_ptr(output)
        .arg_u32(tokens)
        .arg_u32(hidden)
        .launch(stream)
}
fn geometry(kernel: KernelHandle, rows: u32, cols: u32) -> Result<u32> {
    ensure!(
        kernel.0 != 0 && rows > 0 && cols > 0,
        "GPT-OSS expert operator: missing kernel or empty geometry"
    );
    rows.checked_mul(cols)
        .context("GPT-OSS expert element count overflow")
}
fn range(ptr: DevicePtr, bytes: u64, alignment: u64) -> Result<(u64, u64)> {
    ensure!(
        ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
        "GPT-OSS expert null/misaligned address"
    );
    Ok((
        ptr.0,
        ptr.0
            .checked_add(bytes)
            .context("GPT-OSS expert address overflow")?,
    ))
}
fn separate(a: (u64, u64), b: (u64, u64)) -> Result<()> {
    ensure!(
        a.1 <= b.0 || b.1 <= a.0,
        "GPT-OSS expert overlapping buffers"
    );
    Ok(())
}
