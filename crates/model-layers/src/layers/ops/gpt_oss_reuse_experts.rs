// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded expert-row reuse with unchanged per-token arithmetic.
use super::gpt_oss_token_experts::{GptOssTokenExperts, range, separate};
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
/// 2026-10-07: Plan `[32,tokens+1]` contains count followed by token*4+slot.
/// Caller validates complete unique coverage and expert identity before upload.
#[allow(clippy::too_many_arguments)]
pub fn gpt_oss_mxfp4_reuse_experts(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    blocks: DevicePtr,
    scales: DevicePtr,
    input: DevicePtr,
    ids: DevicePtr,
    plan: DevicePtr,
    output: DevicePtr,
    g: &GptOssTokenExperts,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "GPT expert reuse missing kernel");
    let (out_count, input_count, packed) = g.counts()?;
    let out = range(output, u64::from(out_count) * 2, 2)?;
    for source in [
        range(blocks, packed, 1)?,
        range(scales, packed / 16, 1)?,
        range(input, u64::from(input_count) * 2, 2)?,
        range(ids, u64::from(g.tokens) * 16, 4)?,
        range(plan, u64::from(g.tokens + 1) * 32 * 4, 4)?,
    ] {
        separate(out, source)?;
    }
    KernelLaunch::new(gpu, kernel)
        .grid([g.rows.div_ceil(4), 32, g.tokens.div_ceil(4)])
        .block([128, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(scales)
        .arg_ptr(input)
        .arg_ptr(ids)
        .arg_ptr(plan)
        .arg_ptr(output)
        .arg_u32(g.rows)
        .arg_u32(g.cols)
        .arg_u32(g.tokens)
        .arg_u32(if g.per_slot_input {
            g.tokens * g.cols
        } else {
            0
        })
        .launch(stream)
}
