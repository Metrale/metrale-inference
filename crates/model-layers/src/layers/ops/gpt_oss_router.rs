// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Selected-logit BF16 router policy for the pinned 32-expert model.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: Inputs `[rows,32]` BF16 finite logits, outputs `[rows,4]` U32 IDs
/// and dense `[rows,32]` BF16 scores. Lower expert ID wins ties; reference top-k
/// tie order is unspecified and is not claimed bit-identical. Caller supplies
/// `moe_topk::moe_topk_selected_bf16_rows` and owns complete allocation extents.
pub fn gpt_oss_router_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    indices: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0 && rows > 0,
        "router: missing kernel or empty rows"
    );
    ensure!(rows <= i32::MAX as u32, "router: grid overflow");
    let mut ranges = [(0u64, 0u64); 3];
    for (index, (ptr, bytes, alignment)) in [
        (logits, u64::from(rows) * 64, 2),
        (indices, u64::from(rows) * 16, 4),
        (scores, u64::from(rows) * 64, 2),
    ]
    .into_iter()
    .enumerate()
    {
        ensure!(
            ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
            "router: invalid pointer"
        );
        let end = ptr
            .0
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("router: address overflow"))?;
        for &(start, previous_end) in &ranges[..index] {
            ensure!(
                end <= start || previous_end <= ptr.0,
                "router: aliased buffers"
            );
        }
        ranges[index] = (ptr.0, end);
    }
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(indices)
        .arg_ptr(scores)
        .arg_u32(32)
        .arg_u32(4)
        .launch(stream)
}
