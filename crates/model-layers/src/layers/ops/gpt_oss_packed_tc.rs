// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit packed tensor-core diagnostic; no serving policy selection.
use super::gpt_oss_token_experts::{range, separate};
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

fn geometry(tokens: u32, rows: u32, cols: u32, max_expert_rows: u32) -> Result<()> {
    ensure!(
        (1..=128).contains(&tokens)
            && matches!(rows, 2880 | 5760)
            && cols == 2880
            && max_expert_rows > 0
            && max_expert_rows <= tokens,
        "GPT packed TC geometry"
    );
    Ok(())
}
/// 2026-10-07: Caller supplies a validated complete expert plan and 32 live pointer-table entries.
/// Kernel writes expert-packed BF16 rows before separate bias; this is a distinct reduction policy.
#[allow(clippy::too_many_arguments)]
pub fn gpt_oss_packed_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    block_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2: DevicePtr,
    temp: DevicePtr,
    offsets: DevicePtr,
    sorted: DevicePtr,
    tokens: u32,
    rows: u32,
    cols: u32,
    max_expert_rows: u32,
    per_slot_input: bool,
    stream: u64,
) -> Result<()> {
    geometry(tokens, rows, cols, max_expert_rows)?;
    ensure!(kernel.0 != 0, "GPT packed TC missing kernel");
    let out = range(temp, u64::from(4 * tokens * rows) * 2, 2)?;
    for source in [
        range(
            input,
            u64::from(tokens * cols) * if per_slot_input { 8 } else { 2 },
            2,
        )?,
        range(block_ptrs, 256, 8)?,
        range(scale_ptrs, 256, 8)?,
        range(scale2, 128, 4)?,
        range(offsets, 132, 4)?,
        range(sorted, u64::from(tokens) * 16, 4)?,
    ] {
        separate(out, source)?;
    }
    KernelLaunch::new(gpu, kernel)
        .grid([rows.div_ceil(64), max_expert_rows.div_ceil(64), 32])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(block_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(temp)
        .arg_ptr(offsets)
        .arg_ptr(sorted)
        .arg_u32(32)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}
/// 2026-10-07: Existing BF16 row gather with a host-validated inverse permutation.
/// Copy only: no reduction, bias, cast, or activation is folded into this operation.
#[allow(clippy::too_many_arguments)]
pub fn gpt_oss_packed_tc_reorder(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    temp: DevicePtr,
    inverse_rows: DevicePtr,
    output: DevicePtr,
    tokens: u32,
    rows: u32,
    stream: u64,
) -> Result<()> {
    geometry(tokens, rows, 2880, tokens)?;
    ensure!(kernel.0 != 0, "GPT packed TC missing reorder kernel");
    let out = range(output, u64::from(4 * tokens * rows) * 2, 2)?;
    separate(out, range(temp, u64::from(4 * tokens * rows) * 2, 2)?)?;
    separate(out, range(inverse_rows, u64::from(tokens) * 16, 4)?)?;
    KernelLaunch::new(gpu, kernel)
        .grid([4 * tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(temp)
        .arg_ptr(inverse_rows)
        .arg_ptr(output)
        .arg_u32(rows)
        .launch(stream)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_full_checkpoint_geometry() {
        for m in [1, 16, 64, 128] {
            for n in [2880, 5760] {
                assert!(geometry(m, n, 2880, m).is_ok());
            }
        }
        for args in [
            (0, 5760, 2880, 1),
            (129, 5760, 2880, 1),
            (16, 5761, 2880, 16),
            (16, 5760, 2879, 16),
            (16, 5760, 2880, 0),
            (16, 5760, 2880, 17),
        ] {
            assert!(geometry(args.0, args.1, args.2, args.3).is_err());
        }
    }
}
