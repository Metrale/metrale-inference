// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Token-grid expert composition; unchanged dots and BF16 boundaries.
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: Fixed32-expert/top4 geometry with slot-major intermediate storage.
pub struct GptOssTokenExperts {
    pub tokens: u32,
    pub rows: u32,
    pub cols: u32,
    pub per_slot_input: bool,
}
impl GptOssTokenExperts {
    pub(super) fn counts(&self) -> Result<(u32, u32, u64)> {
        ensure!(
            (1..=128).contains(&self.tokens)
                && self.rows > 0
                && self.cols > 0
                && self.cols.is_multiple_of(32),
            "GPT token expert geometry"
        );
        let output = self
            .rows
            .checked_mul(self.tokens)
            .and_then(|n| n.checked_mul(4))
            .context("GPT token expert output overflow")?;
        let input = self
            .cols
            .checked_mul(self.tokens)
            .and_then(|n| n.checked_mul(if self.per_slot_input { 4 } else { 1 }))
            .context("GPT token expert input overflow")?;
        let packed = u64::from(self.rows)
            .checked_mul(u64::from(self.cols))
            .and_then(|n| n.checked_mul(16))
            .context("GPT token expert weight overflow")?;
        Ok((output, input, packed))
    }
}
/// 2026-10-07: Input `[tokens,K]` or `[4,tokens,K]`; output `[4,tokens,N]`.
/// IDs remain `[tokens,4]`, host validated and guarded again by the kernel.
/// Pointer extents remain caller-owned; numeric extents and output aliasing fail closed.
#[allow(clippy::too_many_arguments)]
pub fn gpt_oss_mxfp4_token_experts(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    blocks: DevicePtr,
    scales: DevicePtr,
    input: DevicePtr,
    ids: DevicePtr,
    output: DevicePtr,
    g: &GptOssTokenExperts,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "GPT token expert missing kernel");
    let (out_count, input_count, packed) = g.counts()?;
    let out = range(output, u64::from(out_count) * 2, 2)?;
    for source in [
        range(blocks, packed, 1)?,
        range(scales, packed / 16, 1)?,
        range(input, u64::from(input_count) * 2, 2)?,
        range(ids, u64::from(g.tokens) * 16, 4)?,
    ] {
        separate(out, source)?;
    }
    KernelLaunch::new(gpu, kernel)
        .grid([g.rows.div_ceil(4), 4, g.tokens])
        .block([128, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(scales)
        .arg_ptr(input)
        .arg_ptr(ids)
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
/// 2026-10-07: Separate BF16 bias on `[4,tokens,rows]`; no GEMV epilogue fusion.
#[allow(clippy::too_many_arguments)]
pub fn gpt_oss_token_expert_bias(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    values: DevicePtr,
    bias: DevicePtr,
    ids: DevicePtr,
    tokens: u32,
    rows: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0 && (1..=128).contains(&tokens) && rows > 0,
        "GPT token expert bias geometry/kernel"
    );
    let count = rows
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(4))
        .context("GPT token expert bias overflow")?;
    let out = range(values, u64::from(count) * 2, 2)?;
    separate(out, range(bias, u64::from(rows) * 64, 2)?)?;
    separate(out, range(ids, u64::from(tokens) * 16, 4)?)?;
    KernelLaunch::new(gpu, kernel)
        .grid([count.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(values)
        .arg_ptr(bias)
        .arg_ptr(ids)
        .arg_u32(rows)
        .arg_u32(tokens)
        .launch(stream)
}
pub(super) fn range(ptr: DevicePtr, bytes: u64, align: u64) -> Result<(u64, u64)> {
    ensure!(
        !ptr.is_null() && ptr.0.is_multiple_of(align),
        "GPT token expert null/misaligned buffer"
    );
    Ok((
        ptr.0,
        ptr.0
            .checked_add(bytes)
            .context("GPT token expert address overflow")?,
    ))
}
pub(super) fn separate(a: (u64, u64), b: (u64, u64)) -> Result<()> {
    ensure!(
        a.1 <= b.0 || b.1 <= a.0,
        "GPT token expert output aliases input"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_rejects_tail_groups_and_arithmetic_overflow() {
        for tokens in [1, 2, 15, 16, 17, 31, 64, 127, 128] {
            assert!(
                GptOssTokenExperts {
                    tokens,
                    rows: 35,
                    cols: 96,
                    per_slot_input: true
                }
                .counts()
                .is_ok()
            );
        }
        for (tokens, rows, cols) in [
            (0, 35, 96),
            (129, 35, 96),
            (1, 0, 96),
            (1, 35, 95),
            (16, u32::MAX, 32),
            (16, 35, u32::MAX - 31),
        ] {
            assert!(
                GptOssTokenExperts {
                    tokens,
                    rows,
                    cols,
                    per_slot_input: true
                }
                .counts()
                .is_err()
            );
        }
        assert!(range(DevicePtr(u64::MAX - 1), 4, 2).is_err());
        assert!(separate((10, 20), (12, 22)).is_err());
    }
}
