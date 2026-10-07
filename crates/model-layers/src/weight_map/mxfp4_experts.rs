// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Zero-copy binding of checkpoint MXFP4 expert blocks. The caller
//! declares this format; U8 storage alone never selects a quantization policy.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::WeightQuantFormat;

/// 2026-10-06: Validated `[experts, rows, cols/32, 16]` packed E2M1 bytes and
/// `[experts, rows, cols/32]` raw U8 E8M0 scale bytes. No global scale, conversion,
/// transpose, or activation policy is implied. Rows retain checkpoint order,
/// including interleaved gate/up rows. The store must retain the allocations.
///
/// Exact dimensions come from the architecture contract, not tensor element
/// counts. WeightStore owns tensor metadata but does not expose allocation
/// lengths; this checks metadata extents and address overflow, relying on the
/// store loader to have allocated the declared bytes.
pub struct PackedMxfp4Experts<'a> {
    blocks: &'a WeightTensor,
    scales: &'a WeightTensor,
    experts: usize,
    rows: usize,
    cols: usize,
    packed_stride: usize,
    scale_stride: usize,
}

impl<'a> PackedMxfp4Experts<'a> {
    /// 2026-10-06: Bind the exact rank4/rank3 raw-byte layout used by GPT-OSS.
    /// Typed FP8 tensors are refused: this accessor promises untouched U8 bytes.
    pub fn bind(
        store: &'a WeightStore,
        blocks_key: &str,
        scales_key: &str,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        ensure!(
            experts > 0 && rows > 0 && cols > 0,
            "MXFP4 dimensions must be nonzero"
        );
        ensure!(
            cols.is_multiple_of(32),
            "MXFP4 columns must contain complete groups of 32"
        );
        let blocks = store.get(blocks_key)?;
        let scales = store.get(scales_key)?;
        ensure!(
            blocks.dtype == WeightDtype::UInt8,
            "{blocks_key}: expected raw U8 E2M1 blocks"
        );
        ensure!(
            scales.dtype == WeightDtype::UInt8,
            "{scales_key}: expected raw U8 E8M0 scales"
        );
        ensure!(
            blocks.shape == [experts, rows, cols / 32, 16],
            "{blocks_key}: expected [experts, rows, cols/32, 16]"
        );
        ensure!(
            scales.shape == [experts, rows, cols / 32],
            "{scales_key}: expected [experts, rows, cols/32]"
        );
        let scale_stride = rows
            .checked_mul(cols / 32)
            .context("MXFP4 scale stride overflow")?;
        let packed_stride = scale_stride
            .checked_mul(16)
            .context("MXFP4 packed stride overflow")?;
        check_extent(blocks.ptr, packed_stride, experts, blocks_key)?;
        check_extent(scales.ptr, scale_stride, experts, scales_key)?;
        Ok(Self {
            blocks,
            scales,
            experts,
            rows,
            cols,
            packed_stride,
            scale_stride,
        })
    }

    /// 2026-10-06: Select a contiguous expert without changing nibble or scale bytes.
    pub fn expert(&self, index: usize) -> Result<Mxfp4ExpertView<'_>> {
        ensure!(
            index < self.experts,
            "MXFP4 expert {index} outside count {}",
            self.experts
        );
        // 2026-10-06: bind checked full extents, so these smaller offsets cannot wrap.
        Ok(Mxfp4ExpertView {
            owner: self,
            weight: self.blocks.ptr.offset(index * self.packed_stride),
            scales: self.scales.ptr.offset(index * self.scale_stride),
        })
    }

    /// 2026-10-06: The validated checkpoint expert count.
    pub fn expert_count(&self) -> usize {
        self.experts
    }
}

fn check_extent(ptr: DevicePtr, stride: usize, count: usize, key: &str) -> Result<()> {
    ensure!(!ptr.is_null(), "{key}: null MXFP4 allocation");
    let bytes = stride
        .checked_mul(count)
        .context("MXFP4 tensor size overflow")?;
    let bytes = u64::try_from(bytes).context("MXFP4 byte count exceeds device address width")?;
    ptr.0
        .checked_add(bytes)
        .with_context(|| format!("{key}: MXFP4 address extent overflow"))?;
    Ok(())
}

/// 2026-10-06: A borrowed, format-specific expert view. Deliberately not convertible
/// into the untagged QuantizedWeight used by NVFP4 APIs. Kernel call sites must
/// select the MXFP4 E8M0 policy explicitly. This type does not establish execution.
pub struct Mxfp4ExpertView<'a> {
    owner: &'a PackedMxfp4Experts<'a>,
    weight: DevicePtr,
    scales: DevicePtr,
}

impl Mxfp4ExpertView<'_> {
    /// 2026-10-06: Packed E2M1 bytes, two values per byte, in checkpoint row order.
    pub fn weight(&self) -> DevicePtr {
        self.weight
    }
    /// 2026-10-06: Raw E8M0 scale bytes, one per 32 weights, in row-major order.
    pub fn scales(&self) -> DevicePtr {
        self.scales
    }
    /// 2026-10-06: Explicit weight format for kernel policy selection.
    pub fn format(&self) -> WeightQuantFormat {
        WeightQuantFormat::Mxfp4E8m0
    }
    /// 2026-10-06: Output rows, including both gate/up rows when packed together.
    pub fn rows(&self) -> usize {
        self.owner.rows
    }
    /// 2026-10-06: Logical input columns before nibble packing.
    pub fn cols(&self) -> usize {
        self.owner.cols
    }
    /// 2026-10-06: Bound byte extent of this expert's packed values.
    pub fn packed_bytes(&self) -> usize {
        self.owner.packed_stride
    }
    /// 2026-10-06: Bound byte extent of this expert's E8M0 scales.
    pub fn scale_bytes(&self) -> usize {
        self.owner.scale_stride
    }
}

#[cfg(test)]
#[path = "mxfp4_experts_tests.rs"]
mod tests;
