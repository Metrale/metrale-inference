// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Borrowed BF16 views with validated shape and byte-address extents.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::BTreeSet;

/// 2026-10-07: A checkpoint BF16 tensor. No conversion to FP32 or quantized storage.
/// The allocation remains owned by the store's loader; metadata supplies its extent.
pub struct GptOssBf16Tensor<'a> {
    tensor: &'a WeightTensor,
    bytes: usize,
}

impl GptOssBf16Tensor<'_> {
    /// 2026-10-07: Device pointer, unchanged from the checkpoint store.
    pub fn ptr(&self) -> DevicePtr {
        self.tensor.ptr
    }
    /// 2026-10-07: Validated checkpoint dimensions, unchanged by binding.
    pub fn shape(&self) -> &[usize] {
        &self.tensor.shape
    }
    /// 2026-10-07: Explicit BF16 storage identity; compute precision is separate.
    pub fn dtype(&self) -> WeightDtype {
        self.tensor.dtype
    }
    /// 2026-10-07: Validated metadata byte extent, not a GPU allocation query.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

pub(super) fn checked_product(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &dim| {
        ensure!(dim > 0, "{name}: zero dimension");
        n.checked_mul(dim)
            .with_context(|| format!("{name}: size overflow"))
    })
}

pub(super) fn bind_bf16<'a>(
    store: &'a WeightStore,
    name: &str,
    shape: &[usize],
    names: &mut BTreeSet<String>,
) -> Result<GptOssBf16Tensor<'a>> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.dtype == WeightDtype::BF16,
        "{name}: pinned GPT-OSS checkpoint requires BF16 storage, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape == shape,
        "{name}: expected shape {shape:?}, got {:?}",
        tensor.shape
    );
    ensure!(!tensor.ptr.is_null(), "{name}: null checkpoint address");
    let bytes = checked_product(shape, name)?
        .checked_mul(2)
        .with_context(|| format!("{name}: BF16 byte size overflow"))?;
    let extent =
        u64::try_from(bytes).with_context(|| format!("{name}: byte size exceeds address width"))?;
    tensor
        .ptr
        .0
        .checked_add(extent)
        .with_context(|| format!("{name}: address extent overflow"))?;
    names.insert(name.to_owned());
    Ok(GptOssBf16Tensor { tensor, bytes })
}
