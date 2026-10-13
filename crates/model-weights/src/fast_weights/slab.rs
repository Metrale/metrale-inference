// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: One device slab per shard for the tensors that stay resident as loaded.
//!
//! Each driver allocation is rounded to its own granule (`gpu-runtime`'s
//! `cuda_backend/slab.rs` has the H100 measurements): an 819,200 B NVFP4 expert projection
//! costs 1 MiB on its own, which is ~9 GiB over a 512-expert, 48-layer checkpoint. Carving a
//! shard's small quantized tensors from one allocation removes that.
//!
//! Only tensors that stay as loaded go in the slab. A slab's memory returns only when all of
//! its members are freed, so a tensor a model loader converts and frees (BF16 weights
//! requantized or transposed at bind) would strand its bytes; those, and anything at or above
//! [`SLAB_MAX_TENSOR_BYTES`] (where rounding is a few percent at most), keep their own
//! allocation.

use super::header::TensorMeta;
use crate::weights::WeightDtype;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-10-13: Tensors this large or larger keep their own allocation: a 16 MiB request
/// rounds by at most one 2 MiB granule (12.5%), and from there down to nothing as size grows.
pub(super) const SLAB_MAX_TENSOR_BYTES: usize = 16 << 20;

/// 2026-10-13: Whether a tensor goes in its shard's slab: smaller than the cap, and not BF16
/// (the dtype model loaders rewrite at bind and free).
pub(super) fn slab_eligible(meta: &TensorMeta) -> bool {
    meta.len < SLAB_MAX_TENSOR_BYTES && meta.dtype != WeightDtype::BF16
}

/// 2026-10-13: A device pointer for every slab-eligible tensor of `tensors` (same order;
/// `None` for the rest), all from one `alloc_slab`. When the slab cannot be allocated, every
/// entry is `None` and the caller allocates per tensor, which keeps its managed-memory
/// fallback.
pub(super) fn alloc_shard_slab(
    gpu: &dyn GpuBackend,
    tensors: &[TensorMeta],
) -> Vec<Option<DevicePtr>> {
    let picked: Vec<usize> = (0..tensors.len())
        .filter(|&i| slab_eligible(&tensors[i]))
        .collect();
    let mut out = vec![None; tensors.len()];
    if picked.is_empty() {
        return out;
    }
    let sizes: Vec<usize> = picked.iter().map(|&i| tensors[i].len).collect();
    match gpu.alloc_slab(&sizes) {
        Ok(ptrs) => {
            for (i, p) in picked.into_iter().zip(ptrs) {
                out[i] = Some(p);
            }
        }
        Err(e) => tracing::warn!(
            "slab of {} tensor(s) ({} bytes) not allocated ({e}); allocating them one by one",
            sizes.len(),
            sizes.iter().sum::<usize>()
        ),
    }
    out
}

#[cfg(test)]
#[path = "slab_tests.rs"]
mod tests;
