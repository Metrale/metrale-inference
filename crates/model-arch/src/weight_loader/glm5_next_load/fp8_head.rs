// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: An LM head stored in 8 bits (F8_E4M3 `lm_head.weight` with its block-scaled
//! `lm_head.weight_scale`), widened once to the BF16 head every head path reads.
//!
//! The defer hook keeps both tensors off the device ([`is_fp8_head_tensor`]), so the 8-bit copy
//! is never resident and nothing has to be pruned; [`head_weight`] reads them from their shard,
//! widens them on the host (`fp8_widen`) and hands the BF16 buffer to the store's derived
//! buffers, which free it with the store. A 16-bit head is never deferred and binds as before.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - A 16-bit head binds the store tensor itself; an F8_E4M3 head without a scale, or a
//!   deferred head of any other dtype, is refused.

use super::expert_tp::{meta, read_all};
use super::fp8_widen::widen_fp8_to_bf16;
use super::*;

/// 2026-10-17: The checkpoint's head.
pub(super) const HEAD: &str = "lm_head.weight";
/// 2026-10-17: Its block scales when the head is 8-bit.
pub(super) const HEAD_SCALE: &str = "lm_head.weight_scale";

/// 2026-10-17: Whether the defer hook keeps `name` off the device: an F8_E4M3 head, or a head
/// scale (absent from 16-bit checkpoints).
pub(super) fn is_fp8_head_tensor(name: &str, dtype: WeightDtype) -> bool {
    (name == HEAD && dtype == WeightDtype::FP8E4M3) || name == HEAD_SCALE
}

/// 2026-10-17: The head as a BF16 `[vocab, hidden]` weight: the store's tensor when it is
/// 16-bit, else the widening of the F8_E4M3 head under its scales, adopted by `store.derived()`.
/// The 8-bit pair is read deferred or, from a loader that does not honour the defer hook, back
/// from the device.
pub(super) fn head_weight(gpu: &dyn GpuBackend, store: &WeightStore) -> Result<DenseWeight> {
    let (dtype, shape) = meta(store, HEAD)?;
    if dtype != WeightDtype::FP8E4M3 {
        if store.deferred(HEAD).is_some() {
            bail!("{HEAD} is deferred as {dtype:?}; only an F8_E4M3 head is");
        }
        return dense(store, HEAD);
    }
    let (scale_dtype, scale_shape) = meta(store, HEAD_SCALE).with_context(|| {
        format!("{HEAD} is F8_E4M3 with no {HEAD_SCALE}; an 8-bit head needs its block scales")
    })?;
    let w = read_all(gpu, store, HEAD)?;
    let s = read_all(gpu, store, HEAD_SCALE)?;
    let bf16 = widen_fp8_to_bf16(HEAD, &w, &shape, scale_dtype, &scale_shape, &s)?;
    let weight = gpu.alloc(bf16.len())?;
    gpu.copy_h2d(&bf16, weight)?;
    store
        .derived()
        .adopt("glm5_next lm_head bf16", weight, bf16.len());
    tracing::info!(
        "glm5_next: {HEAD} widened from F8_E4M3 {shape:?} to BF16 under its {scale_dtype:?} scales"
    );
    Ok(DenseWeight { weight })
}

#[cfg(test)]
#[path = "fp8_head_tests.rs"]
mod tests;
