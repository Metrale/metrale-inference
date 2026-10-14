// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `embed_tokens` gathered in place from page-locked host memory.
//!
//! Every consumer of the input embedding is a row gather (the batched embed kernel, the
//! n-gram fused embed, the single-row copy), so the table need not be device-resident: a
//! kernel reads the few rows a step touches through the table's mapped device address. On
//! an 80 GB device this returns the table's 1.18 GiB to the KV cache and the GDN state of
//! Qwen3.8-Flash-Next. The bytes are the checkpoint's, unchanged: placement only, never
//! precision.
//!
//! Opt-in with `METRALE_QWEN4EXP_EMBED_ON_HOST=1`. A tied checkpoint never takes it, since
//! its lm_head is a dense GEMM over the same table.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - the defer hook and [`load`] agree through the store: [`load`] maps the table from host
//!   memory exactly when the hook deferred it, and never re-reads the lever.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::weight_map::DenseWeight;
use metrale_model_weights::weights::{DeferHook, DeferredTensor, WeightDtype};

/// 2026-10-13: The lever that places the input embedding in host memory.
pub(super) const EMBED_ON_HOST_LEVER: &str = "METRALE_QWEN4EXP_EMBED_ON_HOST";

/// 2026-10-13: The defer hook for the input embedding: matches exactly the BF16
/// `{prefix}.embed_tokens.weight` when `on_host` is set and the checkpoint is untied.
pub(super) fn defer_hook(config: &ModelConfig, on_host: bool) -> Option<DeferHook> {
    if !on_host || config.tie_word_embeddings {
        return None;
    }
    let name = super::embed_name(config);
    Some(std::sync::Arc::new(move |n: &str, dtype: WeightDtype| {
        dtype == WeightDtype::BF16 && n == name
    }))
}

/// 2026-10-13: Whether the lever is set (`1`).
pub(super) fn lever_set() -> bool {
    std::env::var(EMBED_ON_HOST_LEVER).as_deref() == Ok("1")
}

/// 2026-10-13: Read the deferred table into page-locked host memory and return its mapped
/// device address. The buffer lives as long as the process, like every other weight.
pub(super) fn load(d: &DeferredTensor, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let bytes = d.byte_size();
    let host = gpu.alloc_host_pinned(bytes)?;
    // 2026-10-13: SAFETY: `host` is a fresh, zeroed `alloc_host_pinned` region of exactly
    // `bytes` bytes, and nothing else holds it until `host_ptr_to_device` below.
    let dst = unsafe { std::slice::from_raw_parts_mut(host, bytes) };
    let mapped = d.read_into(dst).and_then(|()| {
        gpu.host_ptr_to_device(host).with_context(|| {
            format!("this backend cannot map host memory for a kernel; unset {EMBED_ON_HOST_LEVER}")
        })
    });
    match mapped {
        Ok(weight) => {
            tracing::info!(
                "qwen4_exp: embed_tokens ({:.2} GiB BF16) is read in place from page-locked \
                 host memory ({EMBED_ON_HOST_LEVER}=1); its bytes are the checkpoint's.",
                bytes as f64 / (1u64 << 30) as f64
            );
            Ok(DenseWeight { weight })
        }
        Err(e) => {
            gpu.free_host_pinned(host, bytes)?;
            Err(e.context("qwen4_exp: host-resident embedding"))
        }
    }
}

#[cfg(test)]
#[path = "host_embed_tests.rs"]
mod tests;
