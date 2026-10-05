// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The `--mock` weight store: every tensor of a mock plan synthesized on the host
//! (`metrale_ml_utils::synthesize`, the same bytes `met ml-utils mockify` writes) and uploaded,
//! with no weight file read.
//!
//! Owner: model-weights.
//! Invariants:
//! - A tensor the caller's skip predicate rejects is neither synthesized into the store nor
//!   uploaded; the predicate is the real loader's (`FastSafetensorsLoader::should_skip_tensor`),
//!   so a mock holds what a real load of the mock checkpoint would hold.
//! - Every stored tensor has the plan's dtype and shape.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_ml_utils::write::{batches, synthesize_batch, unit_tensor_ids};
use metrale_ml_utils::{Dtype, MockPlan};

use crate::weights::{WeightDtype, WeightStore, WeightTensor};

/// 2026-10-03: The store dtype of a mock tensor.
pub fn store_dtype(d: Dtype) -> Result<WeightDtype> {
    Ok(match d {
        Dtype::Bf16 => WeightDtype::BF16,
        Dtype::F32 => WeightDtype::FP32,
        Dtype::F8E4m3 => WeightDtype::FP8E4M3,
        Dtype::U8 | Dtype::I8 => WeightDtype::UInt8,
        Dtype::I64 => WeightDtype::Int64,
        Dtype::F16 | Dtype::I32 => bail!("a mock tensor of dtype {} has no store dtype", d.name()),
    })
}

/// 2026-10-03: Synthesize and upload every tensor of `plan` that `skip` keeps, on up to `threads`
/// host threads.
pub fn load_synthetic(
    plan: &MockPlan,
    gpu: &dyn GpuBackend,
    skip: &dyn Fn(&str) -> bool,
    threads: usize,
) -> Result<WeightStore> {
    let wanted: Vec<usize> = (0..plan.units.len())
        .filter(|&u| {
            unit_tensor_ids(&plan.units[u])
                .iter()
                .any(|&t| !skip(&plan.tensors[t].name))
        })
        .collect();
    let mut weights = HashMap::new();
    let mut bytes_total = 0u64;
    for batch in batches(plan, &wanted, threads) {
        let out = synthesize_batch(plan, &batch).map_err(|e| anyhow::anyhow!("{e}"))?;
        for (t, bytes) in out.into_iter().flatten() {
            let o = &plan.tensors[t];
            if skip(&o.name) {
                continue;
            }
            let ptr = gpu
                .alloc(bytes.len().max(1))
                .with_context(|| format!("allocating mock tensor {}", o.name))?;
            gpu.copy_h2d(&bytes, ptr)?;
            bytes_total += bytes.len() as u64;
            weights.insert(
                o.name.clone(),
                WeightTensor {
                    ptr,
                    shape: o.shape.iter().map(|&d| d as usize).collect(),
                    dtype: store_dtype(o.dtype)?,
                },
            );
        }
    }
    tracing::info!(
        "Mock weights: {} tensors, {:.2} GB synthesized (mock {})",
        weights.len(),
        bytes_total as f64 / 1e9,
        &plan.digest[..12]
    );
    Ok(WeightStore::from_map(weights))
}

#[cfg(test)]
#[path = "synthetic_tests.rs"]
mod synthetic_tests;
