// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Batch-invariance checks for the GLM-5.3 forward's kernels: for each kernel a row's
//! bytes inside an M-row (or multi-sequence) launch against the same row launched alone, plus the
//! cost of the canonical (row-invariant) tier where it differs from today's.
//!
//! Legs: `dense` (W4A16 projections), `bf16` (BF16-weight FP32-out projections: router, DSA
//! indexer). `GLM_BI_LEGS=dense,bf16` picks legs (default: all).
//!
//! Owner: model-arch examples.
//! Invariants: exits with an error on the first row that differs from its launch alone.
//!
//! Run (GB10, < 1 GB of device memory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!     --example glm5next_batch_inv_bench

#[path = "../common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "../common/graph_timing.rs"]
mod timing;

mod bf16;
mod dense;

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let legs = std::env::var("GLM_BI_LEGS").unwrap_or_else(|_| "dense,bf16".into());
    let mut rng = fixture::Rng(0x6261_7463_6869_6e76);
    let pool = fixture::Pool::new(&g, &mut rng)?;
    for leg in legs.split(',') {
        match leg {
            "dense" => dense::run(&g, s, &pool, &mut rng)?,
            "bf16" => bf16::run(&g, s, &mut rng)?,
            other => anyhow::bail!("GLM_BI_LEGS: unknown leg {other:?}"),
        }
    }
    Ok(())
}
