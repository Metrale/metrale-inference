// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The NVFP4-activation projection of a fixed `--activation-quantization`
//! ([`nvfp4_proj_mx`]): the W4A4 row quantizer and mx GEMV at any row count, in chunks of the
//! widest prepared entry, with no fallback to another format.
//!
//! Owner: model-layers ops (W4A4).
//! Invariants: launches only the prepared kernels, or returns an error before launching.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{W4a4Proj, last_quant, w4a4_route};
use crate::weight_map::QuantizedWeight;

/// 2026-09-30: `[m, k] x [n, k]^T` with NVFP4 activations for any `m`: per chunk of the widest
/// prepared entry, the row quantizer and then the mx GEMV. Every mx entry sums a row the same
/// way and the quantizer is per row, so a row's output bits do not depend on `m` or on the
/// other rows. Errors when the kernels were not prepared or the shape does not fit; it never
/// falls back to another format. For a fixed `--activation-quantization` only.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_proj_mx(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let p = W4a4Proj::prepared(gpu).ok_or_else(|| anyhow::anyhow!("w4a4: kernels not prepared"))?;
    let max_m = p.0.max_m;
    anyhow::ensure!(
        w4a4_route(1, n, k, max_m),
        "w4a4: shape n={n} k={k} outside the mx kernels' contract"
    );
    // 2026-09-30: The scratch may hold another input's quantization now.
    *last_quant().lock().unwrap_or_else(|p| p.into_inner()) = None;
    let act = p.scratch();
    let mut done = 0u32;
    while done < m {
        let rows = (m - done).min(max_m);
        let x = input.offset(done as usize * k as usize * 2);
        let y = output.offset(done as usize * n as usize * 2);
        p.quantize(gpu, x, act, rows, k, stream)?;
        p.gemv(gpu, act, weight, y, rows, n, k, stream)?;
        done += rows;
    }
    Ok(())
}
