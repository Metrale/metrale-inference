// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS=<r>`: under `--dense-quantization
//! w4a16`, a launch of at least `r` rows (a prefill sub-chunk) dequantizes the registered NVFP4
//! weight into one shared BF16 scratch (`dequant_nvfp4_to_bf16_g16`: E2M1 * E4M3, exact in
//! BF16) and runs one cuBLASLt BF16 GEMM with the per-tensor scale as alpha, instead of one
//! 64-row tile launch per 64 rows, each of which re-reads and re-decodes the whole weight
//! (`w4a16_tc_rows`, a decode kernel).
//!
//! Numerics: the products are the row-tile kernel's (the activation times the exact BF16
//! E2M1 * E4M3, FP32 accumulation, the scale applied once to the FP32 sum, one BF16 rounding);
//! only the FP32 summation order differs, so it is opt-in and accuracy-gated. A launch below
//! `r` rows is unchanged.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The scratch is allocated at load (`reserve`, from `glm5next_w4a16_dense::register`), never
//!   during a forward; a launch whose weight does not fit it is an error. 2026-10-12: it is the
//!   one `glm5next_dequant_scratch` shares with `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS`.
//! - Unset or `0` launches exactly what the tier launched before this lever.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-10: The dequantization kernel's module (its file stem) and entry.
const DEQUANT_MODULE: &str = "dequant_nvfp4_bf16";
const DEQUANT_ENTRY: &str = "dequant_nvfp4_to_bf16_g16";
const BLOCK: u32 = 256;
/// 2026-10-12: This lever's name in the shared scratch's errors.
const WHAT: &str = "W4A16 dequant GEMM";

/// 2026-10-10: The narrowest value the lever takes: wider than the batched decode and verify
/// groups the row tiles serve bit-identically at every row count up to 16.
pub const MIN_ROWS_FLOOR: usize = 17;

/// 2026-10-10: Whether a launch of `m` rows takes this path under the lever value `min`.
pub fn takes(min: Option<usize>, m: usize) -> bool {
    min.is_some_and(|r| m >= r)
}

/// 2026-10-10: Reserve the shared BF16 scratch (`glm5next_dequant_scratch`) for an `[n, k]`
/// weight when the lever (`min_rows`, `DenseLevers::dequant_min_rows`) is on; a no-op when it is
/// off. Called at load, as each weight registers.
pub fn reserve(gpu: &dyn GpuBackend, min_rows: Option<usize>, n: usize, k: usize) -> Result<()> {
    if min_rows.is_none() {
        return Ok(());
    }
    crate::glm5next_dequant_scratch::reserve(gpu, n, k, WHAT)
}

/// 2026-10-10: `c[m, n] = s2 * (a[m, k] @ dequant(w)^T)` through the scratch reserved at load
/// ([`dequant_gemm`]).
#[allow(clippy::too_many_arguments)]
pub fn proj(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    // 2026-10-15: `METRALE_GLM_DEQUANT_PERSIST`: the copy kept at load holds the bytes the
    // dequantization would write into the scratch; the GEMM is the same call.
    if let Some(copy) = crate::glm5next_dequant_persist::copy_of(w.weight, n, k)? {
        return gemm(a, copy, c, m, n, k, w.weight_scale_2, stream);
    }
    let s = crate::glm5next_dequant_scratch::get(WHAT)?;
    dequant_gemm(gpu, w, s.ptr, s.elems, a, c, m, n, k, stream)
}

/// 2026-10-10: Dequantize `w` (`[n, k]` NVFP4) into `scratch` (`scratch_elems` BF16), then one
/// cuBLASLt BF16 GEMM `c[m, n] = w.weight_scale_2 * (a[m, k] @ scratch^T)`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_gemm(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    scratch: DevicePtr,
    scratch_elems: usize,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    dequantize(gpu, w, scratch, scratch_elems, n, k, stream)?;
    gemm(a, scratch, c, m, n, k, w.weight_scale_2, stream)
}

/// 2026-10-15: Dequantize `w` (`[n, k]` NVFP4: packed codes and group scales) into `dst`
/// (`dst_elems` BF16) on `stream`: [`dequant_gemm`]'s first step, also the load-time fill of
/// `glm5next_dequant_persist`'s copies.
pub fn dequantize(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    dst: DevicePtr,
    dst_elems: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        n * k <= dst_elems && k.is_multiple_of(16),
        "W4A16 dequant GEMM: a [{n}, {k}] weight does not fit the {dst_elems}-element scratch"
    );
    let kern = gpu
        .op_cache()
        .kernel(gpu, DEQUANT_MODULE, DEQUANT_ENTRY)
        .with_context(|| format!("W4A16 dequant GEMM: {DEQUANT_MODULE}::{DEQUANT_ENTRY}"))?;
    let groups = (n * k / 16) as u64;
    KernelLaunch::new(gpu, kern)
        .grid([groups.div_ceil(u64::from(BLOCK)) as u32, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_ptr(dst)
        .arg_u64(groups)
        .launch(stream)
}

/// 2026-10-15: [`dequant_gemm`]'s second step: one cuBLASLt BF16 GEMM
/// `c[m, n] = alpha * (a[m, k] @ w_bf16[n, k]^T)`.
#[allow(clippy::too_many_arguments)]
fn gemm(
    a: DevicePtr,
    w_bf16: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t_scaled(
        a.0, w_bf16.0, c.0, m as u32, n as u32, k as u32, alpha, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-10: Only launches of at least the lever's rows take the path.
    #[test]
    fn only_launches_from_the_floor_take_the_path() {
        assert!(!takes(None, 4096));
        assert!(!takes(Some(256), 255));
        assert!(takes(Some(256), 256));
        assert!(takes(Some(256), 2048));
    }
}
