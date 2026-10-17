// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS=<r>`: a launch of at least `r` rows (a
//! prefill sub-chunk) of a projection the FP8 dense registry serves (`glm5next_fp8_dense`:
//! `--dense-quantization fp8`, the FP8 part of `w4a16`, `--dsa-quantization w8a8`) dequantizes
//! the registered E4M3 weight with its per-row scale into the shared BF16 scratch
//! (`dequant_fp8_rowscale_to_bf16`) and runs one cuBLASLt BF16 GEMM over the BF16 activations,
//! instead of the W8A8 GEMV family, which re-reads the whole weight every 128 rows
//! (`w8a8_gemv_rowscale_mb16`: 3,600 launches of the 46 MB DSA `o_absorb` alone in one 32k
//! prefill on rank 0).
//!
//! Numerics: W8A16. The weight element becomes `bf16(E4M3 * row_scale)`: one BF16 rounding
//! (relative 2^-9) under E4M3's own 2^-4 step; the activations stay BF16 instead of being
//! quantized per token to E4M3 (so the products are MORE precise than the W8A8 family's), the
//! FP32 sum is cuBLASLt's order with one BF16 rounding. Against the W8A8 family the bits change
//! (activation quantization and summation order): opt-in and accuracy-gated, like
//! `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS`. A launch below `r` rows is unchanged.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The scratch is reserved at load (`reserve`, from `glm5next_fp8_dense::register`), never
//!   during a forward; a weight that does not fit it is an error.
//! - Unset or `0` launches exactly what the registry launched before this lever.
//! - A launch here is not row-invariant (cuBLASLt picks its order by M), so
//!   `METRALE_GLM_BATCH_INVARIANT` refuses the lever (`glm5next_batch_inv`).

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-10-12: The dequantization kernel's module (its file stem) and entry.
const DEQUANT_MODULE: &str = "dequant_nvfp4_bf16";
const DEQUANT_ENTRY: &str = "dequant_fp8_rowscale_to_bf16";
const BLOCK: u32 = 256;
/// 2026-10-12: The kernel's group: 16 consecutive elements per thread.
const GROUP: usize = 16;
/// 2026-10-12: This lever's name in errors.
const WHAT: &str = "FP8 dequant GEMM";

/// 2026-10-12: The narrowest value the lever takes: wider than the batched decode and verify
/// groups, which the W8A8 family serves bit-identically at every row count up to 16.
pub const MIN_ROWS_FLOOR: usize = 17;

/// 2026-10-12: Whether a launch of `m` rows takes this path under the lever value `min`.
pub fn takes(min: Option<usize>, m: usize) -> bool {
    min.is_some_and(|r| m >= r)
}

/// 2026-10-12: Reserve the shared BF16 scratch (`glm5next_dequant_scratch`) for an `[n, k]`
/// weight when the lever (`min_rows`, `DenseLevers::fp8_dequant_min_rows`) is on; a no-op when it
/// is off. Called at load, as each weight registers.
pub fn reserve(gpu: &dyn GpuBackend, min_rows: Option<usize>, n: usize, k: usize) -> Result<()> {
    if min_rows.is_none() {
        return Ok(());
    }
    crate::glm5next_dequant_scratch::reserve(gpu, n, k, WHAT)
}

/// 2026-10-12: `c[m, n] = a[m, k] @ dequant(w)^T` through the scratch reserved at load
/// ([`dequant_gemm`]).
#[allow(clippy::too_many_arguments)]
pub fn proj(
    gpu: &dyn GpuBackend,
    w: &Fp8Weight,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    // 2026-10-15: `METRALE_GLM_DEQUANT_PERSIST=all`: the copy kept at load holds the bytes the
    // dequantization would write into the scratch; the GEMM is the same call.
    if let Some(copy) = crate::glm5next_dequant_persist::copy_of(w.weight, n, k)? {
        return gemm(a, copy, c, m, n, k, stream);
    }
    let s = crate::glm5next_dequant_scratch::get(WHAT)?;
    dequant_gemm(gpu, w, s.ptr, s.elems, a, c, m, n, k, stream)
}

/// 2026-10-12: Dequantize `w` (`[n, k]` E4M3, one FP32 scale per row) into `scratch`
/// (`scratch_elems` BF16), then one cuBLASLt BF16 GEMM `c[m, n] = a[m, k] @ scratch^T`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_gemm(
    gpu: &dyn GpuBackend,
    w: &Fp8Weight,
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
    gemm(a, scratch, c, m, n, k, stream)
}

/// 2026-10-15: Dequantize `w` (`[n, k]` E4M3, one FP32 scale per row) into `dst` (`dst_elems`
/// BF16) on `stream`: [`dequant_gemm`]'s first step, also the load-time fill of
/// `glm5next_dequant_persist`'s copies.
pub fn dequantize(
    gpu: &dyn GpuBackend,
    w: &Fp8Weight,
    dst: DevicePtr,
    dst_elems: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        w.scale_format == WeightQuantFormat::Fp8PerRow && (w.n as usize, w.k as usize) == (n, k),
        "{WHAT}: the weight is [{}, {}] {:?}, launched as [{n}, {k}] per-row",
        w.n,
        w.k,
        w.scale_format
    );
    ensure!(
        n * k <= dst_elems && k.is_multiple_of(GROUP),
        "{WHAT}: a [{n}, {k}] weight does not fit the {dst_elems}-element scratch"
    );
    let kern = gpu
        .op_cache()
        .kernel(gpu, DEQUANT_MODULE, DEQUANT_ENTRY)
        .with_context(|| format!("{WHAT}: {DEQUANT_MODULE}::{DEQUANT_ENTRY}"))?;
    let groups = (n * k / GROUP) as u64;
    KernelLaunch::new(gpu, kern)
        .grid([groups.div_ceil(u64::from(BLOCK)) as u32, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(w.weight)
        .arg_ptr(w.row_scale)
        .arg_ptr(dst)
        .arg_u64(groups)
        .arg_u32(k as u32)
        .launch(stream)
}

/// 2026-10-15: [`dequant_gemm`]'s second step: one cuBLASLt BF16 GEMM
/// `c[m, n] = a[m, k] @ w_bf16[n, k]^T`.
fn gemm(
    a: DevicePtr,
    w_bf16: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t_scaled(
        a.0, w_bf16.0, c.0, m as u32, n as u32, k as u32, 1.0, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-12: Only launches of at least the lever's rows take the path.
    #[test]
    fn only_launches_from_the_floor_take_the_path() {
        assert!(!takes(None, 8208));
        assert!(!takes(Some(256), 255));
        assert!(takes(Some(256), 256));
        assert!(takes(Some(256), 8208));
    }

    /// 2026-10-12: A weight of another shape, another scale layout, or one wider than the scratch
    /// is refused before any launch.
    #[test]
    fn a_mismatched_weight_or_scratch_is_refused_before_launching() {
        use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
        let gpu = MockGpuBackend::new();
        let w = Fp8Weight {
            weight: DevicePtr(0x100),
            row_scale: DevicePtr(0x200),
            n: 4096,
            k: 11264,
            scale_format: WeightQuantFormat::Fp8PerRow,
        };
        let (s, a, c) = (DevicePtr(0x300), DevicePtr(0x400), DevicePtr(0x500));
        let elems = 4096 * 11264;
        let e = dequant_gemm(&gpu, &w, s, elems, a, c, 256, 4096, 1536, 0).unwrap_err();
        assert!(e.to_string().contains("launched as [4096, 1536]"), "{e}");
        let e = dequant_gemm(&gpu, &w, s, elems - 1, a, c, 256, 4096, 11264, 0).unwrap_err();
        assert!(e.to_string().contains("does not fit"), "{e}");
        let blk = Fp8Weight {
            scale_format: WeightQuantFormat::Fp8BlockScaled,
            ..w
        };
        let e = dequant_gemm(&gpu, &blk, s, elems, a, c, 256, 4096, 11264, 0).unwrap_err();
        assert!(e.to_string().contains("per-row"), "{e}");
        assert!(gpu.launches_snapshot().is_empty(), "nothing was launched");
    }
}
