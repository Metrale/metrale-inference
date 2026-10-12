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

use std::sync::OnceLock;

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

/// 2026-10-12: `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS`, read once: `None` when unset or `0`,
/// else the narrowest launch that takes the dequantize + cuBLASLt path. A value below
/// [`MIN_ROWS_FLOOR`], an unparsable one, or one while the FP8 registry is not in use is an
/// error.
pub fn min_rows() -> Result<Option<usize>> {
    static V: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_min_rows(
            std::env::var("METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS")
                .ok()
                .as_deref(),
            crate::glm5next_fp8_dense::enabled(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-12: [`min_rows`]'s parse, pure: the value and whether the FP8 registry is in use.
pub(crate) fn parse_min_rows(v: Option<&str>, registry_on: bool) -> Result<Option<usize>, String> {
    let r = match v {
        None | Some("0") => return Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(r) if r >= MIN_ROWS_FLOOR => r,
            _ => {
                return Err(format!(
                    "METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS={s:?}: expected 0 or a row count of \
                     at least {MIN_ROWS_FLOOR}"
                ));
            }
        },
    };
    if !registry_on {
        return Err(format!(
            "METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS={r} needs --dense-quantization fp8 or w4a16, \
             or --dsa-quantization w8a8"
        ));
    }
    Ok(Some(r))
}

/// 2026-10-12: Whether a launch of `m` rows takes this path under the lever value `min`.
pub fn takes(min: Option<usize>, m: usize) -> bool {
    min.is_some_and(|r| m >= r)
}

/// 2026-10-12: Reserve the shared BF16 scratch (`glm5next_dequant_scratch`) for an `[n, k]`
/// weight when the lever is on; a no-op when it is off. Called at load, as each weight registers.
pub fn reserve(gpu: &dyn GpuBackend, n: usize, k: usize) -> Result<()> {
    if min_rows()?.is_none() {
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
    ensure!(
        w.scale_format == WeightQuantFormat::Fp8PerRow && (w.n as usize, w.k as usize) == (n, k),
        "{WHAT}: the weight is [{}, {}] {:?}, launched as [{n}, {k}] per-row",
        w.n,
        w.k,
        w.scale_format
    );
    ensure!(
        n * k <= scratch_elems && k.is_multiple_of(GROUP),
        "{WHAT}: a [{n}, {k}] weight does not fit the {scratch_elems}-element scratch"
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
        .arg_ptr(scratch)
        .arg_u64(groups)
        .arg_u32(k as u32)
        .launch(stream)?;
    metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t_scaled(
        a.0, scratch.0, c.0, m as u32, n as u32, k as u32, 1.0, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-12: Unset and `0` are off; a row count from 17 is on; 16 and below, text, and any
    /// value while the registry is not in use are errors.
    #[test]
    fn the_lever_parses_strictly() {
        assert_eq!(parse_min_rows(None, true), Ok(None));
        assert_eq!(parse_min_rows(Some("0"), false), Ok(None));
        assert_eq!(parse_min_rows(Some("17"), true), Ok(Some(17)));
        assert_eq!(parse_min_rows(Some("256"), true), Ok(Some(256)));
        for bad in ["16", "1", "x", "-3", ""] {
            assert!(parse_min_rows(Some(bad), true).is_err(), "{bad:?}");
        }
        let e = parse_min_rows(Some("256"), false).unwrap_err();
        assert!(e.contains("needs --dense-quantization"), "{e}");
    }

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
