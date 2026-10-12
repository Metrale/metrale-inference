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

use std::sync::OnceLock;

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

/// 2026-10-10: `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS`, read once: `None` when unset or `0`,
/// else the narrowest launch that takes the dequantize + cuBLASLt path. A value below
/// [`MIN_ROWS_FLOOR`], an unparsable one, or one without `--dense-quantization w4a16` is an
/// error.
pub fn min_rows() -> Result<Option<usize>> {
    static V: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_min_rows(
            std::env::var("METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS")
                .ok()
                .as_deref(),
            crate::glm5next_w4a16_dense::enabled(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-10: [`min_rows`]'s parse, pure: the value and whether the tier is on.
pub(crate) fn parse_min_rows(v: Option<&str>, tier_on: bool) -> Result<Option<usize>, String> {
    let r = match v {
        None | Some("0") => return Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(r) if r >= MIN_ROWS_FLOOR => r,
            _ => {
                return Err(format!(
                    "METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS={s:?}: expected 0 or a row count \
                     of at least {MIN_ROWS_FLOOR}"
                ));
            }
        },
    };
    if !tier_on {
        return Err(format!(
            "METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS={r} needs --dense-quantization w4a16"
        ));
    }
    Ok(Some(r))
}

/// 2026-10-10: Whether a launch of `m` rows takes this path under the lever value `min`.
pub fn takes(min: Option<usize>, m: usize) -> bool {
    min.is_some_and(|r| m >= r)
}

/// 2026-10-10: Reserve the shared BF16 scratch (`glm5next_dequant_scratch`) for an `[n, k]`
/// weight when the lever is on; a no-op when it is off. Called at load, as each weight registers.
pub fn reserve(gpu: &dyn GpuBackend, n: usize, k: usize) -> Result<()> {
    if min_rows()?.is_none() {
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
    ensure!(
        n * k <= scratch_elems && k.is_multiple_of(16),
        "W4A16 dequant GEMM: a [{n}, {k}] weight does not fit the {scratch_elems}-element scratch"
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
        .arg_ptr(scratch)
        .arg_u64(groups)
        .launch(stream)?;
    metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t_scaled(
        a.0,
        scratch.0,
        c.0,
        m as u32,
        n as u32,
        k as u32,
        w.weight_scale_2,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-10: Unset and `0` are off; a row count from 17 is on; 16 and below, text, and any
    /// value without the tier are errors.
    #[test]
    fn the_lever_parses_strictly() {
        assert_eq!(parse_min_rows(None, true), Ok(None));
        assert_eq!(parse_min_rows(Some("0"), false), Ok(None));
        assert_eq!(parse_min_rows(Some("17"), true), Ok(Some(17)));
        assert_eq!(parse_min_rows(Some("256"), true), Ok(Some(256)));
        for bad in ["16", "1", "x", "-3"] {
            assert!(parse_min_rows(Some(bad), true).is_err(), "{bad}");
        }
        assert!(parse_min_rows(Some("256"), false).is_err());
    }

    /// 2026-10-10: Only launches of at least the lever's rows take the path.
    #[test]
    fn only_launches_from_the_floor_take_the_path() {
        assert!(!takes(None, 4096));
        assert!(!takes(Some(256), 255));
        assert!(takes(Some(256), 256));
        assert!(takes(Some(256), 2048));
    }
}
