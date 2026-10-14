// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_W8A8_GEMM_MIN_ROWS=<r>`: a launch of at least `r` rows (a prefill
//! sub-chunk) of a projection the FP8 dense registry serves (`glm5next_fp8_dense`, including the
//! prompt-pass copies of a phase split) runs the W8A8 tensor-core GEMM (`ops::w8a8_proj_wide`,
//! `w8a8_gemm.cu`): one activation quantization and one GEMM per chunk of up to [`WIDE_ROWS`]
//! rows, reading the weight once per token tile instead of once per 128 rows as the GEMV family
//! (`w8a8_gemv_rowscale_mb16`) does.
//!
//! Numerics: BIT-IDENTICAL to the W8A8 GEMV family. The kernel walks K in the engine's class
//! order, so a row's bytes depend on neither the row count, the chunk nor the tile; prefill rows
//! are the bytes the registry writes at decode widths, and the lever is compatible with
//! `METRALE_GLM_BATCH_INVARIANT`. Against `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS` (W8A16 on a
//! dequantized copy) the bits change; this lever wins when both take a launch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset or `0` launches exactly what the registry launched before this lever.
//! - 2026-10-14: A projection skips its activation quantization only when its caller vouches for
//!   the input (`glm5next_fp8_dense::StableInput`) and the scratch holds that input's quantization
//!   at the same rows, K and stream from this guard's lifetime ([`reuses`]); the bytes are the
//!   ones a fresh quantization writes, so the outputs are unchanged.
//! - The activation scratch is reserved at load ([`reserve`], from `glm5next_fp8_dense::prepare`),
//!   never during a forward; a K wider than it is an error.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{
    W8a8GemmKernels, W8a8GemmTile, W8a8Kernels, W8a8Scratch, W8a8Weight, w8a8_gemm, w8a8_proj_wide,
};

/// 2026-10-14: The lever's environment variable.
pub const MIN_ROWS_ENV: &str = "METRALE_GLM_W8A8_GEMM_MIN_ROWS";

/// 2026-10-14: The narrowest value the lever takes: above the batched decode and verify groups,
/// which the GEMV family serves (the bytes are the same either way).
pub const MIN_ROWS_FLOOR: usize = 17;

/// 2026-10-14: The most rows per launch: the activation scratch holds `prefill_rows()` quantized
/// rows up to this (92 MB at 8208 rows and K 11264), so a whole prefill sub-chunk is one launch and
/// one quantization the projections of one input can share. Chunking is by whole rows, so it moves
/// no bits.
pub const WIDE_ROWS: usize = 16384;

/// 2026-10-14: What the scratch holds: `(input, rows, k, stream)` of its last quantization.
type Held = (u64, usize, usize, u64);

/// 2026-10-14: The tile every launch uses (gate (a) timing on GB10).
pub const TILE: W8a8GemmTile = W8a8GemmTile::W128T64S3;

struct Wide {
    gemm: W8a8GemmKernels,
    scratch: W8a8Scratch,
    rows: usize,
    max_k: usize,
    /// 2026-10-14: Set only for a vouched-for input quantized whole into the scratch.
    held: Option<Held>,
}

fn state() -> &'static Mutex<Option<Wide>> {
    static S: OnceLock<Mutex<Option<Wide>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// 2026-10-14: The lever, read once: `None` when unset or `0`, else the narrowest launch that
/// takes the GEMM. A value below [`MIN_ROWS_FLOOR`], an unparsable one, or one while the FP8
/// registry is not in use is an error.
pub fn min_rows() -> Result<Option<usize>> {
    static V: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_min_rows(
            std::env::var(MIN_ROWS_ENV).ok().as_deref(),
            crate::glm5next_fp8_dense::enabled(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`min_rows`]'s parse, pure: the value and whether the FP8 registry is in use.
pub(crate) fn parse_min_rows(v: Option<&str>, registry_on: bool) -> Result<Option<usize>, String> {
    let r = match v {
        None | Some("0") => return Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(r) if r >= MIN_ROWS_FLOOR => r,
            _ => {
                return Err(format!(
                    "{MIN_ROWS_ENV}={s:?}: expected 0 or a row count of at least {MIN_ROWS_FLOOR}"
                ));
            }
        },
    };
    if !registry_on {
        return Err(format!(
            "{MIN_ROWS_ENV}={r} needs an FP8 dense tier (--dense-quantization fp8, w4a16 or \
             decode=w4a16,prefill=fp8, or --dsa-quantization w8a8)"
        ));
    }
    Ok(Some(r))
}

/// 2026-10-14: Whether a launch of `m` rows takes the GEMM under the lever value `min`.
pub fn takes(min: Option<usize>, m: usize) -> bool {
    min.is_some_and(|r| m >= r)
}

/// 2026-10-14: Resolve the GEMM and allocate the activation scratch for K up to `max_k` when the
/// lever is on; a no-op when it is off or the scratch is already that wide. Called at load. Errors
/// when the GEMM is not in this target's kernels: the operator asked for it.
pub fn reserve(gpu: &dyn GpuBackend, max_k: usize) -> Result<()> {
    if min_rows()?.is_none() {
        return Ok(());
    }
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    if g.as_ref().is_some_and(|w| w.max_k >= max_k) {
        return Ok(());
    }
    let gemm = W8a8GemmKernels::load(gpu);
    ensure!(
        gemm.resolved(TILE),
        "{MIN_ROWS_ENV}: {} is not in this target's kernels",
        TILE.entry()
    );
    let rows = crate::glm5next_layer::prefill_rows().min(WIDE_ROWS);
    let scratch = W8a8Scratch::alloc_rows(gpu, rows, u32::try_from(max_k)?)?;
    if let Some(old) = g.replace(Wide {
        gemm,
        scratch,
        rows,
        max_k,
        held: None,
    }) {
        // 2026-10-14: Load time: nothing has launched on the narrower scratch.
        gpu.free(old.scratch.q)?;
        gpu.free(old.scratch.scale)?;
    }
    Ok(())
}

/// 2026-10-14: Whether a projection of `key` may read the scratch as it is: the caller vouches for
/// the input (`stable`), the scratch holds exactly that quantization, and it fits in one launch.
pub(crate) fn reuses(stable: bool, held: Option<Held>, key: Held, rows: usize) -> bool {
    stable && key.1 <= rows && held == Some(key)
}

/// 2026-10-14: Forget what the scratch holds (a `StableInput` guard ended: its input may change).
pub fn forget() {
    if let Some(s) = state().lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        s.held = None;
    }
}

/// 2026-10-14: `c[m, n] = a[m, k] @ w[n, k]^T` (contiguous BF16 rows) on the GEMM, in launches of
/// at most the reserved rows. `stable`: the caller vouches that `a` has not changed since the
/// scratch's quantization of it, if there is one ([`reuses`]).
#[allow(clippy::too_many_arguments)]
pub fn proj(
    gpu: &dyn GpuBackend,
    k8: &W8a8Kernels,
    w: &W8a8Weight,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
    stable: bool,
) -> Result<()> {
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    let s = g
        .as_mut()
        .with_context(|| format!("{MIN_ROWS_ENV}: launched before reserve"))?;
    ensure!(
        k <= s.max_k,
        "{MIN_ROWS_ENV}: K={k} is wider than the reserved {}",
        s.max_k
    );
    let key = (a.0, m, k, stream);
    if reuses(stable, s.held, key, s.rows) {
        return w8a8_gemm(gpu, &s.gemm, TILE, w, &s.scratch, m, c, n as u32, stream);
    }
    // 2026-10-14: The scratch is overwritten below; a failed launch must not leave it vouched for.
    s.held = None;
    let mut done = 0;
    while done < m {
        let rows = (m - done).min(s.rows);
        w8a8_proj_wide(
            gpu,
            k8,
            &s.gemm,
            TILE,
            w,
            a.offset(done * k * 2),
            k as u32,
            rows,
            c.offset(done * n * 2),
            n as u32,
            &s.scratch,
            stream,
        )?;
        done += rows;
    }
    s.held = (stable && m <= s.rows).then_some(key);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

    /// 2026-10-14: Through `proj` on a recording mock: a vouched-for input is quantized once and
    /// then read by each projection (one launch instead of two); another input, an unvouched
    /// call, `forget`, or a width that needs two launches quantizes again. The scratch state is
    /// installed directly: the lever is process-wide and off in tests.
    #[test]
    fn a_vouched_input_is_quantized_once_until_the_scratch_changes_hands() {
        let gpu = MockGpuBackend::new();
        let k8 = W8a8Kernels::load(&gpu);
        *state().lock().unwrap() = Some(Wide {
            gemm: W8a8GemmKernels::load(&gpu),
            scratch: W8a8Scratch::alloc_rows(&gpu, 64, 4096).unwrap(),
            rows: 64,
            max_k: 4096,
            held: None,
        });
        let seg = |p: u64| Fp8Weight {
            weight: DevicePtr(p),
            row_scale: DevicePtr(p + 0x10_0000),
            n: 32,
            k: 4096,
            scale_format: WeightQuantFormat::Fp8PerRow,
        };
        let (w1, w2) = (
            W8a8Weight::new(&[seg(0x5100_0000)]).unwrap(),
            W8a8Weight::new(&[seg(0x5200_0000)]).unwrap(),
        );
        let (h, other, c) = (
            DevicePtr(0x6100_0000),
            DevicePtr(0x6200_0000),
            DevicePtr(0x6300_0000),
        );
        let launches = |w: &W8a8Weight, a: DevicePtr, m: usize, stable: bool| {
            let n = gpu.launches_snapshot().len();
            proj(&gpu, &k8, w, a, c, m, 32, 4096, 9, stable).unwrap();
            gpu.launches_snapshot().len() - n
        };
        assert_eq!(launches(&w1, h, 48, true), 2, "first use quantizes");
        assert_eq!(
            launches(&w2, h, 48, true),
            1,
            "the next projection reads the scratch"
        );
        assert_eq!(launches(&w1, h, 48, true), 1);
        assert_eq!(launches(&w1, other, 48, true), 2, "another input quantizes");
        assert_eq!(
            launches(&w2, h, 48, true),
            2,
            "and the first is quantized again"
        );
        assert_eq!(launches(&w2, h, 47, true), 2, "another row count quantizes");
        assert_eq!(
            launches(&w1, h, 47, false),
            2,
            "an unvouched call quantizes"
        );
        assert_eq!(launches(&w1, h, 47, true), 2, "and leaves nothing to reuse");
        forget();
        assert_eq!(launches(&w1, h, 47, true), 2, "forget drops the record");
        assert_eq!(
            launches(&w1, h, 100, true),
            4,
            "two launches of 64 + 36 rows"
        );
        assert_eq!(
            launches(&w1, h, 100, true),
            4,
            "a chunked width is never reused"
        );
        *state().lock().unwrap() = None;
    }

    /// 2026-10-14: Unset and `0` are off; a row count from 17 is on; 16 and below, text, and any
    /// value while no FP8 tier is on are errors.
    #[test]
    fn the_lever_parses_strictly() {
        assert_eq!(parse_min_rows(None, true), Ok(None));
        assert_eq!(parse_min_rows(Some("0"), false), Ok(None));
        assert_eq!(parse_min_rows(Some("17"), true), Ok(Some(17)));
        assert_eq!(parse_min_rows(Some("1024"), true), Ok(Some(1024)));
        for bad in ["16", "1", "x", "-3", "", " 64"] {
            assert!(parse_min_rows(Some(bad), true).is_err(), "{bad:?}");
        }
        let e = parse_min_rows(Some("64"), false).unwrap_err();
        assert!(e.contains("needs an FP8 dense tier"), "{e}");
    }

    /// 2026-10-14: The scratch is read as it is only for the vouched-for input it holds at the
    /// same rows, K and stream, in one launch; any other key, an unvouched caller, nothing held,
    /// or a chunked width quantizes again.
    #[test]
    fn a_held_quantization_is_reused_only_for_its_exact_vouched_key() {
        let key = (0x6100_0000, 8208, 4096, 7);
        assert!(reuses(true, Some(key), key, 8208));
        assert!(!reuses(false, Some(key), key, 8208), "unvouched");
        assert!(!reuses(true, None, key, 8208), "nothing held");
        assert!(
            !reuses(true, Some(key), key, 4096),
            "chunked: the scratch holds the last chunk"
        );
        for other in [
            (0x6200_0000, 8208, 4096, 7),
            (0x6100_0000, 8207, 4096, 7),
            (0x6100_0000, 8208, 2816, 7),
            (0x6100_0000, 8208, 4096, 8),
        ] {
            assert!(!reuses(true, Some(other), key, 8208), "{other:?}");
        }
    }

    /// 2026-10-14: Only launches from the lever's rows take the GEMM.
    #[test]
    fn only_launches_from_the_floor_take_the_gemm() {
        assert!(!takes(None, 8208));
        assert!(!takes(Some(64), 63));
        assert!(takes(Some(64), 64));
        assert!(takes(Some(64), 8208));
    }
}
