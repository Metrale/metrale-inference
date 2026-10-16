// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The BF16 projection dispatch every GLM-5.3 block shares (`glm_mm`), with the
//! register-resident batched GEMV for 9..=16 rows (`dense_gemv_bf16_batchm_wide`).
//! 2026-10-16: and, under `METRALE_GLM_BATCHM_NARROW=1`, for 2..=8 rows
//! (`dense_gemv_bf16_batchm_reg8`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Only `glm_mm` launches the register entries, only in their bands
//!   (`ops::DenseGemvBatchmPair::for_rows`), with grid y = 1, and only when the caller passes
//!   their handles; every other M takes the dispatch the DSA and MLP `gemm` helpers took before
//!   (cuBLASLt above 16 rows, else `ops::dense_mm_bf16`), which other models share and which
//!   is unchanged.
//! - 2026-10-16: Under `METRALE_DENSE_BATCHM_32=1` exactly 17..=32 rows run as 16-row launches
//!   ([`glm_mm_rows16`]) instead of cuBLASLt, and only at a `DenseCallSite::Decode` or
//!   `DenseCallSite::Verify` launch (`ops::dense_batchm_32_at`): a prefill launch keeps its
//!   route at every row count. Off, nothing changes.
//! - A register entry's row has the bits of the runtime-M batched GEMV's row
//!   (`kernels/gb10/common/dense_gemv_bf16_batchm.cu`), which has `dense_gemv_bf16`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;

/// 2026-10-09: A batched-GEMV set: `narrow` is the runtime-M entry `ops::dense_mm_bf16`
/// launches at 2..=16 rows; `wide` the register-resident entry for 9..=16 rows, or
/// `KernelHandle(0)` to keep `narrow` there. 2026-10-11: `out_bytes`, the set's output element
/// size (2 for the BF16-out entries, 4 for the FP32-out ones), sizes a row of `C`. 2026-10-16:
/// `reg8`, the register-resident entry for 2..=8 rows ([`resolve_reg8`]: `KernelHandle(0)`
/// unless `METRALE_GLM_BATCHM_NARROW=1`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Batchm {
    pub(crate) narrow: KernelHandle,
    pub(crate) reg8: KernelHandle,
    pub(crate) wide: KernelHandle,
    pub(crate) out_bytes: usize,
}

impl From<KernelHandle> for Batchm {
    /// 2026-10-11: A BF16-out narrow entry alone.
    fn from(narrow: KernelHandle) -> Self {
        Self {
            narrow,
            reg8: KernelHandle(0),
            wide: KernelHandle(0),
            out_bytes: 2,
        }
    }
}

/// 2026-10-09: `METRALE_GLM_BATCHM_WIDE=0` keeps the runtime-M batched GEMV at 9..=16 rows
/// (an A/B of the speed; the bits are the same). On otherwise. Read once.
fn wide_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_BATCHM_WIDE").as_deref() != Ok("0"))
}

/// 2026-10-16: `METRALE_GLM_BATCHM_NARROW=1` sends 2..=8 rows to the register-resident reg8
/// entry (router logits, DSA indexer, the other BF16 projections at decode widths): same grid,
/// arguments and per-row bits as the runtime-M entry, whose accumulators live in local memory.
/// Off by default. Only `1` turns it on.
fn narrow_lever(value: Option<&str>) -> bool {
    value == Some("1")
}

/// 2026-10-16: The reg8 entry of one output type (`fp32_out`) for a kernels struct's
/// `resolve`: `KernelHandle(0)` with `METRALE_GLM_BATCHM_NARROW` off, or where the target lacks
/// the entry (both keep the runtime-M entry at 2..=8 rows). Logs the readback for an arm once.
pub(crate) fn resolve_reg8(gpu: &dyn GpuBackend, fp32_out: bool) -> KernelHandle {
    if !narrow_lever(std::env::var("METRALE_GLM_BATCHM_NARROW").ok().as_deref()) {
        return KernelHandle(0);
    }
    let func = if fp32_out {
        "dense_gemv_bf16_batchm_reg8_fp32out"
    } else {
        "dense_gemv_bf16_batchm_reg8"
    };
    let h = metrale_model_layers::layers::try_kernel(gpu, "dense_gemv_bf16_batchm", func);
    static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
    ONCE[usize::from(fp32_out)].call_once(|| {
        tracing::info!(
            "glm_mm: METRALE_GLM_BATCHM_NARROW=1 -- 2..=8-row BF16 projections on the \
             register-resident batched GEMV ({func} loaded={})",
            h.0 != 0
        );
    });
    h
}

impl Batchm {
    /// 2026-10-16: The entries `glm_mm` picks among, with the wide lever applied.
    fn entries(self, wide_on: bool) -> ops::DenseGemvBatchmPair {
        ops::DenseGemvBatchmPair {
            narrow: self.narrow,
            reg8: self.reg8,
            wide: if wide_on { self.wide } else { KernelHandle(0) },
        }
    }

    /// 2026-10-16: The register entry `glm_mm` launches for `m` rows, if any.
    fn register_entry(self, m: usize, wide_on: bool) -> Option<KernelHandle> {
        let m = u32::try_from(m).ok()?;
        let e = self.entries(wide_on);
        let h = e.for_rows(m);
        (h.0 != e.narrow.0).then_some(h)
    }
}

/// 2026-10-16: Whether `METRALE_DENSE_BATCHM_32` (`on`) sends an `m`-row `glm_mm` launched at
/// `site` to [`glm_mm_rows16`]: exactly 17..=32 rows, and only at the batched decode or verify
/// (`ops::dense_batchm_32_at`), never from the row count alone.
fn rows32_takes(on: bool, site: ops::DenseCallSite, m: usize) -> bool {
    ops::dense_batchm_32_at(on, site)
        && u32::try_from(m).is_ok_and(|m| ops::dense_batchm_band(0, true, m))
}

/// 2026-10-16: The site of a single-sequence k-row forward (`decode_k`) from its caller's
/// `is_prefill`: `Prefill` for a prefill sub-chunk, else `General` (the single-sequence verify
/// is not the batched body, so it keeps its route too).
pub(crate) fn single_seq_site(is_prefill: bool) -> ops::DenseCallSite {
    if is_prefill {
        ops::DenseCallSite::Prefill
    } else {
        ops::DenseCallSite::General
    }
}

/// 2026-10-16: The arm readback for `METRALE_DENSE_BATCHM_32`, once per process.
fn log_rows32(m: usize, n: usize, kk: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "glm_mm: METRALE_DENSE_BATCHM_32=1 -- 17..=32-row BF16 projections as 16-row \
             batched-GEMV launches (first m={m} n={n} k={kk})"
        );
    });
}

/// 2026-10-11: [`glm_mm`] in launches of at most `DENSE_GEMV_BATCHM_MAX_M` rows, each on the
/// GEMV tiers (one row: the M = 1 GEMV; 2..=16: the batched GEMV), whose rows have the M = 1
/// GEMV's bits; `C` rows are `n` elements of `batchm.out_bytes`. Errors when the batched handle
/// is missing (the tile GEMM would take the rows, in another order).
#[allow(clippy::too_many_arguments)]
fn glm_mm_rows16(
    gpu: &dyn GpuBackend,
    gemm: KernelHandle,
    gemv: KernelHandle,
    batchm: Batchm,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    site: ops::DenseCallSite,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        batchm.narrow.0 != 0 && gemv.0 != 0,
        "METRALE_GLM_BATCH_INVARIANT=1 or METRALE_DENSE_BATCHM_32=1: a {m}-row projection needs the \
         GEMV tiers"
    );
    let step = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    let c_elem = batchm.out_bytes;
    let mut r0 = 0;
    while r0 < m {
        let rows = step.min(m - r0);
        let (ar, cr) = (a.offset(r0 * kk * 2), c.offset(r0 * n * c_elem));
        glm_mm(
            gpu, gemm, gemv, batchm, ar, b, cr, rows, n, kk, site, stream,
        )?;
        r0 += rows;
    }
    Ok(())
}

/// 2026-10-09: `C[M, N] = A[M, K] @ B[N, K]^T` with BF16 inputs (the output element is the
/// kernels'): the W4A16 or FP8 dense path for a registered weight, then cuBLASLt above `DENSE_GEMV_BATCHM_MAX_M` rows when `cublas_wide_proj` is on (BF16
/// out), the register batched GEMV in its band (`Batchm::register_entry`), else
/// `ops::dense_mm_bf16` (GEMV at one row, `batchm.narrow` at 2..=16, the tile GEMM otherwise).
/// 2026-10-16: `site` is the caller's pass; only `Decode` and `Verify` may take the 17..=32-row
/// band of `METRALE_DENSE_BATCHM_32` ([`rows32_takes`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_mm(
    gpu: &dyn GpuBackend,
    gemm: KernelHandle,
    gemv: KernelHandle,
    batchm: Batchm,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    site: ops::DenseCallSite,
    stream: u64,
) -> Result<()> {
    // 2026-10-09: `--dense-quantization fp8` serves registered weights W8A8; only BF16-output
    // projections are registered (each block's `dense_projections`). Under `w4a16` the W4A16
    // registry answers first, then the FP8 one for the rest of the set.
    if crate::glm5next_w4a16_dense::proj(gpu, b, a, c, m, n, kk, stream)? {
        return Ok(());
    }
    if crate::glm5next_fp8_dense::proj(gpu, b, a, c, m, n, kk, stream)? {
        return Ok(());
    }
    // 2026-10-11: Under `METRALE_GLM_BATCH_INVARIANT`, wider launches run as 16-row launches of
    // the row-invariant batched GEMV ([`glm_mm_rows16`]), never cuBLASLt or the tile GEMM.
    let wide_rows = m > ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    if wide_rows && crate::glm5next_batch_inv::enabled()? {
        return glm_mm_rows16(gpu, gemm, gemv, batchm, a, b, c, m, n, kk, site, stream);
    }
    // 2026-10-16: `METRALE_DENSE_BATCHM_32=1`: 17..=32 rows as the same 16-row launches (an MTP
    // verify of 24 or 32 rows keeps the rows' 1..=16-row bits), ahead of cuBLASLt.
    if rows32_takes(ops::dense_batchm_32(), site, m) {
        log_rows32(m, n, kk);
        return glm_mm_rows16(gpu, gemm, gemv, batchm, a, b, c, m, n, kk, site, stream);
    }
    if wide_rows && crate::glm5next_layer::cublas_wide_proj() {
        return ops::cublas_bf16_proj_dense(a, b, c, m as u32, n as u32, kk as u32, stream);
    }
    if let Some(reg) = batchm.register_entry(m, wide_enabled()) {
        return KernelLaunch::new(gpu, reg)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .arg_u32(n as u32)
            .launch(stream);
    }
    ops::dense_mm_bf16(
        gpu,
        &ops::DenseMmKernels {
            gemm,
            gemv,
            batchm: batchm.narrow,
        },
        a,
        b,
        c,
        m,
        n,
        kk,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

    use super::*;

    fn ptr(a: &MockArg) -> DevicePtr {
        match a {
            MockArg::Buffer(p) => *p,
            MockArg::Bytes(_) => panic!("expected a buffer"),
        }
    }

    fn u32_arg(a: &MockArg) -> u32 {
        match a {
            MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
            MockArg::Buffer(p) => panic!("expected a u32, got {p}"),
        }
    }

    /// 2026-10-11: 40 FP32-out rows run as 16 + 16 + 8 rows on the narrow batched GEMV (no
    /// wide entry given), each launch's A and C advanced by whole rows (C at 4 bytes per
    /// element); a missing batched entry is refused rather than left to the tile GEMM.
    #[test]
    fn rows16_splits_into_whole_row_launches_of_at_most_16() {
        let gpu = MockGpuBackend::new();
        let (gemm, gemv, narrow) = (KernelHandle(11), KernelHandle(12), KernelHandle(13));
        let b = Batchm {
            narrow,
            reg8: KernelHandle(0),
            wide: KernelHandle(0),
            out_bytes: 4,
        };
        let (a, w, c) = (
            DevicePtr(0x1000_0000),
            DevicePtr(0x2000_0000),
            DevicePtr(0x3000_0000),
        );
        let (n, kk) = (288usize, 4096usize);
        glm_mm_rows16(
            &gpu,
            gemm,
            gemv,
            b,
            a,
            w,
            c,
            40,
            n,
            kk,
            ops::DenseCallSite::General,
            7,
        )
        .unwrap();
        let l = gpu.launches_snapshot();
        assert_eq!(l.len(), 3);
        for (i, (rows, r0)) in [(16u32, 0usize), (16, 16), (8, 32)].into_iter().enumerate() {
            assert_eq!(l[i].func, narrow.0, "launch {i}");
            assert_eq!(ptr(&l[i].args[0]), a.offset(r0 * kk * 2));
            assert_eq!(ptr(&l[i].args[1]), w);
            assert_eq!(ptr(&l[i].args[2]), c.offset(r0 * n * 4));
            assert_eq!(u32_arg(&l[i].args[3]), rows);
        }
        let no_batch = Batchm {
            narrow: KernelHandle(0),
            ..b
        };
        assert!(
            glm_mm_rows16(
                &gpu,
                gemm,
                gemv,
                no_batch,
                a,
                w,
                c,
                40,
                n,
                kk,
                ops::DenseCallSite::General,
                7
            )
            .is_err()
        );
        assert_eq!(gpu.launches_snapshot().len(), 3);
    }

    const SITES: [ops::DenseCallSite; 4] = [
        ops::DenseCallSite::Prefill,
        ops::DenseCallSite::Decode,
        ops::DenseCallSite::Verify,
        ops::DenseCallSite::General,
    ];

    fn rows32_rows(on: bool, site: ops::DenseCallSite) -> Vec<usize> {
        (0..=40).filter(|&m| rows32_takes(on, site, m)).collect()
    }

    /// 2026-10-16: Under `METRALE_DENSE_BATCHM_32=1` a prefill launch never takes the 16-row
    /// split, at 17, 24 and 32 rows or any other count, and neither does a `General` one.
    #[test]
    fn prefill_never_takes_rows32() {
        for site in [ops::DenseCallSite::Prefill, ops::DenseCallSite::General] {
            for m in [17, 24, 32] {
                assert!(!rows32_takes(true, site, m), "{site:?} at {m} rows");
            }
            assert!(rows32_rows(true, site).is_empty(), "{site:?}");
            assert!(!rows32_takes(true, site, usize::MAX), "{site:?}");
        }
    }

    /// 2026-10-16: With the lever on, the batched decode and verify take exactly 17..=32 rows;
    /// only `1` turns the lever on.
    #[test]
    fn decode_and_verify_take_exactly_17_to_32_rows() {
        for site in [ops::DenseCallSite::Decode, ops::DenseCallSite::Verify] {
            let want: Vec<usize> = (17..=32).collect();
            assert_eq!(rows32_rows(true, site), want, "{site:?}");
            assert!(!rows32_takes(true, site, usize::MAX), "{site:?}");
        }
        assert!(ops::dense_batchm_32_from(Some("1")));
        for v in [None, Some("0"), Some("true"), Some(""), Some("32")] {
            assert!(!ops::dense_batchm_32_from(v), "{v:?}");
        }
    }

    /// 2026-10-16: With the lever off no site takes the 16-row split at any row count.
    #[test]
    fn lever_off_no_site_takes_rows32() {
        for site in SITES {
            assert!(rows32_rows(false, site).is_empty(), "{site:?}");
        }
    }

    /// 2026-10-16: The default (wide only) takes the wide entry at exactly 9..=16 rows; a reg8
    /// handle adds exactly 2..=8; `METRALE_GLM_BATCHM_WIDE=0` drops 9..=16; one row, 17 rows and
    /// missing handles never take a register entry. Only `1` turns the narrow lever on.
    #[test]
    fn register_entries_follow_the_handles_and_the_wide_lever() {
        let (narrow, reg8, wide) = (KernelHandle(4), KernelHandle(5), KernelHandle(6));
        let picks = |b: Batchm, wide_on: bool| -> Vec<(usize, u64)> {
            (0..=20)
                .filter_map(|m| b.register_entry(m, wide_on).map(|h| (m, h.0)))
                .collect()
        };
        let base = Batchm {
            narrow,
            reg8: KernelHandle(0),
            wide,
            out_bytes: 4,
        };
        let w: Vec<(usize, u64)> = (9..=16).map(|m| (m, 6)).collect();
        let r: Vec<(usize, u64)> = (2..=8).map(|m| (m, 5)).collect();
        assert_eq!(picks(base, true), w);
        assert!(picks(base, false).is_empty());
        let both = Batchm { reg8, ..base };
        assert_eq!(picks(both, true), [r.clone(), w].concat());
        assert_eq!(picks(both, false), r);
        assert!(picks(Batchm::from(narrow), true).is_empty());
        assert!(both.register_entry(usize::MAX, true).is_none());
        assert!(narrow_lever(Some("1")));
        for v in [None, Some("0"), Some("true"), Some(""), Some("on")] {
            assert!(!narrow_lever(v), "{v:?}");
        }
    }
}
