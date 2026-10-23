// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The BF16 projection dispatch every GLM-5.3 block shares (`glm_mm`), with the
//! register-resident batched GEMV for 9..=16 rows (`dense_gemv_bf16_batchm_wide`).
//! 2026-10-21: and for 2..=8 rows (`dense_gemv_bf16_batchm_few`, B loads in flight together).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Only `glm_mm` launches the wide entries, only at `BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M`
//!   rows, and only when the caller passes a wide handle (the few entries: 2..=`BATCHM_FEW_MAX`
//!   rows and a few handle); every other M takes the dispatch the
//!   DSA and MLP `gemm` helpers took before (cuBLASLt above 16 rows, else
//!   `ops::dense_mm_bf16`), which other models share and which is unchanged.
//! - A wide or few entry's row has the bits of the runtime-M batched GEMV's row
//!   (`kernels/gb10/common/dense_gemv_bf16_batchm.cu`), which has `dense_gemv_bf16`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;

use super::GlmLevers;

/// 2026-10-09: `#define BATCHM_WIDE_MIN` in `dense_gemv_bf16_batchm.cu`: the narrowest M the
/// wide entries take.
pub(crate) const BATCHM_WIDE_MIN: usize = 9;
/// 2026-10-21: `bmfew::FEW_MAX_M` in `dense_gemv_bf16_batchm_few.cu`: the widest M the few
/// entries take.
pub(crate) const BATCHM_FEW_MAX: usize = 8;

/// 2026-10-09: A batched-GEMV pair: `narrow` is the runtime-M entry `ops::dense_mm_bf16`
/// launches at 2..=16 rows; `wide` the register-resident entry for 9..=16 rows, or
/// `KernelHandle(0)` to keep `narrow` there; 2026-10-21: `few` the same for 2..=8 rows
/// (`dense_gemv_bf16_batchm_few`). 2026-10-11: `out_bytes`, the pair's output element
/// size (2 for the BF16-out entries, 4 for the FP32-out ones), sizes a row of `C`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Batchm {
    pub(crate) narrow: KernelHandle,
    pub(crate) wide: KernelHandle,
    pub(crate) few: KernelHandle,
    pub(crate) out_bytes: usize,
}

/// 2026-10-21: The `dense_gemv_bf16_batchm_few` entries, BF16 and FP32 out, resolved once per
/// block's kernel set; `KernelHandle(0)` each where the target lacks it (the runtime-M entry runs).
#[derive(Clone, Copy, Debug)]
pub struct FewGemv {
    pub bf16: KernelHandle,
    pub f32: KernelHandle,
}

impl FewGemv {
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let entry = |name| {
            metrale_model_layers::layers::try_kernel(gpu, "dense_gemv_bf16_batchm_few", name)
        };
        Self {
            bf16: entry("dense_gemv_bf16_batchm_few"),
            f32: entry("dense_gemv_bf16_batchm_few_fp32out"),
        }
    }
}

impl Batchm {
    /// 2026-10-21: A block's batched entries for one output type: `out_bytes` 2 takes `few.bf16`,
    /// 4 takes `few.f32`.
    pub(crate) fn of(
        narrow: KernelHandle,
        wide: KernelHandle,
        few: FewGemv,
        out_bytes: usize,
    ) -> Self {
        let few = if out_bytes == 4 { few.f32 } else { few.bf16 };
        Self {
            narrow,
            wide,
            few,
            out_bytes,
        }
    }
}

impl From<KernelHandle> for Batchm {
    /// 2026-10-11: A BF16-out narrow entry alone.
    fn from(narrow: KernelHandle) -> Self {
        Self {
            narrow,
            wide: KernelHandle(0),
            few: KernelHandle(0),
            out_bytes: 2,
        }
    }
}

/// 2026-10-09: Whether `glm_mm` takes the wide entry for `m` rows with handle `wide`. `on` is
/// `METRALE_GLM_BATCHM_WIDE` (`DenseLevers::batchm_wide`): off keeps the runtime-M batched GEMV at
/// 9..=16 rows (an A/B of the speed; the bits are the same).
pub(crate) fn takes_wide(wide: KernelHandle, m: usize, on: bool) -> bool {
    wide.0 != 0 && (BATCHM_WIDE_MIN..=ops::DENSE_GEMV_BATCHM_MAX_M as usize).contains(&m) && on
}

/// 2026-10-21: The register-resident entry `glm_mm` launches for `m` rows, if any: `wide` at
/// 9..=16 rows under `METRALE_GLM_BATCHM_WIDE`, `few` at 2..=8 under `METRALE_GLM_BATCHM_FEW`
/// (`DenseLevers::batchm_few`; off keeps the runtime-M entry, the A/B control).
pub(crate) fn fixed_entry(
    batchm: &Batchm,
    m: usize,
    dense: &crate::glm5next_dense_levers::DenseLevers,
) -> Option<KernelHandle> {
    if takes_wide(batchm.wide, m, dense.batchm_wide) {
        return Some(batchm.wide);
    }
    (batchm.few.0 != 0 && (2..=BATCHM_FEW_MAX).contains(&m) && dense.batchm_few)
        .then_some(batchm.few)
}

/// 2026-10-23: One launch of a fixed-count batched entry (`wide` or `few`) over `m` rows: the
/// launch [`glm_mm`] makes for one, and the plan executor's for a group that names one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_fixed(
    gpu: &dyn GpuBackend,
    entry: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, entry)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(n as u32)
        .launch(stream)
}

/// 2026-10-11: [`glm_mm`] in launches of at most `DENSE_GEMV_BATCHM_MAX_M` rows, each on the
/// GEMV tiers (one row: the M = 1 GEMV; 2..=16: the batched GEMV), whose rows have the M = 1
/// GEMV's bits; `C` rows are `n` elements of `batchm.out_bytes`. Errors when the batched handle
/// is missing (the tile GEMM would take the rows, in another order).
#[allow(clippy::too_many_arguments)]
fn glm_mm_rows16(
    gpu: &dyn GpuBackend,
    levers: &GlmLevers,
    gemm: KernelHandle,
    gemv: KernelHandle,
    batchm: Batchm,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        batchm.narrow.0 != 0 && gemv.0 != 0,
        "METRALE_GLM_BATCH_INVARIANT=1: a {m}-row projection needs the GEMV tiers"
    );
    let step = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    let c_elem = batchm.out_bytes;
    let mut r0 = 0;
    while r0 < m {
        let rows = step.min(m - r0);
        let (ar, cr) = (a.offset(r0 * kk * 2), c.offset(r0 * n * c_elem));
        glm_mm(
            gpu, levers, gemm, gemv, batchm, ar, b, cr, rows, n, kk, stream,
        )?;
        r0 += rows;
    }
    Ok(())
}

/// 2026-10-09: `C[M, N] = A[M, K] @ B[N, K]^T` with BF16 inputs (the output element is the
/// kernels'): the W4A16 or FP8 dense path for a registered weight, then cuBLASLt above
/// `DENSE_GEMV_BATCHM_MAX_M` rows when `levers.cublas_wide_proj` is on (BF16 out), the wide batched GEMV at 9..=16 rows when `batchm.wide` is set, else
/// `ops::dense_mm_bf16` (GEMV at one row, `batchm.narrow` at 2..=16, the tile GEMM otherwise).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_mm(
    gpu: &dyn GpuBackend,
    levers: &GlmLevers,
    gemm: KernelHandle,
    gemv: KernelHandle,
    batchm: Batchm,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // 2026-10-09: `--dense-quantization fp8` serves registered weights W8A8; only BF16-output
    // projections are registered (each block's `dense_projections`). Under `w4a16` the W4A16
    // registry answers first, then the FP8 one for the rest of the set.
    if crate::glm5next_w4a16_dense::proj(gpu, &levers.dense, b, a, c, m, n, kk, stream)? {
        return Ok(());
    }
    if crate::glm5next_fp8_dense::proj(gpu, b, a, c, m, n, kk, stream)? {
        return Ok(());
    }
    // 2026-10-11: Under `METRALE_GLM_BATCH_INVARIANT`, wider launches run as 16-row launches of
    // the row-invariant batched GEMV ([`glm_mm_rows16`]), never cuBLASLt or the tile GEMM.
    let wide_rows = m > ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    if wide_rows && levers.dense.batch_invariant {
        return glm_mm_rows16(gpu, levers, gemm, gemv, batchm, a, b, c, m, n, kk, stream);
    }
    if wide_rows && levers.cublas_wide_proj {
        return ops::cublas_bf16_proj_dense(a, b, c, m as u32, n as u32, kk as u32, stream);
    }
    if let Some(entry) = fixed_entry(&batchm, m, &levers.dense) {
        return launch_fixed(gpu, entry, a, b, c, m, n, kk, stream);
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

    /// 2026-10-21: The fixed-count entry per width: few at 2..=8, wide at 9..=16, neither at 1
    /// or above 16, and each lever and each missing handle falls back to the runtime-M entry
    /// (`None`) at its own widths only.
    #[test]
    fn fixed_entry_is_few_at_2_to_8_and_wide_at_9_to_16() {
        let (wide, few) = (21u64, 22u64);
        let b = Batchm {
            narrow: KernelHandle(20),
            wide: KernelHandle(wide),
            few: KernelHandle(few),
            out_bytes: 4,
        };
        let on = crate::glm5next_dense_levers::DenseLevers::declared();
        assert!(on.batchm_few && on.batchm_wide, "both default on");
        let pick = |b: &Batchm, d, m| fixed_entry(b, m, d).map(|h| h.0);
        for m in [0, 1, 17, 32] {
            assert_eq!(pick(&b, &on, m), None, "m={m}");
        }
        for m in 2..=8 {
            assert_eq!(pick(&b, &on, m), Some(few), "m={m}");
        }
        for m in 9..=16 {
            assert_eq!(pick(&b, &on, m), Some(wide), "m={m}");
        }
        let few_off = crate::glm5next_dense_levers::DenseLevers {
            batchm_few: false,
            ..on
        };
        let wide_off = crate::glm5next_dense_levers::DenseLevers {
            batchm_wide: false,
            ..on
        };
        let no_few = Batchm {
            few: KernelHandle(0),
            ..b
        };
        assert_eq!(
            (pick(&b, &few_off, 2), pick(&b, &few_off, 9)),
            (None, Some(wide))
        );
        assert_eq!(
            (pick(&b, &wide_off, 8), pick(&b, &wide_off, 9)),
            (Some(few), None)
        );
        assert_eq!(
            (pick(&no_few, &on, 8), pick(&no_few, &on, 9)),
            (None, Some(wide))
        );
        let off = crate::glm5next_dense_levers::DenseLevers {
            batchm_few: false,
            ..on
        };
        let read = crate::glm5next_dense_levers::DenseLevers::from_settings(
            &metrale_model_layers::lever_equivalence::settings_from(&[(
                "METRALE_GLM_BATCHM_FEW",
                "0",
            )])
            .expect("0 resolves"),
        )
        .expect("levers resolve");
        assert_eq!(read, off, "`0` switches only the few entry off");
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
            wide: KernelHandle(0),
            few: KernelHandle(0),
            out_bytes: 4,
        };
        let (a, w, c) = (
            DevicePtr(0x1000_0000),
            DevicePtr(0x2000_0000),
            DevicePtr(0x3000_0000),
        );
        let (n, kk) = (288usize, 4096usize);
        let levers = GlmLevers::declared();
        glm_mm_rows16(&gpu, &levers, gemm, gemv, b, a, w, c, 40, n, kk, 7).unwrap();
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
        assert!(glm_mm_rows16(&gpu, &levers, gemm, gemv, no_batch, a, w, c, 40, n, kk, 7).is_err());
        assert_eq!(gpu.launches_snapshot().len(), 3);
    }
}
