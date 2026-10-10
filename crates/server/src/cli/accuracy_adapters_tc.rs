// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters for the tensor-core projections: the BF16 GEMV tiers
//! (`dense_gemv_bf16_tc.cu`), the NVFP4 W4A16 GEMV tiers (`w4a16_gemv_tc.cu`), the NVFP4 row-tile
//! head (`w4a16_tc_rows.cu`) and the FP8 block-128 W8A16 GEMMs (`w8a16_gemm_pipelined*.cu`). The
//! canonical operands are the layouts these kernels read (row-major `[N, K]` weights, E2M1 pairs
//! low nibble first, `[N / 16 | N / 128, K / 16 | K / 128]` scales), so nothing is repacked.
//!
//! Owner: server CLI.
//! Invariants:
//! - Each adapter calls the engine launcher the serve path calls, and first proves the launcher
//!   will run the case's entry point: a router or lever that sends the shape elsewhere
//!   (`METRALE_NO_MTP_TC`, `METRALE_NO_W4A16_TC`, the canonical row tiers that send the W8A16
//!   tile twins to `w8a16_tc_rows`, the 128x128 twin from 256 rows) refuses the case by name.
//! - Where the launcher takes the entry point (dense_gemv_tc, the W8A16 GEMMs), a wrong-symbol
//!   arm runs on the contract's own grid; where it resolves the entry itself (w4a16_gemv_batchm,
//!   w4a16_tc_rows), a different symbol is refused.

use anyhow::Result;
use metrale_accuracy::case::Case;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::{RowTiers, ops, row_tiers};
use metrale_model_layers::weight_map::quantized::DenseWeight;

use super::accuracy_adapters::{Adapter, at, handle, linear_dims, not_runnable, nvfp4_weight};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The tensor-core projection launchers.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    ("dense_gemv_bf16_tc::dense_gemv_bf16_tc8", dense_tc),
    ("dense_gemv_bf16_tc::dense_gemv_bf16_tc16", dense_tc),
    ("dense_gemv_bf16_tc::dense_gemv_bf16_tc32", dense_tc),
    ("w4a16_gemv_tc::w4a16_gemv_tc8", w4a16_tc),
    ("w4a16_gemv_tc::w4a16_gemv_tc16", w4a16_tc),
    ("w4a16_tc_rows::w4a16_tc_rows_64", w4a16_rows),
    (
        "w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m32",
        w8a16_gemm,
    ),
    (
        "w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m64",
        w8a16_gemm,
    ),
    ("w8a16_gemm_pipelined::w8a16_gemm_pipelined", w8a16_gemm),
];

/// 2026-10-09: Fewest rows at which `ops::w8a16_gemm_pipelined` launches its 128x128 twin
/// instead (`W8A16_PIPE128_MIN_ROWS`, gemm_quant_w8a16.rs:128, which `ops` does not re-export).
const W8A16_PIPE128_FROM_ROWS: usize = 256;

fn need(cond: bool, why: impl FnOnce() -> String) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(not_runnable(why()))
    }
}

fn unsplit(case: &Case) -> Result<()> {
    need(case.split.is_empty(), || {
        format!(
            "{} writes rows at stride N; a split case has no launcher",
            case.launcher
        )
    })
}

/// 2026-10-09: The launcher resolves its own entry: only the contract's entry can run.
fn own_entry(dev: &Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<KernelHandle> {
    let own = handle(dev, &case.launcher)?;
    need(own.0 == kernel.0, || {
        format!(
            "{}'s engine launcher resolves its entry itself; `{}` cannot run under it",
            case.launcher, case.kernel
        )
    })?;
    Ok(own)
}

fn dense_tc(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    unsplit(case)?;
    let own = handle(dev, &case.launcher)?;
    // 2026-10-10: The envelope sweep launches the entry on its own grid (every tier's CTA covers
    // ROWS_PER_CTA outputs) without asking the router, up to the tier's own row capacity.
    let grid_x = if dev.sweep {
        need(
            rows >= ops::dense_gemv_tc::MIN_M as usize
                && rows <= sweep_max_rows(&case.launcher).unwrap_or(0),
            || format!("{} does not run {rows} rows", case.launcher),
        )?;
        need(k % ops::dense_gemv_tc::K_STEP as usize == 0, || {
            format!("the BF16 tensor-core GEMV needs K % 64 == 0, not {k}")
        })?;
        (n as u32).div_ceil(ops::dense_gemv_tc::ROWS_PER_CTA)
    } else {
        // 2026-10-09: The router of the drafter's GEMMs (draft.rs takes the same pair).
        let (routed, grid_x) =
            ops::dense_gemv_tc::kernel_for(dev.gpu, rows as u32, n as u32, k as u32).ok_or_else(
                || {
                    not_runnable(format!(
                        "dense_gemv_tc routes no entry for {rows} rows of [{n}, {k}] \
                 (METRALE_NO_MTP_TC set, K % 64 != 0, or rows outside 2..=32)"
                    ))
                },
            )?;
        need(routed.0 == own.0, || {
            format!(
                "dense_gemv_tc routes {rows} rows to another entry than `{}`",
                case.launcher
            )
        })?;
        grid_x
    };
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = DenseWeight {
        weight: dev.upload(case.tensor("w").map_err(not_runnable)?)?,
    };
    let y = dev.output(rows * n * 2)?;
    ops::dense_gemv_tc::launch(
        dev.gpu,
        kernel,
        grid_x,
        x,
        &w,
        y,
        rows as u32,
        n as u32,
        k as u32,
        n as u32,
        dev.stream,
    )?;
    dev.read(y, rows * n * 2)
}

fn w4a16_tc(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    unsplit(case)?;
    let own = own_entry(dev, case, kernel)?;
    if dev.sweep {
        return w4a16_tc_direct(dev, case, own, rows, k, n);
    }
    // 2026-10-09: The route `ops::w4a16_gemv_batchm` takes; its handle cache is keyed by the
    // backend's address, so a stale entry after a target switch is refused here, not launched.
    let (routed, _) = ops::gemv_tc::tc_kernel(dev.gpu, rows as u32, n as u32, k as u32)
        .ok_or_else(|| {
            not_runnable(format!(
                "gemv_tc routes no tensor-core entry for {rows} rows of [{n}, {k}] \
                 (METRALE_NO_W4A16_TC set, K % 128 != 0, or rows outside 1..=16)"
            ))
        })?;
    need(routed.0 == own.0, || {
        format!(
            "gemv_tc routes {rows} rows to another entry than `{}`",
            case.launcher
        )
    })?;
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = nvfp4_weight(dev, case)?;
    let y = dev.output(rows * n * 2)?;
    ops::w4a16_gemv_batchm(
        dev.gpu,
        kernel,
        x,
        &w,
        y,
        rows as u32,
        n as u32,
        k as u32,
        dev.stream,
    )?;
    dev.read(y, rows * n * 2)
}

/// 2026-10-10: The envelope sweep's launch of one tensor-core W4A16 tier at any row count it
/// holds: the tier's own grid and block, as `ops::w4a16_gemv_batchm` launches the tier its
/// router picks (quant_dispatch.rs: `n / cols_per_cta` CTAs of `TC_BLOCK` threads).
fn w4a16_tc_direct(
    dev: &mut Dev<'_>,
    case: &Case,
    kernel: KernelHandle,
    rows: usize,
    k: usize,
    n: usize,
) -> Result<Vec<u8>> {
    need(rows <= sweep_max_rows(&case.launcher).unwrap_or(0), || {
        format!("{} holds fewer than {rows} rows", case.launcher)
    })?;
    need(k.is_multiple_of(128), || {
        format!("the tensor-core GEMV needs K % 128 == 0, not {k}")
    })?;
    let cols = if case.launcher.ends_with("tc8") {
        ops::gemv_tc::TC8_COLS_PER_CTA
    } else {
        ops::gemv_tc::TC16_COLS_PER_CTA
    };
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = nvfp4_weight(dev, case)?;
    let y = dev.output(rows * n * 2)?;
    KernelLaunch::new(dev.gpu, kernel)
        .grid([(n as u32).div_ceil(cols), 1, 1])
        .block([ops::gemv_tc::TC_BLOCK, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_f32(w.weight_scale_2)
        .arg_ptr(y)
        .arg_u32(rows as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(dev.stream)?;
    dev.read(y, rows * n * 2)
}

/// 2026-10-10: The fewest rows a tier runs (the BF16 tensor-core tiers start at two rows).
pub(crate) fn sweep_min_rows(launcher: &str) -> usize {
    if launcher.starts_with("dense_gemv_bf16_tc::") {
        ops::dense_gemv_tc::MIN_M as usize
    } else {
        1
    }
}

/// 2026-10-10: The most rows one launch of a tensor-core tier holds (the routers' tier caps):
/// the envelope sweep runs a tier only up to it.
pub(crate) fn sweep_max_rows(launcher: &str) -> Option<usize> {
    let cap = match launcher {
        "w4a16_gemv_tc::w4a16_gemv_tc8" => ops::gemv_tc::TC8_MAX_M,
        "w4a16_gemv_tc::w4a16_gemv_tc16" => ops::gemv_tc::TC16_MAX_M,
        "dense_gemv_bf16_tc::dense_gemv_bf16_tc8" => ops::dense_gemv_tc::TC8_MAX_M,
        "dense_gemv_bf16_tc::dense_gemv_bf16_tc16" => ops::dense_gemv_tc::TC16_MAX_M,
        "dense_gemv_bf16_tc::dense_gemv_bf16_tc32" => ops::dense_gemv_tc::TC32_MAX_M,
        _ => return None,
    };
    Some(cap as usize)
}

/// 2026-10-09: The declared NVFP4 head in 64-row calls, as lm_head_nvfp4_rows.rs runs it; the
/// launcher picks `_16`, `_32` or `_64` by the call's rows, whose per-row bits the kernel
/// declares equal (w4a16_tc_rows.cu:194).
fn w4a16_rows(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    unsplit(case)?;
    own_entry(dev, case, kernel)?;
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = nvfp4_weight(dev, case)?;
    let y = dev.output(rows * n * 2)?;
    let mut done = 0;
    while done < rows {
        let m = (rows - done).min(ops::W4A16_TC_ROWS_MAX_M as usize);
        ops::w4a16_tc_rows(
            dev.gpu,
            at(x, done * k * 2),
            &w,
            at(y, done * n * 2),
            m as u32,
            n as u32,
            k as u32,
            k as u32,
            n as u32,
            dev.stream,
        )?;
        done += m;
    }
    dev.read(y, rows * n * 2)
}

fn w8a16_gemm(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    unsplit(case)?;
    let block_rows = case.scalar("w_block_rows").map_err(not_runnable)?;
    need(block_rows == 128.0, || {
        format!("a [128, 128]-block weight, not {block_rows}-row blocks")
    })?;
    let twin = case.launcher.contains("_m32::");
    need(!twin || row_tiers() != RowTiers::Canonical, || {
        "the canonical row tiers (METRALE_CANONICAL_TIERS) send the tile twins to \
         w8a16_tc_rows unless METRALE_NO_W8A16_TC_ROWS is set; run without them"
            .into()
    })?;
    need(twin || rows < W8A16_PIPE128_FROM_ROWS, || {
        format!("from {W8A16_PIPE128_FROM_ROWS} rows the launcher runs w8a16_gemm_pipe128")
    })?;
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = dev.upload(case.tensor("w").map_err(not_runnable)?)?;
    let s = dev.upload(case.tensor("w_block").map_err(not_runnable)?)?;
    let y = dev.output(rows * n * 2)?;
    let (m, n32, k32) = (rows as u32, n as u32, k as u32);
    match case.launcher.as_str() {
        "w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m32" => {
            ops::w8a16_gemm_pipelined_m32(dev.gpu, kernel, x, w, s, y, m, n32, k32, dev.stream)?
        }
        "w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m64" => {
            ops::w8a16_gemm_pipelined_m64(dev.gpu, kernel, x, w, s, y, m, n32, k32, dev.stream)?
        }
        _ => ops::w8a16_gemm_pipelined(dev.gpu, kernel, x, w, s, y, m, n32, k32, dev.stream)?,
    }
    dev.read(y, rows * n * 2)
}
