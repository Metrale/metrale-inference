// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters for the W8A8 decode GEMVs (WxAy engine, `w8a8_gemv.cu`). The case's
//! activation is already E4M3 with its scales (what `w8a8_act_quant` leaves in the scratch), so
//! the adapter uploads it as the scratch and calls the engine's `ops::w8a8_gemv`, which picks the
//! token-tile entry for the row count exactly as a serve does.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every rowscale and blk128 entry sums in one order (w8a8_decode.rs `entry_index`), so the
//!   contract names every entry and the launcher's choice changes speed only.

use anyhow::Result;
use metrale_accuracy::case::Case;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::quantized::{Fp8Weight, WeightQuantFormat};

use super::accuracy_adapters::{Adapter, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The W8A8 launchers.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    ("w8a8_gemv::w8a8_gemv_rowscale_mb1_ku8", w8a8),
    ("w8a8_gemv::w8a8_gemv_rowscale_mb2", w8a8),
    ("w8a8_gemv::w8a8_gemv_rowscale_mb4", w8a8),
    ("w8a8_gemv::w8a8_gemv_rowscale_mb8", w8a8),
    ("w8a8_gemv::w8a8_gemv_rowscale_mb16", w8a8),
    ("w8a8_gemv::w8a8_gemv_blk128_mb1_ku8", w8a8),
    ("w8a8_gemv::w8a8_gemv_blk128_mb2", w8a8),
    ("w8a8_gemv::w8a8_gemv_blk128_mb4", w8a8),
    ("w8a8_gemv::w8a8_gemv_blk128_mb8", w8a8),
    ("w8a8_gemv::w8a8_gemv_blk128_mb16", w8a8),
];

fn w8a8(dev: &mut Dev<'_>, case: &Case, _kernel: KernelHandle) -> Result<Vec<u8>> {
    let t = |n: &str| case.tensor(n).map_err(not_runnable);
    let (x, w) = (t("x")?, t("w")?);
    let (rows, k, n) = (x.dims[0], x.dims[1], w.dims[0]);
    let (scale_t, format) = match (case.tensors.get("w_row"), case.tensors.get("w_block")) {
        (Some(s), None) => (s, WeightQuantFormat::Fp8PerRow),
        (None, Some(s)) => (s, WeightQuantFormat::Fp8BlockScaled),
        _ => return Err(not_runnable("a W8A8 weight has row or block scales".into())),
    };
    let act_scale = match format {
        WeightQuantFormat::Fp8PerRow => t("x_row")?,
        _ => t("x_block")?,
    };
    let weight = Fp8Weight {
        weight: dev.upload(w)?,
        row_scale: dev.upload(scale_t)?,
        n: n as u32,
        k: k as u32,
        scale_format: format,
    };
    let ww = ops::W8a8Weight::new(&[weight]).map_err(|e| not_runnable(format!("{e:#}")))?;
    let scratch = ops::W8a8Scratch {
        q: dev.upload(x)?,
        q_bytes: x.bytes.len(),
        scale: dev.upload(act_scale)?,
        scale_bytes: act_scale.bytes.len(),
    };
    let kernels = ops::W8a8Kernels::load(dev.gpu);
    if !ops::w8a8_decode_available(&kernels, &ww, rows, &scratch) {
        return Err(not_runnable(format!(
            "w8a8 not available for {rows} rows of [{n}, {k}]"
        )));
    }
    let y = dev.output(rows * n * 2)?;
    ops::w8a8_gemv(
        dev.gpu, &kernels, &ww, &scratch, rows, y, n as u32, dev.stream,
    )?;
    dev.read(y, rows * n * 2)
}
