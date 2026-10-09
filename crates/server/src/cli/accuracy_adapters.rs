// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters: for each contracted entry point, how the canonical operands of
//! a case (`metrale_accuracy::case`) reach the kernel. Where the engine has a launcher
//! (`metrale_model_layers::layers::ops`), the adapter calls it, so the grid, block and argument
//! order checked are the ones the engine runs. The symbol launched is `case.kernel`; the
//! adapter is chosen by `case.launcher`, so a wrong-symbol mutation runs another entry point
//! under the contract's own launcher.
//!
//! Owner: server CLI.
//! Invariants:
//! - A launcher with no adapter is an error naming it, never a skipped check.
//! - Row and shard loops reproduce what the engine's callers do: one-row kernels run once per
//!   row; a shard runs its weight rows and writes at its output column.

use anyhow::Result;
use metrale_accuracy::case::{Case, Shard};
use metrale_accuracy::runner::RunError;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::quantized::{DenseWeight, QuantizedWeight};

use super::accuracy_gpu::Dev;

/// 2026-10-09: A case the adapter cannot launch (a setup problem, not a kernel fault).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct NotRunnable(String);

pub(crate) fn not_runnable(why: String) -> anyhow::Error {
    anyhow::Error::new(NotRunnable(why))
}

macro_rules! need {
    ($cond:expr, $($why:tt)+) => {
        if !$cond {
            return Err(not_runnable(format!($($why)+)));
        }
    };
}

pub(crate) fn handle(dev: &Dev<'_>, kernel: &str) -> Result<KernelHandle> {
    let (m, f) = kernel
        .split_once("::")
        .ok_or_else(|| not_runnable(format!("`{kernel}` is not module::function")))?;
    dev.gpu.kernel(m, f).map_err(|e| {
        not_runnable(format!(
            "kernel `{kernel}` is not in this target's PTX: {e:#}"
        ))
    })
}

pub(crate) fn at(p: DevicePtr, bytes: usize) -> DevicePtr {
    DevicePtr(p.0 + bytes as u64)
}

pub(crate) fn shards(case: &Case, n: usize) -> Vec<Shard> {
    if case.split.is_empty() {
        vec![Shard {
            lo: 0,
            hi: n,
            out_at: 0,
        }]
    } else {
        case.split.clone()
    }
}

/// 2026-10-09: The linear case's sizes: rows, k, n.
pub(crate) fn linear_dims(case: &Case) -> Result<(usize, usize, usize)> {
    let x = case.tensor("x").map_err(not_runnable)?;
    let w = case.tensor("w").map_err(not_runnable)?;
    need!(
        x.dims[1] == w.dims[1],
        "x {:?} and w {:?} disagree on K",
        x.dims,
        w.dims
    );
    Ok((x.dims[0], x.dims[1], w.dims[0]))
}

/// 2026-10-09: An adapter: launch `case` with the resolved entry point; the output bytes.
pub(crate) type Adapter = fn(&mut Dev<'_>, &Case, KernelHandle) -> Result<Vec<u8>>;

/// 2026-10-09: The adapter of each launcher.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    ("w4a16_gemv::w4a16_gemv_sw", w4a16_one_row),
    ("w4a16_gemv::w4a16_gemv", w4a16_one_row),
    ("w4a16_gemv::w4a16_gemv_batch2", w4a16_batch),
    ("w4a16_gemv::w4a16_gemv_batch3", w4a16_batch),
    ("gemv::dense_gemv_bf16", dense_one_row),
    (
        "dense_gemv_bf16_batchm::dense_gemv_bf16_batchm",
        dense_batchm,
    ),
];

/// 2026-10-09: Every adapter table: this file's projections and each op class's own file.
const TABLES: &[&[(&str, Adapter)]] = &[
    ADAPTERS,
    super::accuracy_adapters_w8a8::ADAPTERS,
    super::accuracy_adapters_quant::ADAPTERS,
    super::accuracy_adapters_norm::ADAPTERS,
    super::accuracy_adapters_tc::ADAPTERS,
    super::accuracy_adapters_attention::ADAPTERS,
];

/// 2026-10-09: An adapter launches `launcher`.
pub(crate) fn has_adapter(launcher: &str) -> bool {
    TABLES
        .iter()
        .flat_map(|t| t.iter())
        .any(|(l, _)| *l == launcher)
}

/// 2026-10-09: Launch `case` and return its output bytes.
pub(crate) fn launch(dev: &mut Dev<'_>, case: &Case) -> std::result::Result<Vec<u8>, RunError> {
    let classify = |e: anyhow::Error| match e.downcast_ref::<NotRunnable>() {
        Some(n) => RunError::Unavailable(n.0.clone()),
        None => RunError::Fault(format!("{e:#}")),
    };
    let adapter = TABLES
        .iter()
        .flat_map(|t| t.iter())
        .find(|(l, _)| *l == case.launcher)
        .map(|(_, a)| *a)
        .ok_or_else(|| {
            RunError::Unavailable(format!("no launch adapter for `{}`", case.launcher))
        })?;
    let kernel = handle(dev, &case.kernel).map_err(classify)?;
    adapter(dev, case, kernel).map_err(classify)
}

/// 2026-10-09: Upload an NVFP4 weight as the engine holds one.
pub(crate) fn nvfp4_weight(dev: &mut Dev<'_>, case: &Case) -> Result<QuantizedWeight> {
    let mut w = QuantizedWeight::null();
    w.weight = dev.upload(case.tensor("w").map_err(not_runnable)?)?;
    w.weight_scale = dev.upload(case.tensor("w_block").map_err(not_runnable)?)?;
    w.weight_scale_2 = case.scalar("w_global").map_err(not_runnable)? as f32;
    Ok(w)
}

/// 2026-10-09: The shard's slice of an NVFP4 weight (rows `lo..`): packed and scale rows move.
fn nvfp4_rows(w: &QuantizedWeight, lo: usize, k: usize) -> QuantizedWeight {
    let mut s = *w;
    s.weight = at(w.weight, lo * k / 2);
    s.weight_scale = at(w.weight_scale, lo * k / 16);
    s
}

fn w4a16_one_row(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = nvfp4_weight(dev, case)?;
    let y = dev.output(rows * n * 2)?;
    for s in shards(case, n) {
        let ws = nvfp4_rows(&w, s.lo, k);
        for r in 0..rows {
            let (xi, yo) = (at(x, r * k * 2), at(y, (r * n + s.out_at) * 2));
            let ns = (s.hi - s.lo) as u32;
            match case.launcher.as_str() {
                "w4a16_gemv::w4a16_gemv_sw" => {
                    ops::w4a16_gemv_sw(dev.gpu, kernel, xi, &ws, yo, ns, k as u32, dev.stream)?
                }
                _ => ops::w4a16_gemv(dev.gpu, kernel, xi, &ws, yo, ns, k as u32, dev.stream)?,
            }
        }
    }
    dev.read(y, rows * n * 2)
}

fn w4a16_batch(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    let want = if case.launcher.ends_with("batch2") {
        2
    } else {
        3
    };
    need!(
        rows == want,
        "{} runs {want} rows, the case has {rows}",
        case.launcher
    );
    need!(
        case.split.is_empty(),
        "{} writes rows at stride N; a split case needs a strided kernel",
        case.launcher
    );
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = nvfp4_weight(dev, case)?;
    let y = dev.output(rows * n * 2)?;
    if want == 2 {
        ops::w4a16_gemv_batch2(dev.gpu, kernel, x, &w, y, n as u32, k as u32, dev.stream)?;
    } else {
        ops::w4a16_gemv_batch3(dev.gpu, kernel, x, &w, y, n as u32, k as u32, dev.stream)?;
    }
    dev.read(y, rows * n * 2)
}

fn dense_one_row(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = dev.upload(case.tensor("w").map_err(not_runnable)?)?;
    let y = dev.output(rows * n * 2)?;
    for s in shards(case, n) {
        let ws = DenseWeight {
            weight: at(w, s.lo * k * 2),
        };
        for r in 0..rows {
            let (xi, yo) = (at(x, r * k * 2), at(y, (r * n + s.out_at) * 2));
            ops::dense_gemv(
                dev.gpu,
                kernel,
                xi,
                &ws,
                yo,
                (s.hi - s.lo) as u32,
                k as u32,
                dev.stream,
            )?;
        }
    }
    dev.read(y, rows * n * 2)
}

fn dense_batchm(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, k, n) = linear_dims(case)?;
    let x = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = dev.upload(case.tensor("w").map_err(not_runnable)?)?;
    let y = dev.output(rows * n * 2)?;
    for s in shards(case, n) {
        let ws = DenseWeight {
            weight: at(w, s.lo * k * 2),
        };
        let yo = at(y, s.out_at * 2);
        ops::dense_gemv_batchm(
            dev.gpu,
            kernel,
            x,
            &ws,
            yo,
            rows as u32,
            (s.hi - s.lo) as u32,
            k as u32,
            n as u32,
            dev.stream,
        )?;
    }
    dev.read(y, rows * n * 2)
}
