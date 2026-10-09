// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters for the W8A8 activation quantizers (`w8a8_act_quant.cu`): the
//! engine's `ops::w8a8_act_quant` writes codes and scales into a scratch; the adapter reads both
//! back from guarded buffers and decodes them with the layout the case declares (one scale per
//! row, or one per 128 values), as the consuming GEMV reads them.
//!
//! Owner: server CLI.
//! Invariants:
//! - The decode follows the case's declared granularity, never the kernel's: a quantizer writing
//!   the other granularity decodes to wrong values.

use anyhow::Result;
use metrale_accuracy::case::Case;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_model_layers::layers::ops;

use super::accuracy_adapters::{Adapter, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The quantizer launchers.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    ("w8a8_act_quant::w8a8_act_quant_row", quant),
    ("w8a8_act_quant::w8a8_act_quant_g128", quant),
];

fn quant(dev: &mut Dev<'_>, case: &Case, _kernel: KernelHandle) -> Result<Vec<u8>> {
    let x = case.tensor("x").map_err(not_runnable)?;
    let (rows, k) = (x.dims[0], x.dims[1]);
    let declared_group = if case.launcher.ends_with("g128") {
        128
    } else {
        k
    };
    // 2026-10-09: The entry point launched is the one the engine's launcher resolves for the
    // granularity `case.kernel` names; a wrong-symbol mutation therefore runs the other
    // granularity's quantizer while the output is decoded with the declared one.
    let scale = if case.kernel.ends_with("g128") {
        ops::W8a8Scale::Block128
    } else {
        ops::W8a8Scale::PerRow
    };
    // 2026-10-09: The scratch holds the larger scale layout either kernel may write.
    let scale_slots = rows * k.div_ceil(128);
    let scratch = ops::W8a8Scratch {
        q: dev.output(rows * k)?,
        q_bytes: rows * k,
        scale: dev.output(scale_slots * 4)?,
        scale_bytes: scale_slots * 4,
    };
    let xd = dev.upload(x)?;
    let kernels = ops::W8a8Kernels::load(dev.gpu);
    ops::w8a8_act_quant(
        dev.gpu, &kernels, scale, xd, k as u32, rows, k as u32, &scratch, dev.stream,
    )?;
    let q = dev.read(scratch.q, rows * k)?;
    let s = dev.read(scratch.scale, scale_slots * 4)?;
    let sc = |i: usize| {
        f64::from(f32::from_le_bytes([
            s[4 * i],
            s[4 * i + 1],
            s[4 * i + 2],
            s[4 * i + 3],
        ]))
    };
    let groups = k / declared_group;
    let mut out = Vec::with_capacity(rows * k * 4);
    for r in 0..rows {
        for c in 0..k {
            let v = metrale_accuracy::elem::e4m3_to_f64(q[r * k + c])
                * sc(r * groups + c / declared_group);
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    Ok(out)
}
