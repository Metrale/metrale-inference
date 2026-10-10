// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters of the RMSNorm and RoPE contracts: the canonical operands of a
//! norm case (`x`, or the stream `h` and branch `s` of a residual-add norm, and `w`) and of a
//! rope case (`q`, `k`, `pos`) through the engine's own launchers in `ops`, so the grid and
//! block checked are the ones the engine runs.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every buffer a kernel writes (the normed row, the in-place stream, the residual copy, the
//!   in-place Q and K) is a guarded output: a write outside it fails the launch.
//! - A residual-add norm given a plain row adds a zero branch (`x + 0` is exact), so the
//!   contract checks the norm it declares; the stream and residual it also writes are checked
//!   only for writes outside their buffers.
//! - A rope case's output is each row's Q heads, then its K heads.

use std::sync::Arc;

use anyhow::Result;
use metrale_accuracy::case::{Case, Enc, Tensor};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::quantized::DenseWeight;

use super::accuracy_adapters::{Adapter, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The adapter of each norm and rope launcher.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    ("norm::rms_norm", norm_rows),
    ("norm::rms_norm_strided", norm_rows),
    ("norm::rms_norm_residual", norm_rows),
    ("norm::residual_add_rms_norm", norm_rows),
    (
        "residual_add_rms_norm_exact::residual_add_rms_norm_exact",
        norm_rows,
    ),
    ("rope::rope_forward", rope_rows),
    ("rope::rope_forward_strided", rope_rows),
];

/// 2026-10-09: Tensor `name` of `case`; a missing one is a setup error.
fn tensor<'a>(case: &'a Case, name: &str) -> Result<&'a Tensor> {
    case.tensor(name).map_err(not_runnable)
}

/// 2026-10-09: Scalar `name` of `case`; a missing one is a setup error.
fn scalar(case: &Case, name: &str) -> Result<f64> {
    case.scalar(name).map_err(not_runnable)
}

/// 2026-10-09: A guarded output buffer holding `t`'s bytes (a kernel that writes in place).
fn in_place(dev: &mut Dev<'_>, t: &Tensor) -> Result<DevicePtr> {
    let p = dev.output(t.bytes.len())?;
    dev.gpu.copy_h2d(&t.bytes, p)?;
    Ok(p)
}

/// 2026-10-09: The residual-add operands: the stream (`h`, or the plain row `x`) and the branch
/// (`s`, or zeros).
fn stream_and_branch(case: &Case) -> Result<(Tensor, Tensor)> {
    if let (Some(h), Some(s)) = (case.tensors.get("h"), case.tensors.get("s")) {
        return Ok((h.clone(), s.clone()));
    }
    let x = tensor(case, "x")?;
    let zeros = Tensor {
        enc: Enc::Bf16,
        dims: x.dims.clone(),
        bytes: Arc::new(vec![0u8; x.bytes.len()]),
    };
    Ok((x.clone(), zeros))
}

/// 2026-10-09: Launch a norm case under its launcher; the normed rows.
fn norm_rows(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (rows, n) = (case.out.0[0], case.out.0[1]);
    if case.out.1 != Enc::Bf16 {
        return Err(not_runnable(format!(
            "{} writes bf16, the case expects {:?}",
            case.launcher, case.out.1
        )));
    }
    let bytes = rows * n * 2;
    let w = DenseWeight {
        weight: dev.upload(tensor(case, "w")?)?,
    };
    let eps = scalar(case, "eps")? as f32;
    let (r, h) = (rows as u32, n as u32);
    let y = dev.output(bytes)?;
    let mut also = Vec::new();
    match case.launcher.as_str() {
        "norm::rms_norm" => {
            let x = dev.upload(tensor(case, "x")?)?;
            ops::rms_norm(dev.gpu, kernel, x, &w, y, r, h, eps, dev.stream)?;
        }
        "norm::rms_norm_strided" => {
            // 2026-10-09: One group of every row, packed: what rms_norm launches.
            let x = dev.upload(tensor(case, "x")?)?;
            let stride = (rows * n) as u32;
            ops::rms_norm_strided(dev.gpu, kernel, x, &w, y, r, 1, h, eps, stride, dev.stream)?;
        }
        "norm::rms_norm_residual" => {
            let x = dev.upload(tensor(case, "x")?)?;
            let res = dev.output(bytes)?;
            ops::rms_norm_residual(dev.gpu, kernel, x, &w, y, res, r, h, eps, dev.stream)?;
            also.push(res);
        }
        fused => {
            let (stream, branch) = stream_and_branch(case)?;
            let hid = in_place(dev, &stream)?;
            let src = dev.upload(&branch)?;
            let res = dev.output(bytes)?;
            let launch = match fused {
                "norm::residual_add_rms_norm" => ops::residual_add_rms_norm,
                "residual_add_rms_norm_exact::residual_add_rms_norm_exact" => {
                    ops::residual_add_rms_norm_exact
                }
                other => return Err(not_runnable(format!("no norm launcher `{other}`"))),
            };
            launch(dev.gpu, kernel, hid, src, &w, y, res, r, h, eps, dev.stream)?;
            also.extend([hid, res]);
        }
    }
    let out = dev.read(y, bytes)?;
    for p in also {
        dev.read(p, bytes)?;
    }
    Ok(out)
}

/// 2026-10-09: Launch a rope case under its launcher; each row's rotated Q, then K.
fn rope_rows(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let (q, k) = (tensor(case, "q")?, tensor(case, "k")?);
    let hd = scalar(case, "head_dim")? as usize;
    let rot = scalar(case, "rotary_dim")? as u32;
    let theta = scalar(case, "theta")? as f32;
    let (rows, qw, kw) = (q.dims[0], q.dims[1], k.dims[1]);
    if hd == 0 || !qw.is_multiple_of(hd) || !kw.is_multiple_of(hd) || k.dims[0] != rows {
        return Err(not_runnable(format!(
            "q {:?} and k {:?} are not whole heads of {hd}",
            q.dims, k.dims
        )));
    }
    let (nq, nk) = ((qw / hd) as u32, (kw / hd) as u32);
    let qd = in_place(dev, q)?;
    let kd = in_place(dev, k)?;
    let pos = dev.upload(tensor(case, "pos")?)?;
    let (t, h) = (rows as u32, hd as u32);
    match case.launcher.as_str() {
        "rope::rope_forward" => ops::rope(
            dev.gpu, kernel, qd, kd, pos, t, nq, nk, h, rot, theta, dev.stream,
        )?,
        "rope::rope_forward_strided" => ops::rope_strided(
            dev.gpu, kernel, qd, kd, pos, t, nq, nk, h, rot, theta, qw as u32, kw as u32,
            dev.stream,
        )?,
        other => return Err(not_runnable(format!("no rope launcher `{other}`"))),
    }
    let (qo, ko) = (dev.read(qd, q.bytes.len())?, dev.read(kd, k.bytes.len())?);
    let mut out = Vec::with_capacity(qo.len() + ko.len());
    for r in 0..rows {
        out.extend_from_slice(&qo[r * qw * 2..(r + 1) * qw * 2]);
        out.extend_from_slice(&ko[r * kw * 2..(r + 1) * kw * 2]);
    }
    Ok(out)
}
