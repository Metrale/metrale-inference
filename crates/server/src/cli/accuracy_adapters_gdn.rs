// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters of the gated delta rule decode kernels (`gdn_recurrence`
//! reference). The case's canonical q/k/v, decay/beta and state reach the kernels through the
//! engine's launchers (`ops::gdn_decode_f32_strided`, `ops::gdn_decode`), and the output is
//! each row's `o` followed by that row's new state, as the reference lays it out.
//!
//! Owner: server CLI.
//! Invariants:
//! - The strided kernel reads q/k/v from one `[rows, q|k|v]` buffer and decay/beta from one
//!   `[rows, decay|beta]` buffer, the layout the batched decode emitter hands it
//!   (model-layers circuit_exec/emitters/gdn_batched.rs), so its row strides are exercised.
//! - The state is updated in place inside a guarded buffer: a write outside it is a fault.
//! - The conv step (`conv1d_l2norm` reference) runs through `ops::conv1d_update_l2norm_strided`
//!   and `ops::conv1d_update_l2norm` as the GDN emitters launch them: the input rows are the
//!   projection's (q|k|v|z, wider than the output's), so a kernel that read the input at the
//!   output stride would read the wrong channels.

use anyhow::Result;
use metrale_accuracy::case::{Case, Enc, Tensor};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::quantized::DenseWeight;

use super::accuracy_adapters::{Adapter, at, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The adapter of each launcher.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    (
        "gated_delta_rule::gated_delta_rule_decode_f32_strided",
        gdn_strided,
    ),
    ("gated_delta_rule::gated_delta_rule_decode_f32", gdn_rows),
    (
        "causal_conv1d::causal_conv1d_update_l2norm_f32_strided",
        conv_strided,
    ),
    ("causal_conv1d::causal_conv1d_update_l2norm_f32", conv_rows),
];

/// 2026-10-09: The head geometry a gdn case records.
struct Geo {
    rows: usize,
    k_heads: usize,
    k_dim: usize,
    v_heads: usize,
    v_dim: usize,
}

impl Geo {
    fn of(case: &Case) -> Result<Geo> {
        let s = |n: &str| case.scalar(n).map(|v| v as usize).map_err(not_runnable);
        let g = Geo {
            rows: case.out.0[0],
            k_heads: s("k_heads")?,
            k_dim: s("k_dim")?,
            v_heads: s("v_heads")?,
            v_dim: s("v_dim")?,
        };
        let o_len = g.v_heads * g.v_dim;
        let s_len = o_len * g.k_dim;
        if case.out.0[1] != o_len + s_len || case.out.1 != Enc::F32 {
            return Err(not_runnable(format!(
                "a gdn output {:?} for {o_len} + {s_len} f32 values per row",
                case.out
            )));
        }
        Ok(g)
    }

    fn qk(&self) -> usize {
        self.k_heads * self.k_dim
    }

    fn o_len(&self) -> usize {
        self.v_heads * self.v_dim
    }

    fn s_len(&self) -> usize {
        self.o_len() * self.k_dim
    }
}

fn f32_tensor<'a>(case: &'a Case, name: &str) -> Result<&'a Tensor> {
    let t = case.tensor(name).map_err(not_runnable)?;
    if t.enc != Enc::F32 {
        return Err(not_runnable(format!(
            "`{name}` is {:?}; the kernel reads f32",
            t.enc
        )));
    }
    Ok(t)
}

/// 2026-10-09: The state, copied into a guarded output buffer the kernel updates in place.
fn state(dev: &mut Dev<'_>, case: &Case, g: &Geo) -> Result<DevicePtr> {
    let t = f32_tensor(case, "state")?;
    let bytes = g.rows * g.s_len() * 4;
    if t.bytes.len() != bytes {
        return Err(not_runnable(format!(
            "a state of {} bytes for {bytes}",
            t.bytes.len()
        )));
    }
    let p = dev.output(bytes)?;
    dev.gpu.copy_h2d(&t.bytes, p)?;
    Ok(p)
}

/// 2026-10-09: Row-major `[rows, a | b | ...]` from row-major parts of equal row count.
fn rows_of(parts: &[&Tensor], rows: usize) -> Tensor {
    let widths: Vec<usize> = parts.iter().map(|t| t.bytes.len() / rows).collect();
    let mut bytes = Vec::with_capacity(widths.iter().sum::<usize>() * rows);
    for r in 0..rows {
        for (t, w) in parts.iter().zip(&widths) {
            bytes.extend_from_slice(&t.bytes[r * w..(r + 1) * w]);
        }
    }
    Tensor {
        enc: Enc::F32,
        dims: vec![rows, widths.iter().sum::<usize>() / 4],
        bytes: std::sync::Arc::new(bytes),
    }
}

/// 2026-10-09: Read `o` and the state back and join them per row.
fn joined(dev: &mut Dev<'_>, g: &Geo, o: DevicePtr, s: DevicePtr) -> Result<Vec<u8>> {
    let (ob, sb) = (g.o_len() * 4, g.s_len() * 4);
    let o = dev.read(o, g.rows * ob)?;
    let s = dev.read(s, g.rows * sb)?;
    let mut out = Vec::with_capacity(g.rows * (ob + sb));
    for r in 0..g.rows {
        out.extend_from_slice(&o[r * ob..(r + 1) * ob]);
        out.extend_from_slice(&s[r * sb..(r + 1) * sb]);
    }
    Ok(out)
}

fn gdn_strided(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let g = Geo::of(case)?;
    let t = |n: &str| f32_tensor(case, n);
    let qkv = dev.upload(&rows_of(&[t("q")?, t("k")?, t("v")?], g.rows))?;
    let gates = dev.upload(&rows_of(&[t("gate")?, t("beta")?], g.rows))?;
    let st = state(dev, case, &g)?;
    let o = dev.output(g.rows * g.o_len() * 4)?;
    let qkv_stride = (2 * g.qk() + g.o_len()) as u32;
    ops::gdn_decode_f32_strided(
        dev.gpu,
        kernel,
        st,
        qkv,
        at(qkv, g.qk() * 4),
        at(qkv, 2 * g.qk() * 4),
        gates,
        at(gates, g.v_heads * 4),
        o,
        g.rows as u32,
        g.k_heads as u32,
        g.v_heads as u32,
        g.k_dim as u32,
        g.v_dim as u32,
        qkv_stride,
        qkv_stride,
        (2 * g.v_heads) as u32,
        g.o_len() as u32,
        dev.stream,
    )?;
    joined(dev, &g, o, st)
}

fn gdn_rows(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let g = Geo::of(case)?;
    let mut up = |n: &str| -> Result<DevicePtr> { dev.upload(f32_tensor(case, n)?) };
    let (q, k, v) = (up("q")?, up("k")?, up("v")?);
    let (gate, beta) = (up("gate")?, up("beta")?);
    let st = state(dev, case, &g)?;
    let o = dev.output(g.rows * g.o_len() * 4)?;
    ops::gdn_decode(
        dev.gpu,
        kernel,
        st,
        q,
        k,
        v,
        gate,
        beta,
        o,
        g.rows as u32,
        g.k_heads as u32,
        g.v_heads as u32,
        g.k_dim as u32,
        g.v_dim as u32,
        dev.stream,
    )?;
    joined(dev, &g, o, st)
}

/// 2026-10-09: The geometry and launch scalars a conv case records.
struct Conv {
    rows: usize,
    dim: usize,
    in_dim: usize,
    d_conv: usize,
    qk_channels: usize,
    head_dim: usize,
    eps: f32,
}

impl Conv {
    fn of(case: &Case) -> Result<Conv> {
        let s = |n: &str| case.scalar(n).map_err(not_runnable);
        let c = Conv {
            rows: case.out.0[0],
            dim: s("dim")? as usize,
            in_dim: s("in_dim")? as usize,
            d_conv: s("d_conv")? as usize,
            qk_channels: s("qk_channels")? as usize,
            head_dim: s("head_dim")? as usize,
            eps: s("eps")? as f32,
        };
        if case.out.0[1] != c.dim * (1 + c.d_conv) || case.out.1 != Enc::F32 {
            return Err(not_runnable(format!(
                "a conv output {:?} for {} channels of {} taps",
                case.out, c.dim, c.d_conv
            )));
        }
        let x = case.tensor("x").map_err(not_runnable)?;
        if x.dims != [c.rows, c.in_dim] || c.in_dim < c.dim {
            return Err(not_runnable(format!(
                "an input {:?} for {} rows of {} channels read from {}",
                x.dims, c.rows, c.dim, c.in_dim
            )));
        }
        for n in ["x", "w"] {
            let e = case.tensor(n).map_err(not_runnable)?.enc;
            if e != Enc::Bf16 {
                return Err(not_runnable(format!(
                    "`{n}` is {e:?}; the kernel reads bf16"
                )));
            }
        }
        Ok(c)
    }
}

/// 2026-10-09: The window, copied into a guarded output buffer the kernel updates in place.
fn window(dev: &mut Dev<'_>, case: &Case, c: &Conv) -> Result<DevicePtr> {
    let t = f32_tensor(case, "window")?;
    let bytes = c.rows * c.dim * c.d_conv * 4;
    if t.bytes.len() != bytes {
        return Err(not_runnable(format!(
            "a window of {} bytes for {bytes}",
            t.bytes.len()
        )));
    }
    let p = dev.output(bytes)?;
    dev.gpu.copy_h2d(&t.bytes, p)?;
    Ok(p)
}

/// 2026-10-09: Read the output and the window back and join them per row.
fn conv_joined(dev: &mut Dev<'_>, c: &Conv, y: DevicePtr, w: DevicePtr) -> Result<Vec<u8>> {
    let (yb, wb) = (c.dim * 4, c.dim * c.d_conv * 4);
    let y = dev.read(y, c.rows * yb)?;
    let w = dev.read(w, c.rows * wb)?;
    let mut out = Vec::with_capacity(c.rows * (yb + wb));
    for r in 0..c.rows {
        out.extend_from_slice(&y[r * yb..(r + 1) * yb]);
        out.extend_from_slice(&w[r * wb..(r + 1) * wb]);
    }
    Ok(out)
}

/// 2026-10-09: The batched conv as the batched GDN emitter launches it
/// (circuit_exec/emitters/gdn_batched.rs): every row in one launch, the input read at the
/// projection row's stride (`in_dim`), the output written at `dim`.
fn conv_strided(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let c = Conv::of(case)?;
    let input = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = DenseWeight {
        weight: dev.upload(case.tensor("w").map_err(not_runnable)?)?,
    };
    let st = window(dev, case, &c)?;
    let y = dev.output(c.rows * c.dim * 4)?;
    ops::conv1d_update_l2norm_strided(
        dev.gpu,
        kernel,
        st,
        input,
        &w,
        y,
        c.dim as u32,
        c.d_conv as u32,
        c.rows as u32,
        c.qk_channels as u32,
        c.head_dim as u32,
        c.eps,
        c.in_dim as u32,
        c.dim as u32,
        dev.stream,
    )?;
    conv_joined(dev, &c, y, st)
}

/// 2026-10-09: The per-sequence conv as the GDN emitter launches it
/// (circuit_exec/emitters/gdn.rs): one launch of batch 1 per row, at that row's input, output
/// and window (the kernel reads its input at `dim`, so a wider row needs the per-row pointer).
fn conv_rows(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let c = Conv::of(case)?;
    let input = dev.upload(case.tensor("x").map_err(not_runnable)?)?;
    let w = DenseWeight {
        weight: dev.upload(case.tensor("w").map_err(not_runnable)?)?,
    };
    let st = window(dev, case, &c)?;
    let y = dev.output(c.rows * c.dim * 4)?;
    for r in 0..c.rows {
        ops::conv1d_update_l2norm(
            dev.gpu,
            kernel,
            at(st, r * c.dim * c.d_conv * 4),
            at(input, r * c.in_dim * 2),
            &w,
            at(y, r * c.dim * 4),
            c.dim as u32,
            c.d_conv as u32,
            1,
            c.qk_channels as u32,
            c.head_dim as u32,
            c.eps,
            dev.stream,
        )?;
    }
    conv_joined(dev, &c, y, st)
}
