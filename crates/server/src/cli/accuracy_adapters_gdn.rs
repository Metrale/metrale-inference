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

use anyhow::Result;
use metrale_accuracy::case::{Case, Enc, Tensor};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops;

use super::accuracy_adapters::{Adapter, at, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The adapter of each launcher.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    (
        "gated_delta_rule::gated_delta_rule_decode_f32_strided",
        gdn_strided,
    ),
    ("gated_delta_rule::gated_delta_rule_decode_f32", gdn_rows),
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
