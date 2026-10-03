// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The GatedDeltaNet emitters of the exact MTP verify chain (`gdn_verify_exact`:
//! `--exact-verify`, or a fixed GDN activation format), mirroring
//! `qwen3_ssm/trait_decode_batched_conv_gdn_exact_chain.rs` for one sequence and, through the
//! submodule, `trait_decode_batched_conv_gdn_multi.rs` with `..._multi_exact.rs` for a batched
//! verify. Each verify row gets the bits a single-token decode step computes.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The conv chain writes the window after each row but the last into the step's conv rollback
//!   slots and the exact chain the h state after each row but the last into its h rollback
//!   slots, inline; a row that writes none gets the state pointer, which the kernels never write
//!   through, as the legacy chain passes.
//! - A group is refused at build wherever the legacy chain declines and a per-row arm the
//!   circuit does not model runs instead: a sigmoid-gated norm (no strided FP32 norm), the fused
//!   output norm, or a parent kernel (FP32 conv, FP32 decode, FP32-input norm, its strided twin)
//!   absent. A missing twin is refused by the kernel table. A step without the rollback slots
//!   fails at launch.

use anyhow::{Context, Result, ensure};
use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::bindings::WeightSlot;
use super::super::compile::{Cx, OpEmitter};
use super::super::program::MAX_VERIFY_STEPS;
use super::{dense, dim, expect_kernel};
use crate::layers::ops;

#[path = "gdn_exact_batch.rs"]
mod batch;

/// 2026-10-03: The kernels the legacy chain requires linked besides its twins
/// (`decode_batched_conv_gdn_exact_chain`'s `conv1d_l2norm_f32_k`, `gdn_f32_k`,
/// `gated_rms_norm_f32_k`, `gated_rms_norm_f32_strided_k`, looked up in `qwen3_ssm/init.rs`).
const PARENTS: [(&str, &str); 4] = [
    ("causal_conv1d", "causal_conv1d_update_l2norm_f32"),
    ("gated_delta_rule", "gated_delta_rule_decode_f32"),
    ("norm", "gated_rms_norm_f32_input"),
    ("norm", "gated_rms_norm_f32_input_strided"),
];

/// 2026-10-03: Refuse a build where the legacy chain declines (see the module header).
pub(super) fn chain_ready(cx: &Cx<'_>, what: &str) -> Result<()> {
    ensure!(
        !cx.config.gdn_norm_sigmoid,
        "{what}: a sigmoid-gated GDN norm has no strided FP32 norm, so the legacy chain declines \
         to a per-row arm no rule models"
    );
    ensure!(
        !crate::layers::qwen3_ssm::gdn_fused_norm_enabled(),
        "{what}: under the fused GDN output norm the legacy chain declines to a per-row arm no \
         rule models"
    );
    for (module, func) in PARENTS {
        ensure!(
            crate::layers::try_kernel(cx.gpu, module, func).0 != 0,
            "{what}: `{module}::{func}` is not linked, so the legacy chain declines to a per-row \
             arm no rule models"
        );
    }
    Ok(())
}

/// 2026-10-03: The verify width `K` of a single-sequence chain group: its rows, 2..=4.
fn chain_width(cx: &Cx<'_>, what: &str) -> Result<usize> {
    let k = cx.rows as usize;
    ensure!(
        cx.mode == Mode::Verify && (2..=MAX_VERIFY_STEPS + 1).contains(&k),
        "{what} serves a single-sequence verify of 2..=4 rows, not {} rows of a {:?} plan",
        k,
        cx.mode
    );
    Ok(k)
}

/// 2026-10-03: The pointers rows `0..k-1` write through: the step's rollback slots for every
/// row but the last, the state itself for the rest (`inter` in the legacy chain).
pub(super) fn inline_steps(
    layer: usize,
    what: &str,
    steps: [DevicePtr; MAX_VERIFY_STEPS],
    state: DevicePtr,
    k: usize,
) -> Result<[DevicePtr; 3]> {
    ensure!(
        steps[..k - 1].iter().all(|p| !p.is_null()),
        "layer {layer} lacks the {what} rollback slots of a {k}-row verify"
    );
    Ok(std::array::from_fn(|t| {
        if t + 1 < k { steps[t] } else { state }
    }))
}

/// 2026-10-03: The GDN dims `(nk, kd, nv, vd)`.
fn gdn_dims(cx: &Cx<'_>) -> Result<(u32, u32, u32, u32)> {
    Ok((
        dim(cx, "lin_k_heads")?,
        dim(cx, "lin_k_dim")?,
        dim(cx, "lin_v_heads")?,
        dim(cx, "lin_v_dim")?,
    ))
}

fn gdn_layer(cx: &Cx<'_>) -> Result<usize> {
    cx.g.node(0).layer.context("a GDN node outside the layers")
}

/// 2026-10-03: `gdn_exact_conv_chain`: `gdn_conv_chain_f32`, the FP32 conv with SiLU and the
/// Q/K L2 norm for every row of one sequence, one launch, the windows after rows `0..K-2`
/// written inline (`decode_batched_conv_gdn_exact_chain`, its first launch).
pub(crate) struct GdnExactConvChain;

impl OpEmitter for GdnExactConvChain {
    fn id(&self) -> &'static str {
        "gdn_exact_conv_chain"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["conv1d_update", "state_snapshot", "l2_norm"])?;
        expect_kernel(cx, 0, "gdn_conv_chain_f32")?;
        chain_ready(cx, self.id())?;
        let k = chain_width(cx, self.id())?;
        ensure!(
            cx.g.group.copies.is_none(),
            "the conv chain writes its windows inline; it takes no copies"
        );
        let (nk, kd, nv, vd) = gdn_dims(cx)?;
        let key = nk * kd;
        let conv_dim = key * 2 + nv * vd;
        let w = dense(cx.weight(0, WeightSlot::GdnConv1d)?, "conv1d")?;
        let d_conv = u32::try_from(cx.config.linear_conv_kernel_dim)?;
        let (qkvz, in_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (out, out_stride) = cx.strided(cx.g.output(2, 0)?)?;
        ensure!(
            out_stride >= conv_dim,
            "the FP32 conv rows are {out_stride} wide; the chain writes {conv_dim}"
        );
        let (layer, h) = (gdn_layer(cx)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| {
                let st = e.gdn_state(layer, 0)?;
                let inter = inline_steps(layer, "conv", st.conv_steps, st.conv, k)?;
                ops::gdn_conv_chain_f32(
                    e.gpu,
                    h,
                    st.conv,
                    qkvz,
                    &w,
                    out,
                    inter,
                    k as u32,
                    conv_dim,
                    d_conv,
                    key * 2,
                    kd,
                    1e-6,
                    in_stride,
                    out_stride,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: `gdn_exact_chain`: `gdn_exact_chain{K}`, the strided FP32 decode's chain over the
/// `K` rows of one sequence with the h state read once, the states after rows `0..K-2` written
/// inline (`decode_batched_conv_gdn_exact_chain`, its second launch).
pub(crate) struct GdnExactChain;

impl OpEmitter for GdnExactChain {
    fn id(&self) -> &'static str {
        "gdn_exact_chain"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["gdn_recurrence"])?;
        chain_ready(cx, self.id())?;
        let k = chain_width(cx, self.id())?;
        expect_kernel(cx, 0, &format!("gdn_exact_chain{k}"))?;
        let (nk, kd, nv, _) = gdn_dims(cx)?;
        let key = (nk * kd) as usize * 4;
        let (qkv, qk_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (q, kk, v) = (qkv, qkv.offset(key), qkv.offset(key * 2));
        let (gate, gb_stride) = cx.strided(cx.g.input(0, 1)?)?;
        let (beta, beta_stride) = cx.strided(cx.g.input(0, 2)?)?;
        ensure!(
            beta == gate.offset(nv as usize * 4) && beta_stride == gb_stride,
            "decay and beta do not share rows"
        );
        let (out, out_stride) = cx.strided(cx.g.output(0, 0)?)?;
        let (layer, h) = (gdn_layer(cx)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| {
                let st = e.gdn_state(layer, 0)?;
                let inter = inline_steps(layer, "h", st.h_steps, st.h, k)?;
                ops::gdn_exact_chain(
                    e.gpu,
                    h,
                    st.h,
                    q,
                    kk,
                    v,
                    gate,
                    beta,
                    out,
                    inter,
                    nk,
                    nv,
                    kd,
                    [qk_stride, qk_stride, gb_stride, out_stride],
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &GdnExactConvChain,
    &GdnExactChain,
    &batch::GdnVerifyRunsExact,
];
