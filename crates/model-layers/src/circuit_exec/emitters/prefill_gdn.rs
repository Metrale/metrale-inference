// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The GatedDeltaNet emitters of the prefill modes (LIFECYCLE-DESIGN.md 15.4). Each
//! mirrors the legacy prefill call site it names: the same `ops::*` function, the same arguments,
//! the row count from the step (`StepEnv::prefill`). Legacy sites: `qwen3_ssm/trait_prefill.rs`
//! (the input norm, the post-mixer add and norm, the synchronizes above 4096 rows) and
//! `qwen3_ssm/trait_prefill_block.rs` (the block body); the core group is
//! `prefill_gdn_core.rs`.
//!
//! The input-norm and post-norm rules this module serves match GatedDeltaNet layers only (their
//! patterns name `linear_attention`); the attention layer's pair is `prefill_attn_input_norm` and
//! `prefill_attn_add_post_norm` (`prefill_attn.rs`), which issue no synchronize.
//!
//! The FP8 projections read `WeightSlot::PrefillCast`: an unscaled E4M3 cast of the BF16
//! dequantization of the checkpoint's weight (`qwen35_dense/gdn_dequant.rs:362-389`), which the
//! legacy load line calls "native FP8 prefill GEMM"; the GEMM takes no weight scale.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A launch whose `ops::*` function issues several kernels in order (the FP8 cast and GEMM,
//!   the conv and its state write, the chunked recurrence) is one bundle covering those plan
//!   kernels ([`push_bundle`], `Cx::push_bundle`), so the plan's launch count is kept and the
//!   real launches are the legacy call's, in its order.
//! - A route switch this module does not model, set in the environment, refuses the build
//!   ([`refuse_switches`]); nothing here picks a kernel the plan did not name.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::LinearRole;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::bindings::{BoundWeight, MixerFacts, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::super::program::RunFn;
use super::{dense, norm_slot};
use crate::layers::ops;

#[path = "prefill_gdn_core.rs"]
mod gdn_core;

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &PrefillRmsNormResidual,
    &PrefillResidualAddRmsNorm,
    &PrefillGdnFp8Proj,
    &gdn_core::PrefillGdnCore,
];

/// 2026-10-03: Rows above which a GatedDeltaNet prefill synchronizes at the layer's entry and
/// after its input norm (`qwen3_ssm/trait_prefill.rs:63-68, 108-112`).
const SYNC_ABOVE_ROWS: u32 = 4096;

/// 2026-10-03: Environment switches that move a GatedDeltaNet prefill launch off the route the
/// rules encode: `(variable, value that moves it, or None for any value, what it does)`. The
/// executor's lever table is the single classification once it covers prefill; until then this
/// module refuses them at build.
const SWITCHES: &[(&str, Option<&str>, &str)] = &[
    ("METRALE_GDN_BF16_WEIGHTS", Some("1"), "BF16 GDN projections on cuBLASLt"),
    ("METRALE_CUTLASS_NVFP4_GEMM", Some("1"), "CUTLASS NVFP4 projections"),
    ("METRALE_CUTLASS_NVFP4_QKVZ", Some("1"), "the CUTLASS NVFP4 qkvz projection"),
    ("METRALE_CUTLASS_NVFP4_SSM_OUT", Some("1"), "the CUTLASS NVFP4 out_proj"),
    ("METRALE_FP8_LDMAB", Some("0"), "the FP8 GEMM without the ldmab kernel"),
    ("METRALE_CONV1D_TP", Some("0"), "the sequential prefill conv"),
    ("METRALE_NO_GDN_FLA", Some("1"), "the recurrence without the chunked FLA arm"),
    ("METRALE_GDN_PIPE", Some("0"), "the vfused FLA state spine"),
    ("METRALE_GDN_VTILE", None, "the vtile FLA state spine, or none"),
    ("METRALE_GDN_TMA", Some("1"), "the TMA FLA state spine"),
    ("METRALE_NO_GDN_FWD_O_MMA8", None, "the FLA output kernel without its 8-warp twin"),
];

/// 2026-10-03: Refuse the build when a switch of [`SWITCHES`] named in `vars` is set.
pub(super) fn refuse_switches(vars: &[&str]) -> Result<()> {
    match switch_refusal(vars, |v| std::env::var(v).ok()) {
        Some(why) => bail!("{why}"),
        None => Ok(()),
    }
}

/// 2026-10-03: The first switch of [`SWITCHES`] named in `vars` whose value under `value_of`
/// moves the route, as a refusal; `None` when none does. Pure, so the table is testable without
/// the process environment.
fn switch_refusal(vars: &[&str], value_of: impl Fn(&str) -> Option<String>) -> Option<String> {
    SWITCHES
        .iter()
        .filter(|(v, _, _)| vars.contains(v))
        .find_map(|(var, moves, what)| {
            let value = value_of(var)?;
            (moves.is_none_or(|m| m == value)).then(|| {
                format!("{var}={value} selects {what}, which the prefill rules do not model")
            })
        })
}

/// 2026-10-03: Push `run` as one launch issuing kernel `first` and the `extra` kernels after it
/// that the same `ops::*` call issues (`Cx::push_bundle`).
pub(super) fn push_bundle(cx: &mut Cx<'_>, first: usize, extra: usize, run: RunFn) -> Result<()> {
    cx.push_bundle(first, extra + 1, run)
}

/// 2026-10-03: The layer of member `i`, which a prefill GatedDeltaNet launch reads its state
/// from (`StepEnv::gdn_state(layer, 0)`: one sequence).
pub(super) fn layer_of(cx: &Cx<'_>, i: usize) -> Result<usize> {
    let n = cx.g.node(i);
    n.layer
        .with_context(|| format!("`{}` is outside the layers", n.id))
}

/// 2026-10-03: Whether member `i`'s layer is a GatedDeltaNet layer.
fn is_gdn(cx: &Cx<'_>, i: usize) -> Result<bool> {
    Ok(matches!(cx.layer(i)?.mixer, MixerFacts::Gdn(_)))
}

/// 2026-10-03: The unscaled E4M3 cast a projection's prefill GEMM reads (`qkvz_fp8`,
/// `out_proj_fp8`), from the layer's binding. Read without the node-format check: the cast is a
/// derived serving copy of the checkpoint's NVFP4 weight, not the weight the node declares.
fn fp8_cast(cx: &Cx<'_>, i: usize, role: LinearRole) -> Result<DevicePtr> {
    let slot = WeightSlot::PrefillCast(role);
    match cx.layer(i)?.weights.get(&slot) {
        Some(BoundWeight::Dense(d)) => Ok(d.weight),
        Some(other) => bail!("{slot:?}: expected the E4M3 cast, the layer holds {}", other.family()),
        None => bail!(
            "`{}`: its layer binds no {slot:?}; the legacy prefill takes another arm \
             (`qwen3_ssm/trait_prefill_proj.rs`, `trait_prefill_helper.rs`)",
            cx.g.node(i).id
        ),
    }
}

/// 2026-10-03: `prefill_rms_norm_residual`: a mixer's input norm over the pass's rows into
/// `norm_output`, copying the stream into `residual` (`qwen3_ssm/trait_prefill.rs:84-95`,
/// `qwen3_attention/trait_impl/prefill_inner.rs:86`). A GatedDeltaNet layer synchronizes
/// before and after it above 4096 rows.
pub(crate) struct PrefillRmsNormResidual;

impl OpEmitter for PrefillRmsNormResidual {
    fn id(&self) -> &'static str {
        "prefill_rms_norm_residual"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["rms_norm"])?;
        let k = cx.handle(0)?;
        let (x, xn) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.output(0, 0)?)?);
        let w = dense(cx.weight(0, norm_slot(cx, 0)?)?, "input norm")?;
        let (res, h) = (cx.fixed.residual, cx.config.hidden_size as u32);
        let eps = cx.config.rms_norm_eps as f32;
        let sync = is_gdn(cx, 0)?;
        cx.push(
            0,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                let big = sync && t > SYNC_ABOVE_ROWS;
                if big {
                    e.gpu.synchronize(e.stream)?;
                }
                ops::rms_norm_residual(e.gpu, k, x, &w, xn, res, t, h, eps, e.stream)?;
                if big {
                    e.gpu.synchronize(e.stream)?;
                }
                Ok(())
            }),
        )
    }
}

/// 2026-10-03: `prefill_residual_add_rms_norm`: `hidden += o`, then the FFN's input norm into
/// its edge, copying the stream into `residual` (`qwen3_ssm/trait_prefill.rs:169-180`,
/// `qwen3_attention/trait_impl/prefill_inner.rs:302`).
pub(crate) struct PrefillResidualAddRmsNorm;

impl OpEmitter for PrefillResidualAddRmsNorm {
    fn id(&self) -> &'static str {
        "prefill_residual_add_rms_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["residual_add", "rms_norm"])?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        ensure!(
            x == cx.ptr(cx.g.output(0, 0)?)?,
            "group {}: the residual stream must be updated in place",
            cx.g.index
        );
        let k = cx.handle(0)?;
        let src = cx.ptr(cx.g.input(0, 1)?)?;
        let xn = cx.ptr(cx.g.output(1, 0)?)?;
        let w = dense(cx.weight(1, norm_slot(cx, 1)?)?, "post norm")?;
        let (res, h) = (cx.fixed.residual, cx.config.hidden_size as u32);
        let eps = cx.config.rms_norm_eps as f32;
        cx.push(
            0,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::residual_add_rms_norm(e.gpu, k, x, src, &w, xn, res, t, h, eps, e.stream)
            }),
        )
    }
}

/// 2026-10-03: `prefill_gdn_fp8_proj`: the qkvz or out_proj projection over the unscaled E4M3
/// cast of its NVFP4 weight, `ops::fp8_gemm_n128`, which casts the activation to E4M3
/// (`w4a16::bf16_to_fp8`) and runs `w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab`
/// (`qwen3_ssm/trait_prefill_proj.rs:274-288`, `trait_prefill_helper.rs:207-238`). The qkvz
/// projection writes `[Q | K | V | Z]` in order, so its layer must be a sequential-qkvz one.
pub(crate) struct PrefillGdnFp8Proj;

impl OpEmitter for PrefillGdnFp8Proj {
    fn id(&self) -> &'static str {
        "prefill_gdn_fp8_proj"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let role = match cx.g.node(0).op {
            metrale_circuit::OpKind::Linear(r @ (LinearRole::Qkvz | LinearRole::GdnOut)) => r,
            other => bail!("`{}` cannot launch {}", self.id(), other.name()),
        };
        cx.g.expect_ops(self.id(), &["linear"])?;
        super::expect_kernel(cx, 0, "bf16_to_fp8")?;
        super::expect_kernel(cx, 1, "fp8_fp8_gemm_ldmab")?;
        refuse_switches(&[
            "METRALE_GDN_BF16_WEIGHTS",
            "METRALE_CUTLASS_NVFP4_GEMM",
            "METRALE_CUTLASS_NVFP4_QKVZ",
            "METRALE_CUTLASS_NVFP4_SSM_OUT",
            "METRALE_FP8_LDMAB",
        ])?;
        let MixerFacts::Gdn(facts) = cx.layer(0)?.mixer else {
            bail!("`{}` reads a GatedDeltaNet projection", self.id());
        };
        let c = cx.config;
        let (n, k) = match role {
            LinearRole::Qkvz => {
                ensure!(
                    facts.qkvz_deinterleaved,
                    "the qkvz projection of an interleaved layer is followed by a deinterleave the \
                     prefill rules do not model"
                );
                (c.ssm_qkvz_size() as u32, c.hidden_size as u32)
            }
            _ => (
                c.hidden_size as u32,
                (c.linear_num_value_heads * c.linear_value_head_dim) as u32,
            ),
        };
        // 2026-10-03: `fp8_gemm_n128` takes the ldmab route only for K a multiple of 32; the
        // out_proj arm of a K that is not runs `fp8_gemm_n128_m128` above 128 rows instead.
        ensure!(
            k.is_multiple_of(32),
            "K = {k} is not a multiple of 32: the FP8 GEMM would not take the ldmab kernel"
        );
        let w = fp8_cast(cx, 0, role)?;
        let (x, y) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.output(0, 0)?)?);
        // 2026-10-03: The fallback kernel the legacy layer hands `fp8_gemm_n128` (`init.rs:400`);
        // with the ldmab route it only names the activation check of a debug build.
        let fallback = cx.gpu.kernel("w4a16", "fp8_gemm_t")?;
        push_bundle(
            cx,
            0,
            1,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::fp8_gemm_n128(e.gpu, fallback, x, w, y, t, n, k, e.stream)
            }),
        )
    }
}

/// 2026-10-03: The kernel handle a `try_*_kernel` lookup returned, or an error naming it.
pub(super) fn present(k: KernelHandle, what: &str) -> Result<KernelHandle> {
    ensure!(k.0 != 0, "{what} is not loaded on this target");
    Ok(k)
}

/// 2026-10-03: The GatedDeltaNet dims of the legacy layer, from the config as it reads them.
pub(super) struct GdnDims {
    pub nk: usize,
    pub kd: usize,
    pub nv: usize,
    pub vd: usize,
}

impl GdnDims {
    pub(super) fn of(c: &metrale_config::ModelConfig) -> Self {
        Self {
            nk: c.linear_num_key_heads,
            kd: c.linear_key_head_dim,
            nv: c.linear_num_value_heads,
            vd: c.linear_value_head_dim,
        }
    }

    pub(super) fn key_dim(&self) -> usize {
        self.nk * self.kd
    }

    pub(super) fn value_dim(&self) -> usize {
        self.nv * self.vd
    }

    pub(super) fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

#[cfg(test)]
#[path = "prefill_gdn_tests.rs"]
mod prefill_gdn_tests;
