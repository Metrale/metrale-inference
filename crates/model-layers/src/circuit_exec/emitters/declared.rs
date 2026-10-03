// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The declared-precision emitters: `w8a8_act_quant` and `w8a8_gemv` (the W8A8 arm
//! of `w8a8_layer.rs`, which every decode path of a declared-W8A8 layer tries first), and
//! `w4a4_act_quant` and `w4a4_gemv` (`ops::w4a4_proj::nvfp4_proj_small_m` on its W4A4 path,
//! from `dense_ffn_decode_batch.rs` `forward_km`).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A quantized edge holds what the legacy scratch holds, packed in one buffer: W8A8
//!   `[rows, k]` E4M3 then the FP32 scales (`W8a8Scratch::packed`), W4A4 E2M1, E4M3 group
//!   scales and FP32 per-row globals (`Nvfp4ActBuf::packed`). The quantize group writes it and
//!   the projection groups read it.
//! - Every plan kernel is checked at compile time against the one the ops launcher runs for
//!   this shape (the W8A8 entry by rows, the W4A4 MX entry `mx_plan` picks); a W4A4 launch the
//!   tier would send to W4A16 is refused.

use anyhow::{Result, anyhow, bail, ensure};
use metrale_circuit::{Format, LinearRole, OpKind, Scale};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::bindings::{BoundWeight, MixerFacts, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::batched::width;
use super::{nvfp4, rows};
use crate::layers::ops::w4a4_proj::{Nvfp4ActBuf, W4a4Proj};
use crate::layers::ops::{self, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight};

/// 2026-09-30: The weight slot a projection of `role` binds.
fn slot_of(role: LinearRole) -> WeightSlot {
    match role {
        LinearRole::GateUp => WeightSlot::FfnGate,
        r => WeightSlot::Linear(r),
    }
}

/// 2026-09-30: The W8A8 scale layout an FP8 activation edge holds.
fn w8a8_scale(f: Format) -> Result<W8a8Scale> {
    match f {
        Format::Fp8E4m3 {
            scale: Scale::PerToken,
        } => Ok(W8a8Scale::PerRow),
        Format::Fp8E4m3 {
            scale: Scale::Group(128),
        } => Ok(W8a8Scale::Block128),
        other => bail!("{} is no W8A8 activation", other.name()),
    }
}

/// 2026-09-30: A W8A8 weight; an error for any other format.
fn w8a8(w: BoundWeight, what: &str) -> Result<(W8a8Weight, W8a8Kernels)> {
    match w {
        BoundWeight::W8a8(w, k) => Ok((w, k)),
        other => Err(anyhow!(
            "{what}: expected a W8A8 weight, the layer holds {}",
            other.family()
        )),
    }
}

/// 2026-09-30: The kernels, K and scale layout of every projection that reads `edge`, which
/// member `i`'s layer binds; they must agree.
fn readers(cx: &Cx<'_>, i: usize, edge: usize) -> Result<(W8a8Kernels, u32, W8a8Scale)> {
    let layer = cx.layer(i)?;
    let mut found = None;
    for &c in &cx.g.circuit.edges[edge].consumers {
        let n = &cx.g.circuit.nodes[c];
        let OpKind::Linear(role) = n.op else {
            bail!("`{}` reads a W8A8 activation and is no projection", n.id);
        };
        let w = *layer
            .weights
            .get(&slot_of(role))
            .ok_or_else(|| anyhow!("`{}`: its layer binds no weight", n.id))?;
        let (w, k) = w8a8(w, &n.id)?;
        let this = (k, w.k(), w.scale());
        match found {
            None => found = Some(this),
            Some((fk, fw, fs)) => ensure!(
                (fw, fs) == (this.1, this.2) && fk.quant(fs).0 == k.quant(fs).0,
                "the readers of one W8A8 activation disagree on its K, layout or kernels"
            ),
        }
    }
    found.ok_or_else(|| anyhow!("a W8A8 activation nothing reads"))
}

/// 2026-09-30: `w8a8_act_quant`: one BF16 activation, or `bf16(silu(gate) * up)` (the SiLU
/// member first), quantized per row into the E4M3 edge.
pub(crate) struct W8a8ActQuant;

impl OpEmitter for W8a8ActQuant {
    fn id(&self) -> &'static str {
        "w8a8_act_quant"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let silu = cx.g.group.nodes.len() == 2;
        if silu {
            cx.g.expect_ops(self.id(), &["silu_mul", "act_quant"])?;
        } else {
            cx.g.expect_ops(self.id(), &["act_quant"])?;
        }
        let last = usize::from(silu);
        let q = cx.g.output(last, 0)?;
        let scale = w8a8_scale(cx.g.circuit.edges[q].format)?;
        let (kernels, k, read_scale) = readers(cx, last, q)?;
        ensure!(
            read_scale == scale,
            "the edge and its readers disagree on the scale layout"
        );
        ensure!(
            width(cx, q)? == k,
            "the activation is not the readers' K = {k}"
        );
        let (h, m) = (cx.handle(0)?, rows(cx)? as usize);
        let want = if silu {
            kernels.quant_silu(scale)
        } else {
            kernels.quant(scale)
        };
        ensure!(
            h.0 != 0 && h.0 == want.0,
            "the plan's quantizer is not the one the W8A8 arm launches"
        );
        let scratch = W8a8Scratch::packed(cx.ptr(q)?, m, k, scale);
        if silu {
            let gu = cx.ptr(cx.g.input(0, 0)?)?;
            let up = gu.offset(m * k as usize * 2);
            cx.push(
                0,
                Box::new(move |e| {
                    ops::w8a8_act_quant_silu(
                        e.gpu, &kernels, scale, gu, up, k, m, k, &scratch, e.stream,
                    )
                }),
            )
        } else {
            let (x, ldx) = cx.strided(cx.g.input(0, 0)?)?;
            cx.push(
                0,
                Box::new(move |e| {
                    ops::w8a8_act_quant(e.gpu, &kernels, scale, x, ldx, m, k, &scratch, e.stream)
                }),
            )
        }
    }
}

/// 2026-09-30: `w8a8_gemv`: one W8A8 projection over the quantized edge, or the FFN's gate and
/// up (gate rows, then up rows) over one.
pub(crate) struct W8a8Gemv;

impl OpEmitter for W8a8Gemv {
    fn id(&self) -> &'static str {
        "w8a8_gemv"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        let OpKind::Linear(role) = cx.g.node(0).op else {
            bail!("not a projection");
        };
        if role == LinearRole::Qkvz {
            let MixerFacts::Gdn(f) = cx.layer(0)?.mixer else {
                bail!("a qkvz projection on a non-GDN layer");
            };
            ensure!(
                f.qkvz_deinterleaved,
                "a W8A8 QKV|Z stacks the sequential layout only"
            );
        }
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        let m = rows(cx)? as usize;
        // 2026-09-30: `(slot, output, N, row pitch)`: gate and up are two planar `[m, inter]`
        // blocks of the gate+up edge (see `ffn.rs`); any other output takes its row stride.
        let targets: Vec<(WeightSlot, DevicePtr, u32, u32)> = if role == LinearRole::GateUp {
            let (y, inter) = (cx.ptr(out)?, width(cx, out)? / 2);
            let up = y.offset(m * inter as usize * 2);
            vec![
                (WeightSlot::FfnGate, y, inter, inter),
                (WeightSlot::FfnUp, up, inter, inter),
            ]
        } else {
            let (y, ldc) = cx.strided(out)?;
            vec![(WeightSlot::Linear(role), y, width(cx, out)?, ldc)]
        };
        ensure!(
            cx.g.group.kernels.len() == targets.len(),
            "one kernel per projection"
        );
        let scale = w8a8_scale(cx.g.circuit.edges[inp].format)?;
        let (x, k) = (cx.ptr(inp)?, width(cx, inp)?);
        for (j, (slot, y, n, ld)) in targets.into_iter().enumerate() {
            let (w, kernels) = w8a8(cx.weight(0, slot)?, role.name())?;
            ensure!(
                w.scale() == scale && (w.n(), w.k()) == (n, k),
                "the weight [{}, {}] {:?} is not this projection's [{n}, {k}] {scale:?}",
                w.n(),
                w.k(),
                w.scale()
            );
            let scratch = W8a8Scratch::packed(x, m, k, scale);
            ensure!(
                ops::w8a8_decode_available(&kernels, &w, m, &scratch),
                "W8A8 does not serve {m} rows of this projection"
            );
            let h = cx.handle(j)?;
            ensure!(
                kernels.gemv_entry(scale, m).is_some_and(|e| e.0 == h.0),
                "the plan's GEMV entry is not the one w8a8_gemv launches for {m} rows"
            );
            cx.push(
                j,
                Box::new(move |e| {
                    ops::w8a8_gemv(e.gpu, &kernels, &w, &scratch, m, y, ld, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-30: `w4a4_act_quant`: `w4a4_quant_rows` of one BF16 activation into the NVFP4 edge.
pub(crate) struct W4a4ActQuant;

impl OpEmitter for W4a4ActQuant {
    fn id(&self) -> &'static str {
        "w4a4_act_quant"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["act_quant:nvfp4/g16"])?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        let (m, k) = (rows(cx)?, width(cx, out)?);
        ensure!(
            k.is_multiple_of(128),
            "K = {k}: the packed NVFP4 parts need K % 128 == 0"
        );
        let p = W4a4Proj::prepared(cx.gpu)
            .ok_or_else(|| anyhow!("the W4A4 kernels are not prepared on this backend"))?;
        let h = cx.handle(0)?;
        ensure!(
            h.0 == p.quant_kernel().0,
            "the plan's quantizer is not w4a4_quant_rows"
        );
        let (x, act) = (cx.ptr(inp)?, Nvfp4ActBuf::packed(cx.ptr(out)?, m, k));
        cx.push(
            0,
            Box::new(move |e| p.quantize(e.gpu, x, act, m, k, e.stream)),
        )
    }
}

/// 2026-09-30: `w4a4_gemv`: the MX GEMV of one projection (down), or of gate and up (gate rows,
/// then up rows), over the NVFP4 edge.
pub(crate) struct W4a4Gemv;

impl OpEmitter for W4a4Gemv {
    fn id(&self) -> &'static str {
        "w4a4_gemv"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        let OpKind::Linear(role) = cx.g.node(0).op else {
            bail!("not a projection");
        };
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        let (m, k) = (rows(cx)?, width(cx, inp)?);
        let y = cx.ptr(out)?;
        let targets: Vec<(WeightSlot, DevicePtr, u32)> = match role {
            LinearRole::GateUp => {
                let inter = width(cx, out)? / 2;
                let up = y.offset(m as usize * inter as usize * 2);
                vec![
                    (WeightSlot::FfnGate, y, inter),
                    (WeightSlot::FfnUp, up, inter),
                ]
            }
            LinearRole::Down => vec![(WeightSlot::Linear(role), y, width(cx, out)?)],
            other => bail!("the W4A4 rules run the FFN only, not {}", other.name()),
        };
        ensure!(
            cx.g.group.kernels.len() == targets.len(),
            "one kernel per projection"
        );
        let act = Nvfp4ActBuf::packed(cx.ptr(inp)?, m, k);
        for (j, (slot, y, n)) in targets.into_iter().enumerate() {
            let w = nvfp4(cx.weight(0, slot)?, role.name())?;
            let p = W4a4Proj::for_launch(cx.gpu, &w, m, n, k).ok_or_else(|| {
                anyhow!(
                    "{m}x{n}x{k} {} runs W4A16 under this tier; the plan runs W4A4",
                    role.name()
                )
            })?;
            let h = cx.handle(j)?;
            ensure!(
                h.0 == p.mx_kernel(m, n, k).0,
                "the plan's MX entry is not the one mx_plan picks for {m}x{n}x{k}"
            );
            cx.push(
                j,
                Box::new(move |e| p.gemv(e.gpu, act, &w, y, m, n, k, e.stream)),
            )?;
        }
        Ok(())
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[&W8a8ActQuant, &W8a8Gemv, &W4a4ActQuant, &W4a4Gemv];
