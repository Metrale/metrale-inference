// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The emitters, one per emitter id of FUSIONS.toml, and the lookup by id. Each
//! calls the existing `ops::*` function its legacy call site calls, with the arguments that
//! site passes; the files say which site each mirrors.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - An emitter refuses a group whose ops, rows or bindings it was not written for; it never
//!   launches a guess.
//! - The only launch code here is the `ops::*` calls; no emitter builds a `KernelLaunch`.

use anyhow::{Context, Result, anyhow, ensure};

use super::bindings::{AttnFacts, MixerFacts, WeightSlot};
use super::compile::{Cx, OpEmitter};

mod attn;
mod batched;
mod ffn;
mod gdn;
mod gdn_verify;
mod linear;
mod norm;

static EMITTERS: [&dyn OpEmitter; 35] = [
    &norm::EmbedCopy,
    &norm::RmsNormResidual,
    &norm::ResidualAddRmsNorm,
    &norm::ResidualAddRmsNormExact,
    &norm::ResidualAdd,
    &norm::RmsNorm,
    &linear::W4a16DecodeGemv,
    &linear::W4a16GemvDual,
    &linear::W4a16GemvQg,
    &linear::SiluMul,
    &linear::LmHead,
    &gdn::DenseGemvBaGates,
    &gdn::Conv1dUpdateL2norm,
    &gdn::GdnDecode,
    &gdn::GatedRmsNorm,
    &attn::RopeMrope,
    &attn::KvWrite,
    &attn::PagedDecode,
    &attn::SigmoidGateMul,
    &attn::DeinterleaveQg,
    &attn::RmsNormStrided,
    &attn::RopeStrided,
    &batched::W4a16GemvBatchm,
    &batched::W4a16GemvBatch,
    &batched::W4a16GemvDualBatch,
    &batched::W4a16GemvQgBatch,
    &batched::W4a16GemmN128,
    &batched::W4a16Gemm,
    &batched::W4a16GemmN128M128,
    &ffn::DenseFfnKm,
    &ffn::DenseFfnMmq,
    &gdn_verify::DenseGemmBaGates,
    &gdn_verify::GdnDecodeWy,
    &linear::Argmax,
    &linear::HostSampling,
];

/// 2026-09-28: The emitter named `id`.
pub(crate) fn emitter(id: &str) -> Result<&'static dyn OpEmitter> {
    EMITTERS
        .iter()
        .copied()
        .find(|e| e.id() == id)
        .ok_or_else(|| anyhow!("no circuit emitter `{id}` is implemented yet"))
}

/// 2026-09-28: A circuit dim as a `u32`.
pub(super) fn dim(cx: &Cx<'_>, name: &str) -> Result<u32> {
    let v = *cx
        .g
        .circuit
        .dims
        .get(name)
        .with_context(|| format!("the circuit has no dim `{name}`"))?;
    u32::try_from(v).with_context(|| format!("dim `{name}` = {v} overflows u32"))
}

/// 2026-09-28: The rows of this plan as a `u32`.
pub(super) fn rows(cx: &Cx<'_>) -> Result<u32> {
    u32::try_from(cx.rows).context("rows overflow u32")
}

/// 2026-09-28: Refuse a plan wider than one row: the kernel is a single-row kernel.
pub(super) fn one_row(cx: &Cx<'_>, what: &str) -> Result<()> {
    ensure!(
        cx.rows == 1,
        "{what} is launched for one row; this plan has {} rows",
        cx.rows
    );
    Ok(())
}

/// 2026-09-28: The attention facts of member `i`'s layer.
pub(super) fn attn_facts(cx: &Cx<'_>, i: usize) -> Result<AttnFacts> {
    match cx.layer(i)?.mixer {
        MixerFacts::Attention(a) => Ok(a),
        MixerFacts::Gdn(_) => Err(anyhow!(
            "`{}` reads attention facts from a GatedDeltaNet layer",
            cx.g.node(i).id
        )),
    }
}

/// 2026-09-28: The norm weight slot of an `rms_norm`/`qk_norm` node, by its template id.
pub(super) fn norm_slot(cx: &Cx<'_>, i: usize) -> Result<WeightSlot> {
    let n = cx.g.node(i);
    Ok(match n.local.as_str() {
        "input_norm" => WeightSlot::InputNorm,
        "post_norm" => WeightSlot::PostNorm,
        "q_norm" => WeightSlot::QNorm,
        "k_norm" => WeightSlot::KNorm,
        other => {
            return Err(anyhow!(
                "`{}`: no norm weight for template id `{other}`",
                n.id
            ));
        }
    })
}

/// 2026-09-28: An unquantized weight; an error for any other format.
pub(super) fn dense(
    w: super::bindings::BoundWeight,
    what: &str,
) -> Result<crate::weight_map::DenseWeight> {
    match w {
        super::bindings::BoundWeight::Dense(d) => Ok(d),
        other => Err(anyhow!(
            "{what}: expected a dense weight, the layer holds {}",
            other.family()
        )),
    }
}

/// 2026-09-28: An NVFP4 weight; an error for any other format.
pub(super) fn nvfp4(
    w: super::bindings::BoundWeight,
    what: &str,
) -> Result<crate::weight_map::QuantizedWeight> {
    match w {
        super::bindings::BoundWeight::Nvfp4(q) => Ok(q),
        other => Err(anyhow!(
            "{what}: expected an NVFP4 weight, the layer holds {}",
            other.family()
        )),
    }
}

/// 2026-09-28: Refuse a group whose kernel is not `func` of the table's module.
pub(super) fn expect_kernel(cx: &Cx<'_>, k: usize, func: &str) -> Result<()> {
    let got = &cx
        .g
        .group
        .kernels
        .get(k)
        .with_context(|| format!("group {} has no kernel {k}", cx.g.index))?
        .func;
    ensure!(
        got == func,
        "group {} launches `{got}`; this emitter launches `{func}`",
        cx.g.index
    );
    Ok(())
}

/// 2026-09-28: Refuse a group that is neither launched once per row nor a single row: the
/// kernel serves one row (one sequence's state) per launch.
pub(super) fn per_row(cx: &Cx<'_>) -> Result<()> {
    ensure!(
        cx.reps() == cx.rows as usize,
        "group {} runs {} launches for {} rows; this kernel serves one row per launch",
        cx.g.index,
        cx.reps(),
        cx.rows
    );
    Ok(())
}
