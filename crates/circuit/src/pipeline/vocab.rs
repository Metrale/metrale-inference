// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The step vocabulary: which steps each op's pipeline has, in order, and the shape
//! and spelling of each step's value. The one table every requirement, declaration and
//! rendering reads.
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - [`steps_of`] matches every [`OpKind`] without a wildcard arm, so a new op does not compile
//!   until its pipeline shape is defined.
//! - Projections (a linear of any role, the LM head, the router and both expert projections)
//!   share one shape; the routed gate/up projection adds the row gather by expert id.

use super::{Num, StepKind, Value};
use crate::format::Format;
use crate::ir::{LinearRole, OpKind};
use crate::state::StateDtype;

const PROJECTION: &[StepKind] = &[
    StepKind::Act,
    StepKind::Weight,
    StepKind::Mma,
    StepKind::Accumulate,
    StepKind::Scale,
];
const GATHERED_PROJECTION: &[StepKind] = &[
    StepKind::Gather,
    StepKind::Act,
    StepKind::Weight,
    StepKind::Mma,
    StepKind::Accumulate,
    StepKind::Scale,
];
const COMPUTE: &[StepKind] = &[StepKind::Compute];
const RECURRENT: &[StepKind] = &[StepKind::State, StepKind::Compute];
const ATTENTION: &[StepKind] = &[
    StepKind::Cache,
    StepKind::Scores,
    StepKind::Softmax,
    StepKind::Accumulate,
];
/// 2026-10-08: Latent attention: the query absorbed through `kv_b_proj` (a projection's act and
/// weight), then attention over the latent cache.
const LATENT_ATTENTION: &[StepKind] = &[
    StepKind::Act,
    StepKind::Weight,
    StepKind::Cache,
    StepKind::Scores,
    StepKind::Softmax,
    StepKind::Accumulate,
];

/// 2026-10-02: The steps of `op`'s pipeline, in order.
pub fn steps_of(op: &OpKind) -> &'static [StepKind] {
    match op {
        OpKind::Linear(_) | OpKind::LmHead | OpKind::Router | OpKind::ExpertDown => PROJECTION,
        OpKind::ExpertGateUp => GATHERED_PROJECTION,
        OpKind::Embed => &[StepKind::Gather],
        OpKind::RmsNorm
        | OpKind::FinalNorm
        | OpKind::QkNorm
        | OpKind::GatedRmsNorm
        | OpKind::L2Norm
        | OpKind::ResidualAdd
        | OpKind::SiluMul
        | OpKind::Relu2
        | OpKind::SigmoidGateMul
        | OpKind::GdnGates
        | OpKind::Rope
        | OpKind::ActQuant(_)
        | OpKind::SwigluClamp
        | OpKind::LayerNorm
        | OpKind::HcPre
        | OpKind::HcPost
        | OpKind::HcContract
        | OpKind::GeluTanhMul
        | OpKind::ScalarMul
        | OpKind::LogitSoftcap => COMPUTE,
        OpKind::Copy | OpKind::Concat | OpKind::Split | OpKind::HcExpand => &[StepKind::Move],
        OpKind::KvWrite => &[StepKind::Cache],
        // 2026-10-10: DeepSeek-V4's shared-KV attention: the window and compressed rows read
        // from their caches, scored, softmaxed with the sink and accumulated.
        OpKind::PagedAttention | OpKind::CompressedAttention => ATTENTION,
        OpKind::MlaAttention => LATENT_ATTENTION,
        // 2026-10-08: The pool tail is the state the compression updates; the selection's
        // scores and top-k run at the reference FP32.
        OpKind::KpoolCompress => RECURRENT,
        OpKind::IndexSelect => &[StepKind::Scores, StepKind::Score],
        OpKind::Conv1dUpdate | OpKind::GdnRecurrence | OpKind::SsmUpdate => RECURRENT,
        OpKind::StateSnapshot => &[StepKind::State],
        OpKind::TopK => &[StepKind::Score],
        OpKind::Blend => &[StepKind::Scatter, StepKind::Combine],
        OpKind::EpReduce => &[StepKind::Reduce],
        OpKind::Argmax => &[StepKind::Compare],
    }
}

/// 2026-10-02: The steps of the op a manifest names by base name (`linear`, `act_quant`,
/// `rms_norm`, ...); `None` for a name outside the vocabulary.
pub fn steps_for_base(base: &str) -> Option<&'static [StepKind]> {
    let op = match base {
        "linear" => OpKind::Linear(LinearRole::Q),
        "act_quant" => OpKind::ActQuant(Format::Bf16),
        other => OpKind::parse(other, None, None).ok()?,
    };
    Some(steps_of(&op))
}

/// 2026-10-02: The shape of a step's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// 2026-10-02: [`Value::Format`].
    Format,
    /// 2026-10-02: [`Value::Weight`].
    Weight,
    /// 2026-10-02: [`Value::Mma`].
    Mma,
    /// 2026-10-02: [`Value::Num`].
    Num,
    /// 2026-10-02: [`Value::OptNum`].
    OptNum,
    /// 2026-10-02: [`Value::Kv`].
    Kv,
    /// 2026-10-02: [`Value::State`].
    State,
}

const STEPS: [(StepKind, &str, Shape); 17] = [
    (StepKind::Gather, "gather", Shape::Format),
    (StepKind::Act, "act", Shape::Format),
    (StepKind::Weight, "weight", Shape::Weight),
    (StepKind::Mma, "mma", Shape::Mma),
    (StepKind::Accumulate, "accumulate", Shape::Num),
    (StepKind::Scale, "scale", Shape::OptNum),
    (StepKind::Compute, "compute", Shape::Num),
    (StepKind::Move, "move", Shape::Format),
    (StepKind::Cache, "cache", Shape::Kv),
    (StepKind::Scores, "scores", Shape::Num),
    (StepKind::Softmax, "softmax", Shape::Num),
    (StepKind::State, "state", Shape::State),
    (StepKind::Score, "score", Shape::Num),
    (StepKind::Scatter, "scatter", Shape::Num),
    (StepKind::Combine, "combine", Shape::OptNum),
    (StepKind::Reduce, "reduce", Shape::Num),
    (StepKind::Compare, "compare", Shape::Format),
];

/// 2026-10-02: The step's spelling.
pub fn step_name(k: StepKind) -> &'static str {
    STEPS
        .iter()
        .find(|(s, _, _)| *s == k)
        .map_or("?", |(_, n, _)| n)
}

/// 2026-10-02: The step a spelling names.
pub fn parse_step(s: &str) -> Option<StepKind> {
    STEPS.iter().find(|(_, n, _)| *n == s).map(|(k, _, _)| *k)
}

/// 2026-10-02: The shape of `k`'s value.
pub fn shape_of(k: StepKind) -> Shape {
    STEPS
        .iter()
        .find(|(s, _, _)| *s == k)
        .map_or(Shape::Num, |(_, _, sh)| *sh)
}

fn num(s: &str) -> Result<Num, String> {
    Num::parse(s).ok_or_else(|| format!("`{s}` is no precision (bf16, f16, f32, e4m3, e2m1)"))
}

/// 2026-10-02: Parse `text` as the value of step `k`.
pub fn parse_value(k: StepKind, text: &str) -> Result<Value, String> {
    let fmt = |s: &str| Format::parse(s).map_err(|e| e.to_string());
    let v = match shape_of(k) {
        Shape::Format => Value::Format(fmt(text)?),
        Shape::Weight => {
            let (stored, operand) = text
                .split_once("->")
                .ok_or_else(|| format!("`{text}` is not `<stored format>-><precision>`"))?;
            Value::Weight {
                stored: fmt(stored)?,
                operand: num(operand)?,
            }
        }
        Shape::Mma => {
            let (a, b) = text
                .split_once('*')
                .ok_or_else(|| format!("`{text}` is not `<precision>*<precision>`"))?;
            Value::Mma {
                a: num(a)?,
                b: num(b)?,
            }
        }
        Shape::Num => Value::Num(num(text)?),
        Shape::OptNum => match text {
            "none" => Value::OptNum(None),
            other => Value::OptNum(Some(num(other)?)),
        },
        Shape::Kv => {
            if text.is_empty() || !text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!("`{text}` is no KV-cache dtype spelling"));
            }
            Value::Kv(text.to_string())
        }
        Shape::State => Value::State(
            StateDtype::parse(text)
                .ok_or_else(|| format!("`{text}` is no state dtype (f32, f16, bf16, fp8)"))?,
        ),
    };
    Ok(v)
}
