// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The numeric pipeline of a node: the formats it reads, the ordered steps inside it
//! (activation and weight preparation, the multiply's operands, accumulation, scale application,
//! element-wise compute, cache and state precision, routing and blending) each with its
//! precision, and the formats it writes. One pipeline is REQUIRED per node of a plan
//! ([`require`]: the circuit's formats under the serving policy) and one is DECLARED per kernel
//! family, point or kernel in `KERNEL_FAMILIES.toml` ([`declare`]); [`check`] refuses a plan
//! whose kernels declare anything but exactly what the plan requires.
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - The step vocabulary of every op is one table ([`vocab::steps_of`]); a pipeline names every
//!   step of its op's vocabulary in that order, and nothing else.
//! - Every value has one canonical spelling that parses back to itself (`Value::parse(k,
//!   v.name()) == Ok(v)`), so the golden plans and the manifests read one text per value.
//! - Nothing defaults: a step whose precision cannot be stated is an error, never a guess.
//! - Precision is about values, not instructions: a BF16 activation held in an FP32 register is
//!   BF16, an E4M3 weight widened exactly is the precision it widens to; which unit and atom
//!   carry the multiply is the compute unit's (`venn::compute`).

use std::fmt;

use crate::format::Format;
use crate::state::StateDtype;

pub mod check;
pub mod declare;
pub mod query;
pub mod require;
pub mod vocab;

#[path = "act_policy.rs"]
mod act_policy;

pub use check::{PlanPipelines, check_plan};
pub use require::{Need, required};

/// 2026-10-02: The element precision of a value inside a kernel (registers, operands, sums).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Num {
    /// 2026-10-02: bfloat16.
    Bf16,
    /// 2026-10-02: IEEE half.
    F16,
    /// 2026-10-02: float32.
    F32,
    /// 2026-10-02: FP8 E4M3 values (an FP8 activation or weight as multiplied).
    E4m3,
    /// 2026-10-02: FP4 E2M1 values (an NVFP4 activation or weight as multiplied).
    E2m1,
}

const NUMS: [(Num, &str); 5] = [
    (Num::Bf16, "bf16"),
    (Num::F16, "f16"),
    (Num::F32, "f32"),
    (Num::E4m3, "e4m3"),
    (Num::E2m1, "e2m1"),
];

impl Num {
    /// 2026-10-02: The canonical spelling.
    pub fn name(self) -> &'static str {
        NUMS.iter()
            .find(|(n, _)| *n == self)
            .map_or("?", |(_, s)| s)
    }

    /// 2026-10-02: Parse the canonical spelling.
    pub fn parse(s: &str) -> Option<Self> {
        NUMS.iter().find(|(_, n)| *n == s).map(|(v, _)| *v)
    }

    /// 2026-10-02: The precision of a tensor format's values: an FP8 tensor is E4M3, an NVFP4 one
    /// E2M1; `None` for `i32`, which no multiply or compute step reads as a number.
    pub fn of_format(f: Format) -> Option<Self> {
        match f {
            Format::Bf16 => Some(Num::Bf16),
            Format::F32 => Some(Num::F32),
            Format::Fp8E4m3 { .. } => Some(Num::E4m3),
            Format::Nvfp4 { .. } => Some(Num::E2m1),
            Format::I32 => None,
        }
    }
}

/// 2026-10-02: A step of a node's pipeline. Each has one value shape ([`vocab::shape_of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StepKind {
    /// 2026-10-02: Rows gathered by expert id (or embedding rows by token id): their format.
    Gather,
    /// 2026-10-02: The format the activation is rounded or quantized to before the multiply.
    Act,
    /// 2026-10-02: The stored weight format and the precision of its unscaled values as multiplied.
    Weight,
    /// 2026-10-02: The multiply's operand precisions, activation first.
    Mma,
    /// 2026-10-02: The precision dot products (or attention's PV sums) accumulate in.
    Accumulate,
    /// 2026-10-02: The precision weight and activation scales are applied in; none without scales.
    Scale,
    /// 2026-10-02: The precision of an element-wise op's or a reduction's arithmetic.
    Compute,
    /// 2026-10-02: Data moved without arithmetic: its format.
    Move,
    /// 2026-10-02: The KV-cache dtype read or written.
    Cache,
    /// 2026-10-02: Attention: the precision of the QK dot products.
    Scores,
    /// 2026-10-02: Attention: the precision of the softmax.
    Softmax,
    /// 2026-10-02: The precision of the recurrent state as stored and updated.
    State,
    /// 2026-10-02: Routing: the precision of the scoring function and of the routing weights.
    Score,
    /// 2026-10-02: Blend: the precision of the weighted sum of each token's top-k expert rows.
    Scatter,
    /// 2026-10-02: Blend: the precision the (gated) shared expert is added in; none without one.
    Combine,
    /// 2026-10-02: A cross-rank reduction's precision.
    Reduce,
    /// 2026-10-02: Greedy selection: the format of the values compared.
    Compare,
}

/// 2026-10-02: A step's value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    /// 2026-10-02: A tensor format (`gather`, `act`, `move`, `compare`).
    Format(Format),
    /// 2026-10-02: `<stored>-><operand>` (`weight`).
    Weight {
        /// 2026-10-02: The weight as loaded.
        stored: Format,
        /// 2026-10-02: Its unscaled values' precision as multiplied.
        operand: Num,
    },
    /// 2026-10-02: `<a>*<b>` (`mma`).
    Mma {
        /// 2026-10-02: The activation operand.
        a: Num,
        /// 2026-10-02: The weight operand.
        b: Num,
    },
    /// 2026-10-02: A precision.
    Num(Num),
    /// 2026-10-02: A precision, or `none` where the step does not apply (no scales, no shared
    /// expert).
    OptNum(Option<Num>),
    /// 2026-10-02: A KV-cache dtype, spelled as `--kv-cache-dtype` spells it.
    Kv(String),
    /// 2026-10-02: A recurrent state's storage element.
    State(StateDtype),
}

/// 2026-10-02: The spelling of a state dtype (the circuit TOML's, [`StateDtype::name`]).
pub fn state_name(d: StateDtype) -> &'static str {
    d.name()
}

impl Value {
    /// 2026-10-02: The canonical spelling.
    pub fn name(&self) -> String {
        match self {
            Value::Format(f) => f.name(),
            Value::Weight { stored, operand } => format!("{}->{}", stored.name(), operand.name()),
            Value::Mma { a, b } => format!("{}*{}", a.name(), b.name()),
            Value::Num(n) => n.name().to_string(),
            Value::OptNum(None) => "none".to_string(),
            Value::OptNum(Some(n)) => n.name().to_string(),
            Value::Kv(k) => k.clone(),
            Value::State(d) => state_name(*d).to_string(),
        }
    }
}

/// 2026-10-02: One step and its value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Step {
    /// 2026-10-02: Which step.
    pub kind: StepKind,
    /// 2026-10-02: Its value.
    pub value: Value,
}

/// 2026-10-02: A node's pipeline: input formats in the node's input order, the op's steps in
/// vocabulary order, output formats in the node's output order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodePipeline {
    /// 2026-10-02: The format each input is read in.
    pub inputs: Vec<Format>,
    /// 2026-10-02: The steps.
    pub steps: Vec<Step>,
    /// 2026-10-02: The format each output is written in, or handed on in inside its group.
    pub outputs: Vec<Format>,
}

/// 2026-10-02: The parts of a pipeline a plan's rule states rather than the reference: steps,
/// and the inputs and outputs whose in-group hand-off format it states (`holds`). Rendered
/// ` (rule)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stated {
    /// 2026-10-02: Stated steps.
    pub steps: std::collections::BTreeSet<StepKind>,
    /// 2026-10-02: Indices of inputs read in a stated hand-off format.
    pub inputs: std::collections::BTreeSet<usize>,
    /// 2026-10-02: Indices of outputs handed on in a stated format.
    pub outputs: std::collections::BTreeSet<usize>,
}

impl NodePipeline {
    /// 2026-10-02: The pipeline as facts: `in <formats>`, `<step> <value>` per step, `out
    /// <formats>`, each part a rule states followed by ` (rule)`; `-` for no inputs or outputs.
    pub fn facts(&self, stated: &Stated) -> Vec<String> {
        let mark = |on: bool| if on { " (rule)" } else { "" };
        let list = |v: &[Format], marked: &dyn Fn(usize) -> bool| {
            if v.is_empty() {
                "-".to_string()
            } else {
                v.iter()
                    .enumerate()
                    .map(|(i, f)| format!("{}{}", f.name(), mark(marked(i))))
                    .collect::<Vec<_>>()
                    .join(",")
            }
        };
        let mut out = vec![format!(
            "in {}",
            list(&self.inputs, &|i| stated.inputs.contains(&i))
        )];
        out.extend(self.steps.iter().map(|s| {
            format!(
                "{} {}{}",
                vocab::step_name(s.kind),
                s.value.name(),
                mark(stated.steps.contains(&s.kind))
            )
        }));
        out.push(format!(
            "out {}",
            list(&self.outputs, &|i| stated.outputs.contains(&i))
        ));
        out
    }

    /// 2026-10-02: `in -> [step value | ...] -> out`, ASCII ([`NodePipeline::facts`]).
    pub fn render(&self, stated: &Stated) -> String {
        let facts = self.facts(stated);
        let (first, rest) = facts.split_first().map_or(("", &[][..]), |(f, r)| (f, r));
        let (last, steps) = rest.split_last().map_or(("", &[][..]), |(l, s)| (l, s));
        format!(
            "{} -> [{}] -> {}",
            first.trim_start_matches("in "),
            steps.join(" | "),
            last.trim_start_matches("out ")
        )
    }
}

impl fmt::Display for NodePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render(&Default::default()))
    }
}

/// 2026-10-02: What differs between a required and a declared pipeline, one entry per field
/// (`in[1]`, `weight`, `out[0]`), as `field: required X, declared Y`.
pub fn differences(required: &NodePipeline, declared: &NodePipeline) -> Vec<String> {
    let mut out = Vec::new();
    let fmts = |label: &str, r: &[Format], d: &[Format], out: &mut Vec<String>| {
        if r.len() != d.len() {
            out.push(format!(
                "{label}: required {} formats, declared {}",
                r.len(),
                d.len()
            ));
            return;
        }
        for (i, (a, b)) in r.iter().zip(d).enumerate() {
            if a != b {
                out.push(format!("{label}[{i}]: required {a}, declared {b}"));
            }
        }
    };
    fmts("in", &required.inputs, &declared.inputs, &mut out);
    for (r, d) in required.steps.iter().zip(&declared.steps) {
        if r != d {
            out.push(format!(
                "{}: required {}, declared {}",
                vocab::step_name(r.kind),
                r.value.name(),
                d.value.name()
            ));
        }
    }
    if required.steps.len() != declared.steps.len() {
        out.push("steps: the two pipelines name different step lists".into());
    }
    fmts("out", &required.outputs, &declared.outputs, &mut out);
    out
}

/// 2026-10-02: Why a pipeline could not be required, declared or matched.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PipelineError {
    /// 2026-10-02: The policy or the circuit does not state what a step needs.
    #[error("node `{node}` ({op}): no required pipeline: {detail}")]
    Required {
        /// 2026-10-02: Node id.
        node: String,
        /// 2026-10-02: Its op.
        op: String,
        /// 2026-10-02: What is missing or contradictory.
        detail: String,
    },
    /// 2026-10-02: A group's kernels are in no family, so their pipeline is undeclared.
    #[error("node `{node}` ({op}) runs {kernels}, which no kernel family declares a pipeline for")]
    Undeclared {
        /// 2026-10-02: Node id.
        node: String,
        /// 2026-10-02: Its op.
        op: String,
        /// 2026-10-02: The group's kernels or emitter.
        kernels: String,
    },
    /// 2026-10-02: A family's declaration cannot be resolved for a node.
    #[error("node `{node}` ({op}) in family `{family}`: {detail}")]
    Declared {
        /// 2026-10-02: Node id.
        node: String,
        /// 2026-10-02: Its op.
        op: String,
        /// 2026-10-02: Family id.
        family: String,
        /// 2026-10-02: Why.
        detail: String,
    },
    /// 2026-10-02: Kernels declare another pipeline than a plan requires, one entry per node.
    #[error(
        "{} node(s) run a kernel whose declared pipeline is not the required one:\n  {}",
        .0.len(),
        .0.iter().map(Mismatch::describe).collect::<Vec<_>>().join("\n  ")
    )]
    Mismatch(Vec<Mismatch>),
}

/// 2026-10-02: One node whose kernel declares another pipeline than its plan requires.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mismatch {
    /// 2026-10-02: Node id.
    pub node: String,
    /// 2026-10-02: The rule of its group.
    pub rule: String,
    /// 2026-10-02: The family that declares the kernel's pipeline.
    pub family: String,
    /// 2026-10-02: What differs ([`differences`]).
    pub diffs: Vec<String>,
}

impl Mismatch {
    /// 2026-10-02: One line.
    pub fn describe(&self) -> String {
        format!(
            "{} (rule `{}`, family `{}`): {}",
            self.node,
            self.rule,
            self.family,
            self.diffs.join("; ")
        )
    }
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod pipeline_tests;
