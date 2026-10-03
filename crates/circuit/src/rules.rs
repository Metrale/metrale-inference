// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Fusion rules: `kernels/<hw>/common/FUSIONS.toml`. Each rule names an op chain,
//! the kernel that runs it, where it applies (rows, modes, caps, policy settings) and how its
//! numerics relate to the reference.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A `bit_identical` rule names its microtest, and a `differs` rule its opt-in lever; a
//!   rule missing either is a load error, as is a lever or microtest on a `reference` rule.
//! - Rule ids are unique; `rows` is a non-empty inclusive range; `modes` is non-empty.
//! - Every rule cites the dispatch site that makes today's choice (`cite`).

use std::collections::{BTreeMap, BTreeSet};

use crate::format::Format;
use crate::ir::{LayerKind, LinearRole, OpKind};

/// 2026-09-28: Which forward a plan is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mode {
    /// 2026-09-28: One sequence, one row.
    Decode,
    /// 2026-09-28: `n` sequences, one row each, padded to the batch ladder.
    MultiSeq,
    /// 2026-09-28: `K` draft rows of one sequence (MTP verify).
    Verify,
    /// 2026-09-28: The MTP draft head proposing one token for each of `n` sequences.
    Draft,
    /// 2026-09-30: `n` sequences' MTP verifies in one forward (`decode_verify_batched`):
    /// `R = Σ k` rows, seq-major, planned for a row table ([`crate::runs::RowTable`]).
    VerifyBatch,
    /// 2026-10-03: One sequence's prefill pass at offset 0: `T` prompt rows, contiguous
    /// attention over the pass itself (LIFECYCLE-DESIGN.md 15.4). Planned per row bucket
    /// ([`crate::buckets`]); `T` is a runtime argument.
    Prefill,
    /// 2026-10-03: A prefill pass at an offset above 0 (a later chunk, or the tail pass of a
    /// split): paged attention over the sequence's KV blocks.
    PrefillChunk,
}

impl Mode {
    /// 2026-09-28: Every mode, in plan-file order.
    /// 2026-09-30: [`Mode::VerifyBatch`] is not among them: its plans are keyed by a row table,
    /// not a row count.
    pub const ALL: [Mode; 4] = [Mode::Decode, Mode::MultiSeq, Mode::Verify, Mode::Draft];

    /// 2026-09-28: The spelling in rules, file names and the CLI.
    pub fn name(self) -> &'static str {
        match self {
            Mode::Decode => "decode",
            Mode::MultiSeq => "multi_seq",
            Mode::Verify => "verify",
            Mode::Draft => "draft",
            Mode::VerifyBatch => "verify_batch",
            Mode::Prefill => "prefill",
            Mode::PrefillChunk => "prefill_chunk",
        }
    }

    /// 2026-10-03: The prefill modes, whose plans are keyed by a row bucket.
    pub const PREFILL: [Mode; 2] = [Mode::Prefill, Mode::PrefillChunk];

    /// 2026-10-03: A prefill mode.
    pub fn is_prefill(self) -> bool {
        matches!(self, Mode::Prefill | Mode::PrefillChunk)
    }

    /// 2026-09-28: Parse [`Mode::name`].
    pub fn parse(s: &str) -> Option<Self> {
        Mode::ALL
            .into_iter()
            .chain([Mode::VerifyBatch])
            .chain(Mode::PREFILL)
            .find(|m| m.name() == s)
    }
}

/// 2026-09-28: A kernel entry point as the engine looks it up: the module name (the source
/// stem after KERNEL.toml `[modules]` renames, e.g. `norm` for `rms_norm.cu`) and the
/// function.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId {
    /// 2026-09-28: Module, e.g. `norm`.
    pub module: String,
    /// 2026-09-28: Function, e.g. `residual_add_rms_norm`.
    pub func: String,
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", self.module, self.func)
    }
}

/// 2026-09-28: How a rule's output relates to the reference numerics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Numerics {
    /// 2026-09-28: Byte-identical to the unfused chain, proven by the named microtest.
    BitIdentical {
        /// 2026-09-28: The microtest.
        microtest: String,
    },
    /// 2026-09-28: Today's default: the reference definition, grandfathered.
    Reference,
    /// 2026-09-28: Changes numerics; selected only when `lever` is opted in.
    Differs {
        /// 2026-09-28: The opt-in lever.
        lever: String,
    },
}

impl Numerics {
    /// 2026-09-28: `bit_identical`, `reference` or `differs`.
    pub fn class(&self) -> &'static str {
        match self {
            Numerics::BitIdentical { .. } => "bit_identical",
            Numerics::Reference => "reference",
            Numerics::Differs { .. } => "differs",
        }
    }
}

/// 2026-09-28: One element of a rule's op chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternOp {
    /// 2026-09-28: The op the node must be. For `linear` with a `roles` list, any of
    /// [`PatternOp::roles`] matches and this holds the first.
    pub op: OpKind,
    /// 2026-09-28: Linear roles any of which matches; empty means `op` exactly.
    pub roles: BTreeSet<LinearRole>,
    /// 2026-09-28: The node's layer must be of this kind, when set.
    pub layer_kind: Option<LayerKind>,
    /// 2026-09-28: The node's template-local id must be this, when set.
    pub local: Option<String>,
    /// 2026-09-28: The node's weight format must be this, when set.
    pub weight: Option<Format>,
    /// 2026-09-28: The format the kernel reads its first input as. Checked after fusion
    /// against the format the producing group writes, so a rule set whose kernels disagree
    /// on a materialised edge is refused.
    pub input: Option<Format>,
    /// 2026-09-28: The format the kernel stores this element's outputs in, when it is not
    /// the circuit's declared format (the verify conv stores BF16 where decode keeps F32).
    pub writes: Option<Format>,
    /// 2026-09-28: The kernel writes this element's outputs even though a later element
    /// reads them, so they may have readers outside the group.
    pub keep: bool,
    /// 2026-09-30: The kernel stores this element's outputs in memory, where a later launch of
    /// the group reads them back (a per-run schedule's conv rows), so they are materialized even
    /// when every reader is in the group.
    pub stored: bool,
    /// 2026-09-28: The element joins the group by sitting in the same block as the first
    /// element rather than by reading an output of an earlier one (a shared expert beside
    /// the routed experts).
    pub sibling: bool,
    /// 2026-10-02: The format the kernel hands this element's outputs to later members of the
    /// group in, where it is not the format the plan stores (a sum kept in FP32 registers);
    /// `None` hands them on as stored ([`crate::pipeline::require`]).
    pub holds: Option<Format>,
    /// 2026-10-02: Steps of this element's node that the routing runs at another value than
    /// the reference requirement (an activation quantized inside the group, a sum rounded to
    /// BF16), stated so the plan shows them and the kernel must declare them.
    pub steps: BTreeMap<crate::pipeline::StepKind, crate::pipeline::Value>,
}

/// 2026-09-28: How many times a group's kernels launch per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Repeat {
    /// 2026-09-28: Once.
    Once,
    /// 2026-09-28: Once per row (a per-sequence loop).
    PerRow,
    /// 2026-09-28: Once per `n`-row chunk.
    Chunk(u64),
    /// 2026-09-29: Once per row but the last.
    PerRowButLast,
    /// 2026-09-30: Per run of a batched verify, as the rule's selectors say
    /// ([`Rule::runs`], [`crate::fuser::Group::runs`]); not a function of the rows alone.
    PerRun,
}

impl Repeat {
    /// 2026-09-28: Launches of each kernel for `rows` rows. 2026-09-30: `None` for
    /// [`Repeat::PerRun`], whose count is its runs' ([`crate::fuser::Group::launch_count`]).
    pub fn count(self, rows: u64) -> Option<u64> {
        match self {
            Repeat::Once => Some(1),
            Repeat::PerRow => Some(rows),
            Repeat::Chunk(c) => Some(rows.div_ceil(c)),
            Repeat::PerRowButLast => Some(rows.saturating_sub(1)),
            Repeat::PerRun => None,
        }
    }

    /// 2026-09-28: `once`, `per_row` or `chunk<n>`.
    pub fn name(self) -> String {
        match self {
            Repeat::Once => "once".into(),
            Repeat::PerRow => "per_row".into(),
            Repeat::Chunk(c) => format!("chunk{c}"),
            Repeat::PerRowButLast => "per_row_but_last".into(),
            Repeat::PerRun => "per_run".into(),
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "once" => Some(Repeat::Once),
            "per_row" => Some(Repeat::PerRow),
            "per_row_but_last" => Some(Repeat::PerRowButLast),
            "per_run" => Some(Repeat::PerRun),
            _ => s
                .strip_prefix("chunk")?
                .parse::<u64>()
                .ok()
                .filter(|&c| c > 0)
                .map(Repeat::Chunk),
        }
    }
}

/// 2026-09-28: One fusion rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// 2026-09-28: Unique id.
    pub id: String,
    /// 2026-09-28: The op chain, each element reading an output of the one before.
    pub pattern: Vec<PatternOp>,
    /// 2026-09-28: The kernels the emitter launches for the chain, in order. Empty for a
    /// chain that runs as a copy-engine transfer (the embedding row copy).
    pub kernels: Vec<KernelId>,
    /// 2026-09-28: How often the kernels launch per step.
    pub repeat: Repeat,
    /// 2026-09-29: Copy-engine transfers the emitter issues besides its kernels, and how
    /// often; `None` for none (the verify conv's state snapshots are `per_row_but_last`).
    pub copies: Option<Repeat>,
    /// 2026-09-28: The executor's emitter for this kernel.
    pub emitter: String,
    /// 2026-09-28: Inclusive row range.
    pub rows: (u64, u64),
    /// 2026-09-28: Modes it applies in.
    pub modes: BTreeSet<Mode>,
    /// 2026-09-28: Kernel capability bits it needs.
    pub requires: BTreeSet<String>,
    /// 2026-09-28: Policy settings that must hold (`kv_cache_dtype = "bf16"`).
    pub when: BTreeMap<String, String>,
    /// 2026-09-28: Numerics class.
    pub numerics: Numerics,
    /// 2026-09-28: Higher wins.
    pub priority: i64,
    /// 2026-09-28: The dispatch site(s) that make this choice today.
    pub cite: String,
    /// 2026-09-30: A `per_run` rule's selectors (`[[rule.run]]`), in file order; empty for any
    /// other repeat.
    pub runs: Vec<crate::runs::RunSelect>,
}

/// 2026-09-28: Why FUSIONS.toml did not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuleError {
    /// 2026-09-28: Not valid TOML or not the rules shape.
    #[error("FUSIONS.toml: {0}")]
    Parse(String),
    /// 2026-09-28: Two rules share an id.
    #[error("duplicate rule id `{0}`")]
    DuplicateId(String),
    /// 2026-09-30: A malformed `[[runtime]]` route ([`crate::runtime`]).
    #[error("runtime `{route}`: {detail}")]
    Runtime {
        /// 2026-09-30: Route id.
        route: String,
        /// 2026-09-30: What was wrong.
        detail: String,
    },
    /// 2026-09-28: A pattern op the vocabulary does not have.
    #[error("rule `{rule}`: {detail}")]
    Op {
        /// 2026-09-28: Rule id.
        rule: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
    /// 2026-09-28: `bit_identical` without `microtest`.
    #[error("rule `{0}` is bit_identical but names no microtest")]
    MissingMicrotest(String),
    /// 2026-09-28: `differs` without `lever`.
    #[error("rule `{0}` is differs but names no opt-in lever")]
    MissingLever(String),
    /// 2026-09-28: A `microtest` or `lever` on a class that takes none, or an unknown class.
    #[error("rule `{rule}`: {detail}")]
    Numerics {
        /// 2026-09-28: Rule id.
        rule: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
    /// 2026-09-28: Empty pattern, empty modes, unknown mode, bad row range or empty cite.
    #[error("rule `{rule}`: {detail}")]
    Shape {
        /// 2026-09-28: Rule id.
        rule: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
}

#[path = "rules_parse.rs"]
mod parse;
pub(crate) use parse::parse_file;
pub use parse::parse_rules;

#[cfg(test)]
#[path = "rules_tests.rs"]
mod rules_tests;
