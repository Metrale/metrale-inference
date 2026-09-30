// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The state a circuit keeps between steps, declared per block
//! (`[[block.<name>.state]]`), and the one function that sizes it (M5, LIFECYCLE-DESIGN.md
//! section 3.3): the recurrent slots of GatedDeltaNet and Mamba2 layers with their verify
//! intermediates and checkpoints, and the paged KV blocks of attention layers.
//!
//! A declaration says what one unit of a state holds: one sequence slot of a recurrent state,
//! one token of one side of a KV cache. How many units exist (slots, the padding dummy, verify
//! intermediates per slot, KV blocks) is a plan input ([`StateInputs`]), decided by the engine
//! and passed in explicitly; this module multiplies and never reads a process global.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every byte count is `units x elements x element size`, the element count evaluated from
//!   the circuit's dims when it is instantiated; one function ([`StatePlan::new`]) produces
//!   every term, so the reserve and the allocation cannot size a state two ways.
//! - A format named by a key (`{kv_cache_dtype}`) must be given in [`StateInputs::formats`]; a
//!   missing key is an error, never a default.

use std::collections::BTreeMap;

#[cfg(doc)]
use crate::ir::Circuit;
use crate::ir::Section;

/// 2026-09-30: What a state is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateKind {
    /// 2026-09-30: A recurrent state, one unit per sequence slot (GatedDeltaNet and Mamba2
    /// `h`, their conv windows).
    Recurrent,
    /// 2026-09-30: One side of a paged KV cache, one unit per token.
    PagedKv,
}

impl StateKind {
    /// 2026-09-30: The spelling in the circuit TOML.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "recurrent" => Some(Self::Recurrent),
            "paged_kv" => Some(Self::PagedKv),
            _ => None,
        }
    }
}

/// 2026-09-30: How long a unit of state lives (LIFECYCLE-DESIGN.md section 3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lifetime {
    /// 2026-09-30: Allocated once with the model (a KV pool; its blocks are per sequence).
    Model,
    /// 2026-09-30: Claimed with a sequence's slot and released with it.
    Sequence,
    /// 2026-09-30: Valid from a verify's snapshot to its commit or rollback.
    Verify,
}

impl Lifetime {
    /// 2026-09-30: The spelling in the circuit TOML.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "model" => Some(Self::Model),
            "sequence" => Some(Self::Sequence),
            "verify" => Some(Self::Verify),
            _ => None,
        }
    }

    /// 2026-09-30: The spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Sequence => "sequence",
            Self::Verify => "verify",
        }
    }

    /// 2026-09-30: The one lifetime a state of `kind` may declare: a recurrent state lives with
    /// its sequence, a KV side with the model's pool. `Verify` belongs to the verify holdings
    /// ([`Holding::lifetime`]), never to a declaration.
    pub fn of_kind(kind: StateKind) -> Self {
        match kind {
            StateKind::Recurrent => Self::Sequence,
            StateKind::PagedKv => Self::Model,
        }
    }
}

/// 2026-09-30: Which verify intermediate count a recurrent state keeps per verify slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifySteps {
    /// 2026-09-30: [`VerifyInputs::h_steps`] of the slot (`K - 1`: the state after the last
    /// row stays live).
    H,
    /// 2026-09-30: [`VerifyInputs::conv_steps`] (all `K` windows).
    Conv,
}

impl VerifySteps {
    /// 2026-09-30: The spelling in the circuit TOML (`verify = "h_steps"`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "h_steps" => Some(Self::H),
            "conv_steps" => Some(Self::Conv),
            _ => None,
        }
    }
}

/// 2026-09-30: How a node touches a state of its block (`state = { h = "update" }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateAccess {
    /// 2026-09-30: Reads it (attention over the KV cache).
    Read,
    /// 2026-09-30: Writes new units without reading them (the KV write of this step's rows).
    Write,
    /// 2026-09-30: Reads and replaces it in place (a recurrence, a conv window).
    Update,
    /// 2026-09-30: Copies it after each row but the last into the verify intermediates.
    Snapshot,
}

impl StateAccess {
    /// 2026-09-30: The spelling in the circuit TOML.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "read" => Some(Self::Read),
            "write" => Some(Self::Write),
            "update" => Some(Self::Update),
            "snapshot" => Some(Self::Snapshot),
            _ => None,
        }
    }
}

/// 2026-09-30: The storage element of a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateDtype {
    F32,
    F16,
    Bf16,
    /// 2026-09-30: One E4M3 byte per element; its scales live outside the state (a static
    /// per-layer KV scale).
    Fp8,
}

impl StateDtype {
    /// 2026-09-30: The spelling in the circuit TOML and in [`StateInputs::formats`].
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            "fp8" => Some(Self::Fp8),
            _ => None,
        }
    }

    /// 2026-09-30: Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }
}

/// 2026-09-30: A state's storage format: fixed by the circuit, or named by a key the plan
/// inputs give.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateFormat {
    Fixed(StateDtype),
    Keyed(String),
}

impl StateFormat {
    /// 2026-09-30: `{key}` or a dtype spelling.
    pub fn parse(s: &str) -> Option<Self> {
        match s.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
            Some(key) if !key.is_empty() => Some(Self::Keyed(key.to_string())),
            Some(_) => None,
            None => StateDtype::parse(s).map(Self::Fixed),
        }
    }
}

/// 2026-09-30: One declared state of one block instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDecl {
    /// 2026-09-30: `<block prefix>.<local>` (`l0.gdn.h`, `draft.attn.k`).
    pub id: String,
    /// 2026-09-30: The template-local id (`h`, `conv`, `k`, `v`).
    pub local: String,
    /// 2026-09-30: The block template.
    pub block: String,
    /// 2026-09-30: The layer, when the block is a layer's.
    pub layer: Option<usize>,
    /// 2026-09-30: Main or draft.
    pub section: Section,
    /// 2026-09-30: What it is.
    pub kind: StateKind,
    /// 2026-09-30: Storage format.
    pub format: StateFormat,
    /// 2026-09-30: Elements per unit (slot or token).
    pub elements: u64,
    /// 2026-09-30: The verify intermediates a recurrent state keeps; `None` for a KV side.
    pub verify: Option<VerifySteps>,
    /// 2026-09-30: How long a live unit lives ([`Lifetime::of_kind`]).
    pub lifetime: Lifetime,
}

/// 2026-09-30: Why a state plan could not be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StateError {
    /// 2026-09-30: A keyed format the inputs do not give.
    #[error("state `{state}` reads format key `{key}`, which the plan inputs do not give")]
    MissingFormat {
        /// 2026-09-30: The state.
        state: String,
        /// 2026-09-30: The key.
        key: String,
    },
    /// 2026-09-30: Inputs that do not fit the circuit.
    #[error("state plan: {0}")]
    Inputs(String),
    /// 2026-09-30: A size that does not fit u64.
    #[error("state `{0}` overflows")]
    Overflow(String),
}

/// 2026-09-30: The verify pools of a speculative serve over the recurrent states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyInputs {
    /// 2026-09-30: Per verify slot (the dummy, when there is one, last): the h intermediates it
    /// holds (`K - 1` for a K-row verify, tiered by the engine's ladder).
    pub h_steps: Vec<u64>,
    /// 2026-09-30: Conv intermediates every verify slot holds (all K).
    pub conv_steps: u64,
}

/// 2026-09-30: One paged KV cache: blocks of `block_size` tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvInputs {
    pub blocks: u64,
    pub block_size: u64,
}

/// 2026-09-30: The plan inputs the engine decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateInputs {
    /// 2026-09-30: Storage format per key the circuit names (`kv_cache_dtype`, `ssm_h_storage`).
    pub formats: BTreeMap<String, StateDtype>,
    /// 2026-09-30: Recurrent slots, the padding dummy included when the plan has one.
    pub slots: u64,
    /// 2026-09-30: The speculative verify pools; `None` without speculation.
    pub verify: Option<VerifyInputs>,
    /// 2026-09-30: The target's KV cache; `None` leaves the attention states out.
    pub kv: Option<KvInputs>,
    /// 2026-09-30: The draft head's own KV cache; `None` leaves the draft's states out.
    pub draft_kv: Option<KvInputs>,
}

/// 2026-09-30: What a term of the plan holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Holding {
    /// 2026-09-30: The live state of every slot.
    Live,
    /// 2026-09-30: Verify intermediates (the per-row states of a verify).
    Steps,
    /// 2026-09-30: The pre-verify checkpoint, one per verify slot.
    Checkpoint,
    /// 2026-09-30: KV blocks.
    Blocks,
}

impl Holding {
    /// 2026-09-30: How long this holding of a state declared `declared` lives: the verify
    /// intermediates and checkpoints only across one verify, the rest as declared.
    pub fn lifetime(self, declared: Lifetime) -> Lifetime {
        match self {
            Holding::Steps | Holding::Checkpoint => Lifetime::Verify,
            Holding::Live | Holding::Blocks => declared,
        }
    }
}

/// 2026-09-30: One sized term: a state, what it holds, its units and bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTerm {
    pub state: String,
    pub holding: Holding,
    /// 2026-09-30: `holding.lifetime(declared)`.
    pub lifetime: Lifetime,
    pub dtype: StateDtype,
    pub units: u64,
    pub bytes: u64,
}

/// 2026-09-30: Every state of a circuit, sized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePlan {
    pub terms: Vec<StateTerm>,
}

impl StatePlan {
    /// 2026-09-30: Size every state of `states` (a circuit's [`Circuit::states`], or the
    /// declarations of one block) under `inputs`.
    pub fn new(states: &[StateDecl], inputs: &StateInputs) -> Result<Self, StateError> {
        let mut terms = Vec::new();
        for s in states {
            let dtype = match &s.format {
                StateFormat::Fixed(d) => *d,
                StateFormat::Keyed(k) => {
                    *inputs
                        .formats
                        .get(k)
                        .ok_or_else(|| StateError::MissingFormat {
                            state: s.id.clone(),
                            key: k.clone(),
                        })?
                }
            };
            let unit = s
                .elements
                .checked_mul(dtype.size())
                .ok_or_else(|| StateError::Overflow(s.id.clone()))?;
            let mut push = |holding, units: u64| -> Result<(), StateError> {
                let bytes = unit
                    .checked_mul(units)
                    .ok_or_else(|| StateError::Overflow(s.id.clone()))?;
                terms.push(StateTerm {
                    state: s.id.clone(),
                    holding,
                    lifetime: holding.lifetime(s.lifetime),
                    dtype,
                    units,
                    bytes,
                });
                Ok(())
            };
            match s.kind {
                StateKind::Recurrent => {
                    push(Holding::Live, inputs.slots)?;
                    if let Some(v) = &inputs.verify {
                        let steps = match s.verify {
                            Some(VerifySteps::H) => v.h_steps.iter().sum(),
                            Some(VerifySteps::Conv) => v.conv_steps * v.h_steps.len() as u64,
                            None => {
                                return Err(StateError::Inputs(format!(
                                    "recurrent state `{}` declares no verify intermediates",
                                    s.id
                                )));
                            }
                        };
                        push(Holding::Steps, steps)?;
                        push(Holding::Checkpoint, v.h_steps.len() as u64)?;
                    }
                }
                StateKind::PagedKv => {
                    let kv = match s.section {
                        Section::Main => inputs.kv,
                        Section::Draft => inputs.draft_kv,
                    };
                    if let Some(kv) = kv {
                        let tokens = kv
                            .blocks
                            .checked_mul(kv.block_size)
                            .ok_or_else(|| StateError::Overflow(s.id.clone()))?;
                        push(Holding::Blocks, tokens)?;
                    }
                }
            }
        }
        Ok(Self { terms })
    }

    /// 2026-09-30: Total bytes of the terms `keep` selects.
    pub fn bytes_where(&self, keep: impl Fn(&StateTerm) -> bool) -> u64 {
        self.terms.iter().filter(|t| keep(t)).map(|t| t.bytes).sum()
    }

    /// 2026-09-30: Total bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes_where(|_| true)
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
