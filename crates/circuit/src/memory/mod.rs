// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The circuit memory model (owner directive 2026-10-02): for each node, the memory it
//! holds (weights and their load-time copies, the activations it materializes at the peak run,
//! its kernel family's workspace, the state it touches), for each declared state and cache its
//! bytes as a function of the runtime parameters, and their sum against a device's budget.
//!
//! - [`weights`] / [`copies`]: stored weights at the declared formats and the derived copies.
//! - [`caches`]: the cache kinds of [`crate::state::StateKind`], sized from [`CacheInputs`].
//! - [`budget`]: the class's driver terms (`HARDWARE.toml [memory]`), the util budget and the
//!   inverse queries.
//! - [`render`]: the tables and the JSON of `met circuit memory`.
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - Pure (SBIO): every text, plan and count arrives as an argument. Counts the engine decides
//!   (slots, verify slots, snapshot slots, KV blocks, the legacy arena) are inputs, so the
//!   engine's own functions decide them and this module only multiplies.
//! - Absent inputs size nothing; no count or format is defaulted here.
//! - Activations are the buffer planner's arena for each run given ([`ActivationRun`]); the
//!   peak run sets the term. A legacy arena, when given, is what the budget charges instead
//!   (it is what the default forward allocates), and the workspaces it already holds
//!   (`arena = true`) are not charged twice. A workspace is sized at the widest run.
//! - A state is attributed to the node that owns it (updates or writes it); caches, which no
//!   node touches, appear only in the state table.

pub mod budget;
pub mod caches;
pub mod copies;
pub mod render;
pub mod weights;
mod workspace;

use std::collections::BTreeMap;

pub use budget::{BudgetError, DriverTerms};
pub use caches::{CacheInputs, CacheTerm, LookupInputs};
pub use copies::{CopyRule, parse_copies};
pub use weights::NodeWeights;
pub use workspace::WorkspaceTerm;

use crate::fuser::FusionPlan;
use crate::ir::{Circuit, NodeIdx};
use crate::planner;
use crate::state::{StateAccess, StateError, StateInputs, StatePlan};
use crate::venn::families::Families;

/// 2026-10-02: Why a memory model could not be evaluated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    /// 2026-10-02: A dim a term needs.
    #[error("node `{node}`: the memory model needs dim `{dim}`, which the circuit does not define")]
    MissingDim {
        /// 2026-10-02: Node id.
        node: String,
        /// 2026-10-02: The dim.
        dim: String,
    },
    /// 2026-10-02: A size that does not evaluate (overflow, or a shape off its format's groups).
    #[error("node `{node}`: {what} has no byte size")]
    Size {
        /// 2026-10-02: Node id.
        node: String,
        /// 2026-10-02: What was sized.
        what: String,
    },
    /// 2026-10-02: The state or cache plan.
    #[error(transparent)]
    State(#[from] StateError),
    /// 2026-10-02: The buffer planner.
    #[error(transparent)]
    Plan(#[from] planner::PlanError),
    /// 2026-10-02: A workspace expression.
    #[error("family `{family}` workspace `{name}`: {detail}")]
    Workspace {
        /// 2026-10-02: Family id.
        family: String,
        /// 2026-10-02: Workspace name.
        name: String,
        /// 2026-10-02: What was wrong.
        detail: String,
    },
    /// 2026-10-02: The budget or a query over it.
    #[error(transparent)]
    Budget(#[from] BudgetError),
}

/// 2026-10-02: One fused plan whose activations the model sizes, at its row count.
#[derive(Debug, Clone, Copy)]
pub struct ActivationRun<'a> {
    /// 2026-10-02: What the run is (`decode C=16`, `verify 16x4`, `prefill T=8192`).
    pub label: &'a str,
    /// 2026-10-02: Rows the plan is sized for.
    pub rows: u64,
    /// 2026-10-02: The plan.
    pub plan: &'a FusionPlan,
}

/// 2026-10-02: Everything a memory model is evaluated from.
#[derive(Clone, Copy)]
pub struct MemoryInputs<'a> {
    /// 2026-10-02: The circuit at the served formats.
    pub served: &'a Circuit,
    /// 2026-10-02: The same model at its checkpoint's declared formats.
    pub declared: &'a Circuit,
    /// 2026-10-02: Derived-copy rules.
    pub copies: &'a [CopyRule],
    /// 2026-10-02: Serve settings the copy rules read.
    pub settings: &'a BTreeMap<String, String>,
    /// 2026-10-02: Recurrent slots and KV blocks (the engine's counts).
    pub states: &'a StateInputs,
    /// 2026-10-02: The draft head's KV element type when it differs from the target's (an
    /// MTP head keeps its own pool, `kv_cache_dtype` of the draft section's states).
    pub draft_kv_dtype: Option<crate::state::StateDtype>,
    /// 2026-10-02: Cache units (the engine's counts).
    pub caches: &'a CacheInputs,
    /// 2026-10-02: Activation runs; the largest arena is the term.
    pub runs: &'a [ActivationRun<'a>],
    /// 2026-10-02: Kernel families for the workspaces, and the device's SM count they read.
    pub families: Option<(&'a Families, u64)>,
    /// 2026-10-02: The legacy buffer arena the default forward allocates, bytes.
    pub legacy_arena: Option<u64>,
    /// 2026-10-02: The class's driver terms and the util budget, bytes.
    pub driver: DriverTerms,
    /// 2026-10-02: `device memory x util`.
    pub budget_bytes: u64,
    /// 2026-10-02: Small-allocation chunk slack measured from a ledger; 0 when unknown.
    pub chunk_slack: u64,
}

/// 2026-10-02: One node's memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeMemory {
    /// 2026-10-02: The node.
    pub node: NodeIdx,
    /// 2026-10-02: Stored weight bytes.
    pub stored: u64,
    /// 2026-10-02: Derived copy bytes.
    pub derived: u64,
    /// 2026-10-02: Bytes of the edges it materializes in the peak run.
    pub activations: u64,
    /// 2026-10-02: Its family's workspace (shared by every node of the family).
    pub workspace: u64,
    /// 2026-10-02: Bytes of the states it owns (updates or writes).
    pub state: u64,
}

/// 2026-10-02: One activation run's arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArena {
    /// 2026-10-02: The run's label.
    pub label: String,
    /// 2026-10-02: Rows.
    pub rows: u64,
    /// 2026-10-02: Planned arena (liveness-shared) bytes.
    pub arena: u64,
    /// 2026-10-02: Materialized bytes before reuse.
    pub materialized: u64,
}

/// 2026-10-02: The sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Totals {
    pub weights_stored: u64,
    /// 2026-10-02: Of the stored bytes, the checkpoint tensors no weight node binds (norms, conv
    /// and gate parameters, a vision tower), measured by the caller ([`MemoryReport::with_outside`]).
    pub weights_outside: u64,
    pub weights_derived: u64,
    /// 2026-10-02: Of the derived bytes, the leaked ones.
    pub weights_leaked: u64,
    /// 2026-10-02: The planner's peak arena.
    pub activations_planned: u64,
    /// 2026-10-02: The activations the budget charges (the legacy arena when given).
    pub activations: u64,
    /// 2026-10-02: Workspaces the budget charges.
    pub workspace: u64,
    /// 2026-10-02: Recurrent pool and KV.
    pub states: u64,
    /// 2026-10-02: Device caches.
    pub caches: u64,
    /// 2026-10-02: Host caches.
    pub host: u64,
    pub driver: u64,
    /// 2026-10-02: Everything on the device.
    pub device: u64,
}

/// 2026-10-02: An evaluated memory model.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryReport {
    pub nodes: Vec<NodeMemory>,
    pub weights: Vec<NodeWeights>,
    pub states: StatePlan,
    pub caches: Vec<CacheTerm>,
    pub runs: Vec<RunArena>,
    pub workspaces: Vec<WorkspaceTerm>,
    pub totals: Totals,
    pub budget_bytes: u64,
}

impl MemoryReport {
    /// 2026-10-02: Charge `bytes` of checkpoint tensors no weight node binds to the stored
    /// weights (the caller reads them from the checkpoint's headers).
    pub fn with_outside(mut self, bytes: u64) -> Self {
        self.totals.weights_outside += bytes;
        self.totals.weights_stored += bytes;
        self.totals.device += bytes;
        self
    }

    /// 2026-10-02: `budget - device` (negative: over budget).
    pub fn headroom(&self) -> i128 {
        i128::from(self.budget_bytes) - i128::from(self.totals.device)
    }
}

/// 2026-10-02: KV blocks `sequences` sequences of `tokens` tokens each hold, as the scheduler's
/// admission reserves them (`tokens / block_size + 1` per sequence: the block the next token
/// writes into is reserved with the prompt), plus the pool's one dummy block. `None` for a zero
/// block size or on overflow.
pub fn kv_blocks_for(sequences: u64, tokens: u64, block_size: u64) -> Option<u64> {
    let per_seq = tokens.checked_div(block_size)?.checked_add(1)?;
    sequences.checked_mul(per_seq)?.checked_add(1)
}

/// 2026-10-02: Every non-cache state of `c` sized under `inputs`, the draft section's at
/// `draft_kv` when given.
fn state_plan(
    c: &Circuit,
    inputs: &StateInputs,
    draft_kv: Option<crate::state::StateDtype>,
) -> Result<StatePlan, MemoryError> {
    let Some(dt) = draft_kv else {
        return Ok(StatePlan::new(&c.states, inputs)?);
    };
    let (draft, main): (Vec<_>, Vec<_>) = c
        .states
        .iter()
        .cloned()
        .partition(|s| s.section == crate::ir::Section::Draft);
    let mut di = inputs.clone();
    di.formats.insert("kv_cache_dtype".into(), dt);
    let mut plan = StatePlan::new(&main, inputs)?;
    plan.terms.extend(StatePlan::new(&draft, &di)?.terms);
    Ok(plan)
}

/// 2026-10-02: Evaluate the model.
pub fn evaluate(inp: &MemoryInputs<'_>) -> Result<MemoryReport, MemoryError> {
    let c = inp.served;
    let weights = weights::node_weights(c, inp.declared, inp.copies, inp.settings)?;
    let states = state_plan(c, inp.states, inp.draft_kv_dtype)?;
    let caches = caches::cache_terms(&c.states, &inp.states.formats, inp.caches)?;
    let mut runs = Vec::new();
    let mut peak: Option<(u64, planner::BufferPlan)> = None;
    for r in inp.runs {
        let b = planner::plan_buffers(c, r.plan, r.rows)?;
        runs.push(RunArena {
            label: r.label.to_string(),
            rows: r.rows,
            arena: b.arena_bytes,
            materialized: b.materialized_bytes,
        });
        if peak.as_ref().is_none_or(|p| b.arena_bytes > p.0) {
            peak = Some((b.arena_bytes, b));
        }
    }
    let workspaces = match inp.families {
        Some((fams, sm)) => {
            let mut all = Vec::new();
            for r in inp.runs {
                all.push(workspace::workspace_terms(c, r.plan, fams, r.rows, sm)?);
            }
            workspace::widest(all)
        }
        None => Vec::new(),
    };
    let mut nodes: Vec<NodeMemory> = (0..c.nodes.len())
        .map(|node| NodeMemory {
            node,
            stored: 0,
            derived: 0,
            activations: 0,
            workspace: 0,
            state: 0,
        })
        .collect();
    for w in &weights {
        nodes[w.node].stored = w.stored;
        nodes[w.node].derived = w.derived.iter().map(|d| d.bytes).sum();
    }
    if let Some((_, b)) = &peak {
        for s in &b.slots {
            if let Some(p) = c.edges[s.edge].producer {
                nodes[p].activations += s.bytes;
            }
        }
    }
    for w in &workspaces {
        for &n in &w.nodes {
            nodes[n].workspace = nodes[n].workspace.max(w.bytes);
        }
    }
    // 2026-10-02: A state belongs to the node that updates or writes it; readers and
    // snapshotters touch it without owning it, so the per-node column sums to the states.
    for (i, n) in c.nodes.iter().enumerate() {
        nodes[i].state = n
            .state
            .iter()
            .filter(|(_, a)| matches!(a, StateAccess::Update | StateAccess::Write))
            .map(|&(si, _)| states.bytes_where(|t| t.state == c.states[si].id))
            .sum();
    }
    let mut t = Totals::default();
    for w in &weights {
        t.weights_stored += w.stored;
        for d in &w.derived {
            t.weights_derived += d.bytes;
            if d.leaked {
                t.weights_leaked += d.bytes;
            }
        }
    }
    t.activations_planned = peak.as_ref().map_or(0, |p| p.0);
    t.activations = inp.legacy_arena.unwrap_or(t.activations_planned);
    t.workspace = workspaces
        .iter()
        .filter(|w| inp.legacy_arena.is_none() || !w.arena)
        .map(|w| w.bytes)
        .sum();
    t.states = states.bytes();
    t.caches = caches.iter().filter(|x| !x.host).map(|x| x.bytes).sum();
    t.host = caches.iter().filter(|x| x.host).map(|x| x.bytes).sum();
    t.driver = inp.driver.bytes(inp.budget_bytes, inp.chunk_slack);
    t.device = t.weights_stored
        + t.weights_derived
        + t.activations
        + t.workspace
        + t.states
        + t.caches
        + t.driver;
    Ok(MemoryReport {
        nodes,
        weights,
        states,
        caches,
        runs,
        workspaces,
        totals: t,
        budget_bytes: inp.budget_bytes,
    })
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod memory_tests;
