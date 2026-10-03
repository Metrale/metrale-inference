// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The tensor-core policy of a kernel class (`kernels/<class>/HARDWARE.toml`
//! `[tensor_core_policy]`): which ops, at which modes, row counts and weight formats, a plan MUST
//! run on a tensor-core kernel, and the stated exemptions. [`audit`] checks one fused plan
//! against it from the compute units the class's kernel families declare (`venn::compute`).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Pure: the policy arrives as text, the plan and families as values.
//! - Nothing defaults: a requirement states its ops, modes, minimum rows and weight formats
//!   (`["any"]` is stated, not assumed); an exemption states its ops, modes, rows, the kernels it
//!   allows, its kind and its reason, and a `measured` exemption names its evidence.
//! - A class's policy is its own; it is not inherited along `HARDWARE.toml` `inherits`, since
//!   another class's tensor cores and kernels differ. A class without one is reported as such.
//! - A covered node whose group runs no tensor-core kernel is a violation unless an exemption
//!   lists the op, the mode, the row count and every kernel of the group. A covered node in a
//!   placeholder group (no kernel of the class covers the op) is a gap, listed, never a pass.

use std::collections::BTreeSet;

use crate::format::Format;
use crate::fuser::FusionPlan;
use crate::ir::{Circuit, LinearRole, OpKind};
use crate::rules::{KernelId, Mode};
use crate::venn::compute::{ComputeUnit, group_unit};
use crate::venn::families::Families;

/// 2026-10-02: An op a policy entry names: a base name (`linear` is every role) or `linear:<role>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpMatch {
    /// 2026-10-02: Base op name.
    pub base: String,
    /// 2026-10-02: For `linear:<role>`, the role.
    pub role: Option<LinearRole>,
}

impl OpMatch {
    fn parse(s: &str) -> Result<Self, String> {
        let (base, role) = match s.split_once(':') {
            Some(("linear", r)) => (
                "linear",
                Some(LinearRole::parse(r).ok_or_else(|| format!("unknown linear role `{r}`"))?),
            ),
            Some(_) => return Err(format!("op `{s}`: only `linear` takes a role")),
            None => (s, None),
        };
        if base != "linear" {
            OpKind::parse(base, None, None).map_err(|e| e.to_string())?;
        }
        Ok(OpMatch {
            base: base.to_string(),
            role,
        })
    }

    fn matches(&self, op: &OpKind) -> bool {
        op.base_name() == self.base
            && match (op, self.role) {
                (OpKind::Linear(r), Some(want)) => *r == want,
                _ => true,
            }
    }
}

/// 2026-10-02: The weight formats an entry covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Weights {
    /// 2026-10-02: Every format, and ops that read no weight (`["any"]`).
    Any,
    /// 2026-10-02: Only these.
    Only(BTreeSet<Format>),
}

impl Weights {
    fn covers(&self, w: Option<Format>) -> bool {
        match self {
            Weights::Any => true,
            Weights::Only(set) => w.is_some_and(|w| set.contains(&w)),
        }
    }
}

/// 2026-10-02: Ops that must run on tensor cores at these modes, from `min_rows` rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Require {
    /// 2026-10-02: The ops.
    pub ops: Vec<OpMatch>,
    /// 2026-10-02: The modes.
    pub modes: BTreeSet<Mode>,
    /// 2026-10-02: The plan's row count from which the requirement applies.
    pub min_rows: u64,
    /// 2026-10-02: The weight formats.
    pub weights: Weights,
}

/// 2026-10-02: Why an op may stay off tensor cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExemptKind {
    /// 2026-10-02: Measured: the CUDA-core kernel beats the tensor-core tile there.
    Measured,
    /// 2026-10-02: The shape leaves no MMA tile to fill (e.g. an output of one column).
    Shape,
    /// 2026-10-02: A known violation with no tensor-core kernel yet: the backlog.
    Backlog,
}

impl ExemptKind {
    /// 2026-10-02: The spelling in the policy file.
    pub fn name(self) -> &'static str {
        match self {
            ExemptKind::Measured => "measured",
            ExemptKind::Shape => "shape",
            ExemptKind::Backlog => "backlog",
        }
    }
}

/// 2026-10-02: A stated exemption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exempt {
    /// 2026-10-02: The ops.
    pub ops: Vec<OpMatch>,
    /// 2026-10-02: The modes.
    pub modes: BTreeSet<Mode>,
    /// 2026-10-02: Inclusive rows.
    pub rows: (u64, u64),
    /// 2026-10-02: The kernels a non-tensor-core group may launch for those ops.
    pub kernels: BTreeSet<KernelId>,
    /// 2026-10-02: Its kind.
    pub kind: ExemptKind,
    /// 2026-10-02: Why.
    pub reason: String,
    /// 2026-10-02: The measurement, for `measured`.
    pub evidence: Option<String>,
}

/// 2026-10-02: A class's policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcPolicy {
    /// 2026-10-02: Requirements.
    pub require: Vec<Require>,
    /// 2026-10-02: Exemptions.
    pub exempt: Vec<Exempt>,
}

#[path = "tc_policy_parse.rs"]
mod parse;
pub use parse::parse_policy;
#[path = "tc_policy_lint.rs"]
mod lint;
pub use lint::lint_rules;

/// 2026-10-02: One covered node that runs off tensor cores.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Finding {
    /// 2026-10-02: `block.node`.
    pub site: String,
    /// 2026-10-02: Op name (with its role).
    pub op: String,
    /// 2026-10-02: Declared weight format.
    pub weight: Option<String>,
    /// 2026-10-02: Mode.
    pub mode: Mode,
    /// 2026-10-02: The plan's rows.
    pub rows: u64,
    /// 2026-10-02: The group's kernels.
    pub kernels: Vec<KernelId>,
    /// 2026-10-02: The unit they run on (`host` for a group without kernels).
    pub unit: String,
}

/// 2026-10-02: One plan's audit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TcAudit {
    /// 2026-10-02: Covered nodes.
    pub covered: usize,
    /// 2026-10-02: Covered nodes on tensor cores.
    pub on_tensor_cores: usize,
    /// 2026-10-02: Covered sites off tensor cores that an exemption allows, with its index.
    pub exempted: Vec<(Finding, usize)>,
    /// 2026-10-02: Covered sites off tensor cores that nothing allows.
    pub violations: Vec<Finding>,
    /// 2026-10-02: Covered sites no kernel of the class plans (placeholder groups): gaps, which
    /// the gap table reports; listed here because the kernel they need is tensor-core work.
    pub gaps: Vec<Finding>,
}

/// 2026-10-02: The unit a group's kernels run on, from the families; an error names a kernel
/// no family declares. `None` for a group without kernels.
pub fn unit_of_group(
    families: &Families,
    kernels: &[KernelId],
) -> Result<Option<ComputeUnit>, String> {
    let mut units = Vec::with_capacity(kernels.len());
    for k in kernels {
        units.push(families.compute_of(k).ok_or_else(|| {
            format!("kernel `{k}` is in no kernel family, so its compute unit is undeclared")
        })?);
    }
    Ok(group_unit(units))
}

/// 2026-10-02: A group's unit as a plan line shows it ([`ComputeUnit::tag`]): `None` for a group
/// without kernels, `undeclared` when a kernel is in no family.
pub fn unit_tag(families: &Families, kernels: &[KernelId]) -> Option<String> {
    match unit_of_group(families, kernels) {
        Ok(u) => u.map(|u| u.tag()),
        Err(_) => Some("undeclared".to_string()),
    }
}

/// 2026-10-02: Audit `plan` of `c` against `policy`, with the units `families` declare.
/// Covered nodes in groups emitted by `skip_emitter` (the hardware planner's placeholders) are
/// gaps, not violations.
pub fn audit(
    c: &Circuit,
    plan: &FusionPlan,
    families: &Families,
    policy: &TcPolicy,
    skip_emitter: &str,
) -> Result<TcAudit, String> {
    let (mode, rows) = (plan.mode, plan.rows);
    let mut out = TcAudit::default();
    let mut seen = BTreeSet::new();
    for g in &plan.groups {
        let placeholder = g.emitter == skip_emitter;
        let mut unit: Option<Option<ComputeUnit>> = None;
        for &n in &g.nodes {
            let node = &c.nodes[n];
            let covered = policy.require.iter().any(|r| {
                r.modes.contains(&mode)
                    && rows >= r.min_rows
                    && r.weights.covers(node.weight)
                    && r.ops.iter().any(|o| o.matches(&node.op))
            });
            if !covered {
                continue;
            }
            if placeholder {
                let f = Finding {
                    site: format!("{}.{}", node.block, node.local),
                    op: node.op.name(),
                    weight: node.weight.map(|w| w.name()),
                    mode,
                    rows,
                    kernels: Vec::new(),
                    unit: "no planned kernel".to_string(),
                };
                if seen.insert(f.clone()) {
                    out.gaps.push(f);
                }
                continue;
            }
            out.covered += 1;
            let u = match &unit {
                Some(u) => u.clone(),
                None => {
                    let u = unit_of_group(families, &g.kernels)?;
                    unit = Some(u.clone());
                    u
                }
            };
            if u.as_ref().is_some_and(ComputeUnit::is_tensor_core) {
                out.on_tensor_cores += 1;
                continue;
            }
            let f = Finding {
                site: format!("{}.{}", node.block, node.local),
                op: node.op.name(),
                weight: node.weight.map(|w| w.name()),
                mode,
                rows,
                kernels: g.kernels.clone(),
                unit: u.map_or_else(|| "host".to_string(), |u| u.name()),
            };
            if !seen.insert(f.clone()) {
                continue;
            }
            let allowed = policy.exempt.iter().position(|e| {
                e.modes.contains(&mode)
                    && (e.rows.0..=e.rows.1).contains(&rows)
                    && e.ops.iter().any(|o| o.matches(&node.op))
                    && !g.kernels.is_empty()
                    && g.kernels.iter().all(|k| e.kernels.contains(k))
            });
            match allowed {
                Some(i) => out.exempted.push((f, i)),
                None => out.violations.push(f),
            }
        }
    }
    Ok(out)
}

/// 2026-10-02: One line per violation, for an error.
pub fn describe(violations: &[Finding]) -> String {
    violations
        .iter()
        .map(|f| {
            let k: Vec<String> = f.kernels.iter().map(|k| k.to_string()).collect();
            format!(
                "{} {} ({}) at {} n={} runs on {} [{}]",
                f.site,
                f.op,
                f.weight.as_deref().unwrap_or("no weight"),
                f.mode.name(),
                f.rows,
                f.unit,
                if k.is_empty() {
                    "no kernel".to_string()
                } else {
                    k.join(" + ")
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n  ")
}

/// 2026-10-02: Audit `planned` (and each runtime route's arm) of `c` on `r`'s class: `None`
/// when the class states no policy; an error ([`super::HwError::TensorCore`]) on a violation.
pub fn enforce(
    r: &super::plan::Resolved,
    c: &Circuit,
    planned: &super::plan::Planned,
) -> Result<Option<TcAudit>, super::HwError> {
    let Some(policy) = &r.chain[0].tensor_core else {
        return Ok(None);
    };
    let mut total = TcAudit::default();
    let arms = std::iter::once(planned).chain(planned.routes.iter().map(|rp| &rp.planned));
    for p in arms {
        let a = audit(c, &p.plan, &r.families, policy, super::plan::NOVEL_EMITTER)
            .map_err(super::HwError::Plan)?;
        total.covered += a.covered;
        total.on_tensor_cores += a.on_tensor_cores;
        for e in a.exempted {
            if !total.exempted.contains(&e) {
                total.exempted.push(e);
            }
        }
        for v in a.violations {
            if !total.violations.contains(&v) {
                total.violations.push(v);
            }
        }
        for g in a.gaps {
            if !total.gaps.contains(&g) {
                total.gaps.push(g);
            }
        }
    }
    if !total.violations.is_empty() {
        return Err(super::HwError::TensorCore {
            class: r.chain[0].name.clone(),
            violations: describe(&total.violations),
        });
    }
    Ok(Some(total))
}

#[cfg(test)]
#[path = "tc_policy_tests.rs"]
mod tc_policy_tests;
