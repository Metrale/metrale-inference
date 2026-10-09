// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Plan one model on one device: resolve the device's class (chain, rules, kernel
//! sources, families), check the class's build against the device, fuse every requested mode
//! and row count with the kernels the device can run, and mark every node no rule covers as a
//! gap instead of failing.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A rule that matches a node whose edge format it cannot read is left out of that plan and
//!   listed ([`Planned::refused`]); the plan never fails on it and never runs it. 2026-10-02: So
//!   is a rule whose kernels declare another pipeline than the plan requires of a node it
//!   covers ([`crate::pipeline`]): a kernel never runs a precision the circuit does not ask for.
//! - A node no available rule covers is planned by a placeholder group (`emitter = "novel"`,
//!   rule id `novel.<op>`), never by a kernel the device cannot run; the placeholder has the
//!   lowest priority, so it never displaces a real rule.
//! - On the class a model's golden plans were written for, with its own kernel target, the
//!   plan equals the golden plan (crates/server/src/cli/circuit_hw_tests.rs holds it).

use std::collections::{BTreeMap, BTreeSet};

use super::avail::{Availability, availability, check_build, kernel_status, without_fp4_kernel};
use super::class::{ClassInfo, ClassRules, chain, class_families, class_rules, planning_chain};
use super::device::{Device, MmaKind, Registry};
use super::estimate::{DeviceRoofline, device_roofline};
use super::exec::{Exec, fp4_fallback, node_exec};
use super::sources::{ClassSources, KernelTree};
use super::{HwError, ModelUnderPlan};
use crate::fuser::{FuseError, FusionPlan, fuse};
use crate::ir::{Circuit, NodeIdx};
use crate::rules::{KernelId, Mode, Numerics, PatternOp, Repeat, Rule};
use crate::runtime::RuntimeRoute;
use crate::venn::Run;
use crate::venn::families::{Families, Roofline};
use crate::venn::roofline::nvfp4_mma;

/// 2026-09-30: The emitter of a placeholder group.
pub const NOVEL_EMITTER: &str = "novel";

/// 2026-09-30: One fused plan and the nodes only a placeholder covers.
#[derive(Debug, Clone)]
pub struct Planned {
    /// 2026-09-30: Mode and rows.
    pub run: Run,
    /// 2026-09-30: The plan.
    pub plan: FusionPlan,
    /// 2026-09-30: Nodes in placeholder groups.
    pub novel: BTreeSet<NodeIdx>,
    /// 2026-09-30: Rules left out of this plan because they matched a node whose edge format
    /// they cannot read (the pattern matches op and weight, not the input format), and why.
    pub refused: Vec<(String, String)>,
    /// 2026-09-30: The runtime routes that apply to this run, each with its own plan
    /// ([`crate::runtime`]).
    pub routes: Vec<RoutePlan>,
    /// 2026-10-02: Every planned node's required pipeline, which its kernel declares too.
    pub pipelines: crate::pipeline::PlanPipelines,
}

/// 2026-09-30: A route that applies to a run, with its arm's plan.
#[derive(Debug, Clone)]
pub struct RoutePlan {
    /// 2026-09-30: The route.
    pub route: RuntimeRoute,
    /// 2026-09-30: The arm's plan (its own `routes` are empty).
    pub planned: Planned,
}

/// 2026-09-30: A device's class, resolved for one model.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// 2026-09-30: The device.
    pub device: Device,
    /// 2026-09-30: The class and its ancestors, own first.
    pub chain: Vec<ClassInfo>,
    /// 2026-09-30: The classes rules and families come from ([`planning_chain`]).
    pub planning: Vec<ClassInfo>,
    /// 2026-09-30: The class's rules.
    pub rules: ClassRules,
    /// 2026-09-30: The class's sources for the model.
    pub sources: ClassSources,
    /// 2026-09-30: What the device can run.
    pub availability: Availability,
    /// 2026-09-30: The families, available on the device, with the class's points.
    pub families: Families,
    /// 2026-09-30: Roofline constants.
    pub roofline: DeviceRoofline,
    /// 2026-10-01: The model's NVFP4-activation nodes the class compiles no FP4 block-scale
    /// kernel for ([`without_fp4_kernel`]).
    pub without_fp4: BTreeSet<NodeIdx>,
    /// 2026-10-01: How each node of the model runs here ([`node_exec`]; `None`: reads no
    /// weight). Its execution columns, its "FP4 costing" row and its cost all read this.
    pub exec: Vec<Option<Exec>>,
}

impl Resolved {
    /// 2026-09-30: The rules, empty for a class without any.
    pub fn rule_list(&self) -> &[Rule] {
        match &self.rules {
            ClassRules::Rules { rules, .. } => rules,
            ClassRules::None => &[],
        }
    }

    /// 2026-09-30: The runtime routes, empty for a class without any.
    pub fn route_list(&self) -> &[RuntimeRoute] {
        match &self.rules {
            ClassRules::Rules { runtime, .. } => runtime,
            ClassRules::None => &[],
        }
    }

    /// 2026-10-01: The constants node `n` is costed with: the device's, with the NVFP4 slot
    /// (which `node_cost` reads only for an NVFP4 activation) at the peak of the node's
    /// execution.
    pub fn roofline_of(&self, n: NodeIdx) -> Roofline {
        let base = self.roofline.roofline;
        match self.exec[n] {
            Some(e) => Roofline {
                nvfp4_tflops: e.peak(&base).0,
                ..base
            },
            None => base,
        }
    }

    /// 2026-10-01: The report's "FP4 costing", read from the nodes' executions: the device's note
    /// where it has no FP4 MMA; otherwise `native` only when every NVFP4-activation op the report
    /// costs in `c` runs on it, and the ops that do not named with the path they take.
    pub fn fp4_costing(&self, c: &Circuit) -> String {
        let base = &self.roofline.roofline;
        let fallback = fp4_fallback(&self.device);
        if !self.device.runs(MmaKind::Fp4BlockScale) {
            let lacks = match fallback {
                Exec::ExactFp8Emulation => "FP4 MMA",
                _ => "FP4 and FP8 MMA",
            };
            return format!(
                "{lacks} not native: NVFP4-activation nodes costed at {}",
                fallback.peak(base).1
            );
        }
        let sections: BTreeSet<_> = report_runs()
            .iter()
            .map(|r| crate::fuser::section_of(r.mode))
            .collect();
        let mut by_exec: BTreeMap<Exec, BTreeSet<String>> = BTreeMap::new();
        for b in c.blocks.iter().filter(|b| sections.contains(&b.section)) {
            for i in (b.first..b.end).filter(|&i| nvfp4_mma(c, &c.nodes[i])) {
                if let Some(e) = self.exec[i] {
                    by_exec.entry(e).or_default().insert(c.nodes[i].op.name());
                }
            }
        }
        let list = |s: &BTreeSet<String>| s.iter().cloned().collect::<Vec<_>>().join(", ");
        let native = by_exec.remove(&Exec::Native(MmaKind::Fp4BlockScale));
        let mut parts: Vec<String> = by_exec
            .iter()
            .map(|(e, ops)| {
                format!(
                    "no path for {} (the class compiles no FP4 block-scale kernel for them): costed at {}",
                    list(ops),
                    e.peak(base).1
                )
            })
            .collect();
        match (native, parts.is_empty()) {
            (None, true) => "not used: the model has no NVFP4-activation node".into(),
            (Some(_), true) => "native".into(),
            (native, false) => {
                if let Some(ops) = native {
                    parts.push(format!("native for {}", list(&ops)));
                }
                parts.join("; ")
            }
        }
    }
}

/// 2026-09-30: Resolve `device_id` for `model` from `tree`.
pub fn resolve(
    registry: &Registry,
    device_id: &str,
    tree: &dyn KernelTree,
    model: &ModelUnderPlan,
) -> Result<Resolved, HwError> {
    let device = registry.device(device_id)?.clone();
    let repo = tree.as_repo();
    let chain = chain(repo, &device.class)?;
    let own = &chain[0];
    if own.arch != device.arch {
        return Err(HwError::Class(format!(
            "device `{}` declares arch {}, kernels/{}/HARDWARE.toml builds {}",
            device.id, device.arch, own.name, own.arch
        )));
    }
    check_build(&device, own, &registry.guards)?;
    let rules = class_rules(repo, &device.class)?;
    let sources = tree
        .class_sources(&device.class, &model.kernel_model, &model.kernel_quant)
        .map_err(HwError::Class)?;
    let rule_list: &[Rule] = match &rules {
        ClassRules::Rules { rules, .. } => rules,
        ClassRules::None => &[],
    };
    let availability = availability(&device, own, &sources, rule_list, &registry.guards);
    let planning = planning_chain(repo, &device.class)?;
    let mut families = class_families(repo, &planning, &sources)?;
    families.families.retain(|f| {
        f.kernels.is_empty()
            || f.kernels.iter().any(|k| {
                kernel_status(&device, own, &sources, &registry.guards, k)
                    .0
                    .is_ok()
            })
    });
    // 2026-09-30: A family whose points the class compiles none of cannot be instantiated here,
    // whatever its kernel names match.
    families
        .families
        .retain(|f| f.kernels.is_empty() || !f.points.is_empty());
    let base = base_roofline(repo, &planning)?;
    let measured = (families_class(repo, &device.class)?).then_some(&base);
    let roofline = device_roofline(&device, measured, &base);
    let kernel = |k: &KernelId| match kernel_status(&device, own, &sources, &registry.guards, k) {
        (Ok(()), req) => Some(req.is_some_and(|i| i.kind == MmaKind::Fp4BlockScale)),
        (Err(_), _) => None,
    };
    let c = &model.circuit;
    let without_fp4 = without_fp4_kernel(c, rule_list, &families, &kernel);
    let exec = c
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| node_exec(&device, c, n, without_fp4.contains(&i)))
        .collect();
    Ok(Resolved {
        device,
        chain,
        planning,
        rules,
        sources,
        availability,
        families,
        roofline,
        without_fp4,
        exec,
    })
}

/// 2026-09-30: The roofline block of the nearest KERNEL_FAMILIES.toml along the chain (its
/// `context_tokens` is the estimate's assumption on every device).
fn base_roofline(
    repo: &dyn crate::venn::repo::Repo,
    chain: &[ClassInfo],
) -> Result<crate::venn::families::Roofline, HwError> {
    for c in chain {
        if let Ok(t) = repo.read(&format!("kernels/{}/common/KERNEL_FAMILIES.toml", c.name)) {
            let f = crate::venn::parse_families(&t).map_err(|e| HwError::Class(e.to_string()))?;
            return Ok(f.roofline);
        }
    }
    Err(HwError::Class(format!(
        "no KERNEL_FAMILIES.toml along `{}`'s chain",
        chain.first().map_or("", |c| c.name.as_str())
    )))
}

/// 2026-09-30: The class holds its own KERNEL_FAMILIES.toml, so its `[roofline]` is a
/// measurement on it.
fn families_class(repo: &dyn crate::venn::repo::Repo, class: &str) -> Result<bool, HwError> {
    let Ok(t) = repo.read(&format!("kernels/{class}/common/KERNEL_FAMILIES.toml")) else {
        return Ok(false);
    };
    let f = crate::venn::parse_families(&t).map_err(|e| HwError::Class(e.to_string()))?;
    Ok(f.hardware == class)
}

/// 2026-09-30: The placeholder for `op` reading `input`: a quantized read is stated, since the
/// fuser matches a pattern's quantized input format exactly (`fuser_match.rs`).
fn placeholder(op: &crate::ir::OpKind, input: Option<crate::format::Format>) -> Rule {
    let input = input.filter(|f| !f.is_plain());
    Rule {
        id: match input {
            Some(f) => format!("novel.{}.{}", op.name(), f.name()),
            None => format!("novel.{}", op.name()),
        },
        pattern: vec![PatternOp {
            op: *op,
            roles: BTreeSet::new(),
            layer_kind: None,
            local: None,
            weight: None,
            input,
            writes: None,
            keep: false,
            stored: false,
            sibling: false,
            holds: None,
            steps: BTreeMap::new(),
        }],
        kernels: Vec::new(),
        repeat: Repeat::Once,
        copies: None,
        emitter: NOVEL_EMITTER.to_string(),
        rows: (1, u64::MAX),
        modes: Mode::ALL.into_iter().collect(),
        requires: BTreeSet::new(),
        when: Default::default(),
        numerics: Numerics::Reference,
        runs: Vec::new(),
        priority: i64::MIN,
        cite: "no rule of this class covers the op on this device".into(),
    }
}

/// 2026-09-30: Fuse `circuit` at `run` with the rules and kernels of `r`, covering what no rule
/// covers with placeholders; each runtime route that applies is planned beside it.
pub fn fuse_on(
    r: &Resolved,
    circuit: &Circuit,
    policy: &crate::fuser::Policy,
    run: Run,
) -> Result<Planned, HwError> {
    let mut primary = fuse_arm(r, circuit, policy, run)?;
    for route in r
        .route_list()
        .iter()
        .filter(|x| x.applies(policy, run.mode, run.rows))
    {
        let planned = fuse_arm(r, circuit, &route.policy(policy), run)?;
        if planned.plan.digest != primary.plan.digest {
            primary.routes.push(RoutePlan {
                route: route.clone(),
                planned,
            });
        }
    }
    Ok(primary)
}

fn fuse_arm(
    r: &Resolved,
    circuit: &Circuit,
    policy: &crate::fuser::Policy,
    run: Run,
) -> Result<Planned, HwError> {
    let mut rules: Vec<Rule> = r.rule_list().to_vec();
    let mut refused = Vec::new();
    loop {
        match fuse(
            circuit,
            &rules,
            &r.availability.kernels,
            policy,
            run.mode,
            run.rows,
        ) {
            Ok(plan) => {
                let pipelines = match crate::pipeline::check_plan(
                    circuit,
                    &plan,
                    &rules,
                    &r.families,
                    &policy.settings,
                ) {
                    Ok(p) => p,
                    Err(crate::pipeline::PipelineError::Mismatch(ms)) => {
                        // 2026-10-02: Refuse each rule whose kernels run another precision than
                        // the node requires, and plan again without it.
                        for m in ms {
                            if rules.iter().any(|x| x.id == m.rule) {
                                rules.retain(|x| x.id != m.rule);
                                refused.push((m.rule.clone(), m.describe()));
                            }
                        }
                        continue;
                    }
                    Err(e) => return Err(HwError::Plan(e.to_string())),
                };
                let novel = plan
                    .groups
                    .iter()
                    .filter(|g| g.emitter == NOVEL_EMITTER)
                    .flat_map(|g| g.nodes.iter().copied())
                    .collect();
                return Ok(Planned {
                    run,
                    plan,
                    novel,
                    refused,
                    routes: Vec::new(),
                    pipelines,
                });
            }
            Err(FuseError::Uncovered { node, .. }) => {
                let idx = circuit
                    .node(&node)
                    .ok_or_else(|| HwError::Plan(format!("uncovered node `{node}` not found")))?;
                let n = &circuit.nodes[idx];
                let p = placeholder(&n.op, n.inputs.first().map(|&e| circuit.edges[e].format));
                if rules.iter().any(|x| x.id == p.id) {
                    return Err(HwError::Plan(format!(
                        "node `{node}` stays uncovered with its placeholder `{}`",
                        p.id
                    )));
                }
                rules.push(p);
            }
            Err(e @ FuseError::FormatConflict { .. }) => {
                let FuseError::FormatConflict { rule, .. } = &e else {
                    unreachable!("matched above")
                };
                let id = rule.clone();
                rules.retain(|x| x.id != id);
                refused.push((id, e.to_string()));
            }
            Err(e) => return Err(HwError::Plan(e.to_string())),
        }
    }
}

/// 2026-09-30: The runs every hardware report shows: decode at C=1, and multi-sequence decode at
/// C=16 and C=128.
pub fn report_runs() -> Vec<Run> {
    vec![
        Run {
            mode: Mode::Decode,
            rows: 1,
        },
        Run {
            mode: Mode::MultiSeq,
            rows: 16,
        },
        Run {
            mode: Mode::MultiSeq,
            rows: 128,
        },
    ]
}
