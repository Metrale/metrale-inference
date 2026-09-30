// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Classify one target node against the kernel families (see [`super`] for the
//! classes): which families can run it, its point in each, the nearest point a compared model
//! runs (or, failing that, the nearest instantiated point), and the evidence at its point.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - A family is a candidate for a node when one of its op specs matches the node (op, role,
//!   weight and input formats, template parameters, consumer) and the row count is inside the
//!   rows one launch covers.
//! - Only nodes of the same op are compared, so both points carry the same parameters.
//! - The primary finding is the best candidate by (class, differing point parameters, compared
//!   with a model before a bare point, manifest order); every other candidate whose class is an
//!   opportunity is kept as a secondary finding.

use std::collections::BTreeMap;

use super::families::{Extract, Families, Family, How, OpSpec, ParamKind, Values};
use super::{Class, Compared, Diff, Finding, Subject, VennError};
use crate::ir::{Circuit, Node, NodeIdx, OpKind};

/// 2026-09-29: A node a compared model runs, placed in its family.
#[derive(Debug, Clone)]
pub struct Usage<'a> {
    /// 2026-09-29: Recipe.
    pub recipe: &'a str,
    /// 2026-09-29: The node.
    pub node: &'a Node,
    /// 2026-09-29: Its family.
    pub family: &'a Family,
    /// 2026-09-29: What its plan group launches.
    pub kernels: String,
    /// 2026-09-29: Its compile-time and policy values.
    pub point: Values,
    /// 2026-09-29: Its runtime values.
    pub runtime: Values,
}

/// 2026-09-29: Every node the compared plans run, placed in the family of its plan group.
pub fn usages<'a>(
    against: &[Subject<'a>],
    fams: &'a Families,
) -> Result<Vec<Usage<'a>>, VennError> {
    let mut out = Vec::new();
    for s in against {
        let Some(plan) = s.plan else {
            return Err(VennError::Load(format!(
                "{}: a compared model needs a plan",
                s.recipe
            )));
        };
        for g in &plan.groups {
            let kernels = if g.kernels.is_empty() {
                format!("({} emitter)", g.emitter)
            } else {
                g.kernels
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(" + ")
            };
            for &n in &g.nodes {
                let node = &s.circuit.nodes[n];
                let family = fams
                    .of_group(&g.kernels, &g.emitter, &node.op)
                    .ok_or_else(|| VennError::UnmappedKernel {
                        recipe: s.recipe.to_string(),
                        node: node.id.clone(),
                        op: node.op.name(),
                        kernels: kernels.clone(),
                    })?;
                let (point, runtime) = point_of(family, s, node)?;
                out.push(Usage {
                    recipe: s.recipe,
                    node,
                    family,
                    kernels: kernels.clone(),
                    point,
                    runtime,
                });
            }
        }
    }
    Ok(out)
}

/// 2026-09-29: The families that can run `n` at `rows` rows, in manifest order.
pub fn candidates<'f>(fams: &'f Families, c: &Circuit, n: &Node, rows: u64) -> Vec<&'f Family> {
    fams.families
        .iter()
        .filter(|f| (f.rows.0..=f.rows.1).contains(&rows))
        .filter(|f| f.ops.iter().any(|s| op_matches(s, c, n)))
        .collect()
}

fn op_matches(s: &OpSpec, c: &Circuit, n: &Node) -> bool {
    let role_ok = match n.op {
        OpKind::Linear(r) => s.roles.is_empty() || s.roles.contains(&r),
        _ => true,
    };
    let input = n.inputs.first().map(|&e| c.edges[e].format);
    let feeds_ok = s.feeds.is_empty()
        || n.outputs.first().is_some_and(|&e| {
            c.edges[e]
                .consumers
                .iter()
                .any(|&k| names(&s.feeds, &c.nodes[k].op))
        });
    let after_ok = s.after.is_empty()
        || n.inputs.first().is_some_and(|&e| {
            c.edges[e]
                .producer
                .is_some_and(|p| names(&s.after, &c.nodes[p].op))
        });
    let beside_ok = s.beside.is_empty()
        || n.inputs.first().is_some_and(|&e| {
            c.edges[e]
                .consumers
                .iter()
                .any(|&k| c.nodes[k].id != n.id && names(&s.beside, &c.nodes[k].op))
        });
    s.op == n.op.base_name()
        && after_ok
        && beside_ok
        && role_ok
        && (s.weight.is_empty() || n.weight.is_some_and(|w| s.weight.contains(&w)))
        && (s.activation.is_empty() || input.is_some_and(|a| s.activation.contains(&a)))
        && s.params.iter().all(|(k, v)| n.params.get(k) == Some(v))
        && feeds_ok
}

/// 2026-09-29: `op` is in `set`, by base name (`linear`) or qualified name (`linear:q`).
fn names(set: &std::collections::BTreeSet<String>, op: &OpKind) -> bool {
    set.contains(op.base_name()) || set.contains(&op.name())
}

fn extract(s: &Subject<'_>, n: &Node, ex: &Extract) -> Option<String> {
    let c = s.circuit;
    let first_out = || n.outputs.first().map(|&e| &c.edges[e]);
    match ex {
        Extract::Dim(d) => c.dims.get(d).map(u64::to_string),
        Extract::Weight => n.weight.map(|w| w.name()),
        Extract::Activation => n.inputs.first().map(|&e| c.edges[e].format.name()),
        Extract::Output => first_out().map(|e| e.format.name()),
        Extract::Op => Some(n.op.name()),
        Extract::ConsumerOp => first_out()
            .and_then(|e| e.consumers.first())
            .map(|&k| c.nodes[k].op.name()),
        Extract::Param(k) => n.params.get(k).cloned(),
        Extract::Setting(k) => s.settings.get(k).cloned(),
        Extract::InDim => n.inputs.first().map(|&e| c.edges[e].dim_value.to_string()),
        Extract::OutDim => first_out().map(|e| e.dim_value.to_string()),
    }
}

/// 2026-09-29: `n`'s (compile-time and policy, runtime) values in `f`.
pub fn point_of(f: &Family, s: &Subject<'_>, n: &Node) -> Result<(Values, Values), VennError> {
    let (mut point, mut runtime) = (Values::new(), Values::new());
    for p in &f.params {
        let Some(ex) = p.from.get(n.op.base_name()) else {
            continue;
        };
        let v = extract(s, n, ex)
            .or_else(|| p.absent.clone())
            .ok_or_else(|| VennError::MissingValue {
                family: f.id.clone(),
                param: p.name.clone(),
                node: n.id.clone(),
            })?;
        if p.kind == ParamKind::Runtime {
            runtime.insert(p.name.clone(), v);
        } else {
            point.insert(p.name.clone(), v);
        }
    }
    Ok((point, runtime))
}

fn diffs(f: &Family, target: &Values, other: &Values) -> Vec<Diff> {
    let mut out = Vec::new();
    for (k, v) in target {
        let o = other.get(k).cloned().unwrap_or_default();
        if &o != v {
            let kind = f.param(k).map_or(ParamKind::Runtime, |p| p.kind);
            out.push(Diff {
                param: k.clone(),
                kind,
                target: v.clone(),
                other: o,
            });
        }
    }
    out
}

fn opportunity(pointed: &[&Diff]) -> Class {
    if pointed.iter().all(|d| d.kind == ParamKind::Policy) {
        Class::PolicyVariant
    } else {
        Class::ParameterizationOpportunity
    }
}

fn evidence(f: &Family, point: &Values, rows: u64) -> Vec<super::families::EvidenceSource> {
    f.evidence
        .iter()
        .filter(|e| e.rows.contains(&rows) && point.iter().all(|(k, v)| e.point.get(k) == Some(v)))
        .map(|e| e.source.clone())
        .collect()
}

fn finding(
    f: &Family,
    s: &Subject<'_>,
    n: &Node,
    rows: u64,
    used: &[Usage<'_>],
) -> Result<Finding, VennError> {
    let (point, runtime) = point_of(f, s, n)?;
    let inst = f
        .points
        .iter()
        .find(|p| point.iter().all(|(k, v)| p.values.get(k) == Some(v)));
    let ev = evidence(f, &point, rows);
    let shared = if ev.is_empty() {
        Class::SharedUnmeasured
    } else {
        Class::Shared
    };
    let best = used
        .iter()
        .filter(|u| std::ptr::eq(u.family, f) && u.node.op.base_name() == n.op.base_name())
        .map(|u| {
            let mut d = diffs(f, &point, &u.point);
            d.extend(diffs(f, &runtime, &u.runtime));
            (u, d)
        })
        .min_by_key(|(_, d)| d.iter().filter(|x| x.kind != ParamKind::Runtime).count());
    let (class, compared, d) = match best {
        Some((u, d)) => {
            let pointed: Vec<&Diff> = d.iter().filter(|x| x.kind != ParamKind::Runtime).collect();
            let class = if pointed.is_empty() || inst.is_some_and(|p| p.how != How::Copy) {
                shared
            } else {
                opportunity(&pointed)
            };
            let compared = Compared::Model {
                recipe: u.recipe.to_string(),
                node: u.node.id.clone(),
                kernels: u.kernels.clone(),
            };
            (class, compared, d)
        }
        None => {
            let (q, d) = f
                .points
                .iter()
                .map(|p| (p, diffs(f, &point, &p.values)))
                .min_by_key(|(_, d)| d.len())
                .ok_or_else(|| VennError::Load(format!("family `{}` has no points", f.id)))?;
            let pointed: Vec<&Diff> = d.iter().collect();
            let class = if pointed.is_empty() {
                shared
            } else {
                opportunity(&pointed)
            };
            let compared = Compared::Point {
                how: q.how,
                files: q.files.clone(),
            };
            (class, compared, d)
        }
    };
    Ok(Finding {
        family: f.id.clone(),
        class,
        point,
        compared,
        diffs: d,
        instantiated: inst.map(|p| p.how),
        evidence: if matches!(class, Class::Shared | Class::SharedUnmeasured) {
            ev
        } else {
            Vec::new()
        },
    })
}

/// 2026-09-29: The primary finding of target node `idx` (`None`: novel) and the secondary
/// opportunities.
pub fn classify_node(
    target: &Subject<'_>,
    idx: NodeIdx,
    rows: u64,
    used: &[Usage<'_>],
    fams: &Families,
) -> Result<(Option<Finding>, Vec<Finding>), VennError> {
    let n = &target.circuit.nodes[idx];
    let planned = match target.plan {
        Some(plan) => {
            let g = plan
                .groups
                .iter()
                .find(|g| g.nodes.contains(&idx))
                .ok_or_else(|| {
                    VennError::Load(format!(
                        "{}: node `{}` is in no plan group",
                        target.recipe, n.id
                    ))
                })?;
            let f = fams
                .of_group(&g.kernels, &g.emitter, &n.op)
                .ok_or_else(|| VennError::UnmappedKernel {
                    recipe: target.recipe.to_string(),
                    node: n.id.clone(),
                    op: n.op.name(),
                    kernels: g
                        .kernels
                        .iter()
                        .map(|k| k.to_string())
                        .collect::<Vec<_>>()
                        .join(" + "),
                })?;
            Some(f)
        }
        None => None,
    };
    let order: BTreeMap<&str, usize> = fams
        .families
        .iter()
        .enumerate()
        .map(|(i, f)| (f.id.as_str(), i))
        .collect();
    let mut all = Vec::new();
    let mut fams_for: Vec<&Family> = candidates(fams, target.circuit, n, rows);
    if let Some(f) = planned
        && !fams_for.iter().any(|g| std::ptr::eq(*g, f))
    {
        fams_for.push(f);
    }
    for f in fams_for {
        all.push(finding(f, target, n, rows, used)?);
    }
    let key = |x: &Finding| {
        (
            x.class,
            x.diffs
                .iter()
                .filter(|d| d.kind != ParamKind::Runtime)
                .count(),
            matches!(x.compared, Compared::Point { .. }),
            order[x.family.as_str()],
        )
    };
    all.sort_by_key(key);
    let primary_pos = match planned {
        Some(f) => all.iter().position(|x| x.family == f.id),
        None => (!all.is_empty()).then_some(0),
    };
    let Some(pos) = primary_pos else {
        return Ok((None, Vec::new()));
    };
    let primary = all.remove(pos);
    let also = all
        .into_iter()
        .filter(|x| {
            matches!(
                x.class,
                Class::PolicyVariant | Class::ParameterizationOpportunity
            )
        })
        .collect();
    Ok((Some(primary), also))
}
