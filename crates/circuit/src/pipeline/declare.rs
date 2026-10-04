// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The DECLARED side: the pipeline each kernel family implements, as
//! `KERNEL_FAMILIES.toml` states it (`pipeline.<op>`, per point `[[family.point]]
//! pipeline.<op>`, per kernel `kernel_pipeline."<kernel>".<op>`), and its resolution for one
//! node of one plan group.
//!
//! An op key is an op base name (`linear`, `rms_norm`) or a linear role (`linear:shared_down`)
//! where one family's roles differ, optionally narrowed to where the node's first input comes
//! from an op (`linear after act_quant`) or its first output feeds one (`silu_mul feeds
//! act_quant`), as `[[family.op]]` `after` / `feeds` narrow matching: one kernel can hand the
//! same product on differently depending on what follows it. Each entry states `in`, every step of the op's vocabulary
//! ([`super::vocab`]) and `out`, or is the word `"uninstantiated"`: an op the family lists as a
//! parameterization target (a policy no point instantiates), which no node may resolve to. A
//! value may name a compile-time or policy parameter of the family as `{param}`, filled from
//! the node's point (a KV dtype, a quantizer's format).
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - Every op a family implements has a family-level pipeline (for a linear with roles: each
//!   role, by the role's key or the base `linear` key); points and kernels only override.
//! - Every entry parses under every point it can apply to, at load; resolution cannot meet a
//!   value it has not seen parse.
//! - The most specific declaration wins: a kernel's, then the node's point's, then the family's.
//!   Two kernels of one group that declare different pipelines for a node are an error.

use std::collections::{BTreeMap, BTreeSet};

use super::vocab::{parse_step, parse_value, step_name, steps_for_base};
use super::{NodePipeline, Step, StepKind};
use crate::format::Format;
use crate::ir::{LinearRole, Node, OpKind};
use crate::rules::KernelId;
use crate::venn::families::{OpSpec, Values};

/// 2026-10-02: One declared pipeline as written: values may hold `{param}` placeholders.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawPipeline {
    /// 2026-10-02: Input formats.
    pub inputs: Vec<String>,
    /// 2026-10-02: Steps, in vocabulary order.
    pub steps: Vec<(StepKind, String)>,
    /// 2026-10-02: Output formats.
    pub outputs: Vec<String>,
}

/// 2026-10-02: One declaration of one op.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Entry {
    /// 2026-10-02: The pipeline the kernels run.
    Pipeline(RawPipeline),
    /// 2026-10-02: Listed for the Venn diagram only: no source implements it yet.
    Uninstantiated,
}

/// 2026-10-02: Op key to declaration.
pub type ByOp = BTreeMap<String, Entry>;

/// 2026-10-02: A family's family-level and kernel-level declarations (point-level ones live on
/// the points, `venn::families::Point::pipeline`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FamilyPipelines {
    /// 2026-10-02: The family's own.
    pub family: ByOp,
    /// 2026-10-02: Kernels that differ.
    pub kernels: BTreeMap<KernelId, ByOp>,
}

/// 2026-10-02: Parse one `pipeline` table (op key to `{ in, <steps>, out }` or
/// `"uninstantiated"`).
pub fn parse_by_op(tables: &BTreeMap<String, toml::Value>) -> Result<ByOp, String> {
    let mut out = ByOp::new();
    for (key, v) in tables {
        let entry = match v {
            toml::Value::Table(t) => Entry::Pipeline(raw(key, t)?),
            toml::Value::String(s) if s == "uninstantiated" => Entry::Uninstantiated,
            _ => {
                return Err(format!(
                    "pipeline `{key}` is neither a table nor \"uninstantiated\""
                ));
            }
        };
        out.insert(key.clone(), entry);
    }
    Ok(out)
}

/// 2026-10-02: An op key's op (`act_quant:nvfp4/g16` with its format), base op and linear role
/// (its `after` / `feeds` narrowing checked).
struct Key<'a> {
    op: &'a str,
    base: &'a str,
    role: Option<LinearRole>,
}

fn op_name_ok(name: &str) -> bool {
    match name.split_once(':') {
        Some(("linear", r)) => LinearRole::parse(r).is_some(),
        Some(_) => false,
        None => steps_for_base(name).is_some(),
    }
}

fn base_of(key: &str) -> Result<Key<'_>, String> {
    let (head, after, feeds) = {
        let (head, feeds) = match key.split_once(" feeds ") {
            Some((h, f)) => (h, Some(f)),
            None => (key, None),
        };
        match head.split_once(" after ") {
            Some((h, a)) => (h, Some(a), feeds),
            None => (head, None, feeds),
        }
    };
    if let Some(bad) = [after, feeds]
        .into_iter()
        .flatten()
        .find(|n| !op_name_ok(n))
    {
        return Err(format!("pipeline `{key}`: `{bad}` is no op"));
    }
    let (base, role) = match head.split_once(':') {
        None => (head, None),
        Some(("linear", r)) => (
            "linear",
            Some(
                LinearRole::parse(r)
                    .ok_or_else(|| format!("pipeline `{key}`: unknown linear role `{r}`"))?,
            ),
        ),
        Some(("act_quant", f)) if crate::format::Format::parse(f).is_ok() => ("act_quant", None),
        Some(_) => {
            return Err(format!(
                "pipeline `{key}`: only `linear` (a role) and `act_quant` (a format) qualify"
            ));
        }
    };
    Ok(Key {
        op: head,
        base,
        role,
    })
}

fn raw(key: &str, t: &toml::Table) -> Result<RawPipeline, String> {
    let base = base_of(key)?.base;
    let steps = steps_for_base(base).ok_or_else(|| format!("pipeline `{key}`: unknown op"))?;
    let list = |field: &str| -> Result<Vec<String>, String> {
        let arr = t
            .get(field)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("pipeline `{key}` states no `{field}` list"))?;
        arr.iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("pipeline `{key}` `{field}`: a non-string format"))
            })
            .collect()
    };
    for k in t.keys() {
        let known = k == "in" || k == "out" || parse_step(k).is_some_and(|s| steps.contains(&s));
        if !known {
            return Err(format!(
                "pipeline `{key}`: `{k}` is not a step of `{base}` ({})",
                steps
                    .iter()
                    .map(|s| step_name(*s))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    let mut out_steps = Vec::with_capacity(steps.len());
    for &s in steps {
        let v = t
            .get(step_name(s))
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("pipeline `{key}` states no `{}`", step_name(s)))?;
        out_steps.push((s, v.to_string()));
    }
    Ok(RawPipeline {
        inputs: list("in")?,
        steps: out_steps,
        outputs: list("out")?,
    })
}

/// 2026-10-02: Check a family's declarations against the ops it implements, its kernels and
/// its points (`points`: each point's values and own declarations).
pub fn validate(
    ops: &[OpSpec],
    kernels: &[KernelId],
    decl: &FamilyPipelines,
    points: &[(&Values, &ByOp)],
) -> Result<(), String> {
    let implemented: BTreeSet<&str> = ops.iter().map(|o| o.op.as_str()).collect();
    let roles_ok = |r: LinearRole| {
        ops.iter()
            .any(|o| o.op == "linear" && (o.roles.is_empty() || o.roles.contains(&r)))
    };
    let check_keys = |by: &ByOp, whose: &str| -> Result<(), String> {
        for key in by.keys() {
            let k = base_of(key)?;
            let listed = implemented.contains(k.op) || implemented.contains(k.base);
            if !listed || k.role.is_some_and(|r| !roles_ok(r)) {
                return Err(format!(
                    "{whose} declares a pipeline for `{key}`, which the family does not implement"
                ));
            }
        }
        Ok(())
    };
    check_keys(&decl.family, "the family")?;
    // 2026-10-02: Coverage counts plain keys only (`contains_key` below never names a narrowed
    // one): a narrowed key overrides where it applies.
    for o in ops {
        let covered = if o.op == "linear" && !o.roles.is_empty() {
            o.roles.iter().all(|r| {
                decl.family.contains_key(&format!("linear:{}", r.name()))
                    || decl.family.contains_key("linear")
            })
        } else {
            decl.family.contains_key(&o.op)
        };
        if !covered {
            return Err(format!("no `pipeline.{}` for op `{}`", o.op, o.op));
        }
    }
    for (k, by) in &decl.kernels {
        if !kernels.contains(k) {
            return Err(format!(
                "kernel_pipeline names `{k}`, which is not its kernel"
            ));
        }
        check_keys(by, &format!("kernel_pipeline `{k}`"))?;
    }
    for (values, own) in points {
        check_keys(own, &format!("point {values:?}"))?;
        let mut all: Vec<(&String, &Entry)> = decl.family.iter().collect();
        all.extend(own.iter());
        all.extend(decl.kernels.values().flat_map(|b| b.iter()));
        for (key, e) in all {
            if let Entry::Pipeline(p) = e {
                resolve(p, values)
                    .map_err(|e| format!("pipeline `{key}` at point {values:?}: {e}"))?;
            }
        }
    }
    Ok(())
}

/// 2026-10-02: `text` with each `{param}` replaced by its value in `values`.
fn substitute(text: &str, values: &Values) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('{') {
        out.push_str(&rest[..at]);
        let end = rest[at..]
            .find('}')
            .ok_or_else(|| format!("`{text}` opens a `{{` it does not close"))?;
        let name = &rest[at + 1..at + end];
        let v = values
            .get(name)
            .ok_or_else(|| format!("`{{{name}}}` is no parameter of the point"))?;
        out.push_str(v);
        rest = &rest[at + end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// 2026-10-02: `p` at the point `values`.
pub fn resolve(p: &RawPipeline, values: &Values) -> Result<NodePipeline, String> {
    let fmts = |list: &[String]| -> Result<Vec<Format>, String> {
        list.iter()
            .map(|s| Format::parse(&substitute(s, values)?).map_err(|e| e.to_string()))
            .collect()
    };
    let mut steps = Vec::with_capacity(p.steps.len());
    for (kind, text) in &p.steps {
        let value = parse_value(*kind, &substitute(text, values)?)
            .map_err(|e| format!("`{}`: {e}", step_name(*kind)))?;
        steps.push(Step { kind: *kind, value });
    }
    Ok(NodePipeline {
        inputs: fmts(&p.inputs)?,
        steps,
        outputs: fmts(&p.outputs)?,
    })
}

/// 2026-10-02: Where a node sits: its op, the op producing its first input and the op of the
/// first node reading its first output (what a narrowed key matches).
#[derive(Debug, Clone, Copy)]
pub struct Site<'a> {
    /// 2026-10-02: The node.
    pub node: &'a Node,
    /// 2026-10-02: The producer of its first input.
    pub after: Option<&'a OpKind>,
    /// 2026-10-02: The first reader of its first output.
    pub feeds: Option<&'a OpKind>,
}

impl<'a> Site<'a> {
    /// 2026-10-02: Node `n` of `c`.
    pub fn of(c: &'a crate::ir::Circuit, n: crate::ir::NodeIdx) -> Self {
        let node = &c.nodes[n];
        let after = node
            .inputs
            .first()
            .and_then(|&e| c.edges[e].producer)
            .map(|p| &c.nodes[p].op);
        let feeds = node
            .outputs
            .first()
            .and_then(|&e| c.edges[e].consumers.first())
            .map(|&k| &c.nodes[k].op);
        Site { node, after, feeds }
    }
}

fn names(op: &OpKind) -> Vec<String> {
    match op {
        OpKind::Linear(_) | OpKind::ActQuant(_) => vec![op.name(), op.base_name().to_string()],
        _ => vec![op.base_name().to_string()],
    }
}

/// 2026-10-02: The most specific entry of `by` for `site`: role before base, both narrowings
/// before one, `feeds` before `after`, before none.
fn lookup<'a>(by: &'a ByOp, site: &Site<'_>) -> Option<&'a Entry> {
    let around = |o: Option<&OpKind>| o.map(names).unwrap_or_default();
    let (after, feeds) = (around(site.after), around(site.feeds));
    for op in names(&site.node.op) {
        let mut keys = Vec::new();
        for a in &after {
            for f in &feeds {
                keys.push(format!("{op} after {a} feeds {f}"));
            }
        }
        keys.extend(feeds.iter().map(|f| format!("{op} feeds {f}")));
        keys.extend(after.iter().map(|a| format!("{op} after {a}")));
        keys.push(op.clone());
        if let Some(e) = keys.iter().find_map(|k| by.get(k)) {
            return Some(e);
        }
    }
    None
}

/// 2026-10-02: The pipeline a family declares for the node at `site`, run by `group` (the plan
/// group's kernels), at the node's point `point` (the values it carries) among `points` (each
/// point's values and own declarations).
pub fn declared_for(
    decl: &FamilyPipelines,
    points: &[(&Values, &ByOp)],
    group: &[KernelId],
    site: &Site<'_>,
    point: &Values,
) -> Result<NodePipeline, String> {
    let op = site.node.op.name();
    let matching: Vec<&(&Values, &ByOp)> = points
        .iter()
        .filter(|(v, _)| point.iter().all(|(k, x)| v.get(k) == Some(x)))
        .collect();
    if matching.is_empty() {
        return Err(format!(
            "the node's point {point:?} is not one the family instantiates"
        ));
    }
    let by_kernel: BTreeSet<&Entry> = group
        .iter()
        .filter_map(|k| decl.kernels.get(k))
        .filter_map(|by| lookup(by, site))
        .collect();
    let chosen = match by_kernel.into_iter().collect::<Vec<_>>().as_slice() {
        [] => {
            let own: BTreeSet<Option<&Entry>> =
                matching.iter().map(|(_, by)| lookup(by, site)).collect();
            match own.into_iter().collect::<Vec<_>>().as_slice() {
                [Some(p)] => *p,
                [None] => lookup(&decl.family, site)
                    .ok_or_else(|| format!("no pipeline is declared for `{op}`"))?,
                _ => {
                    return Err(format!(
                        "the points matching {point:?} declare different pipelines for `{op}`"
                    ));
                }
            }
        }
        [one] => *one,
        _ => {
            return Err(format!(
                "the group's kernels declare different pipelines for `{op}`"
            ));
        }
    };
    match chosen {
        Entry::Pipeline(p) => resolve(p, point),
        Entry::Uninstantiated => Err(format!(
            "the family lists `{op}` as a parameterization target that no source implements"
        )),
    }
}

#[cfg(test)]
#[path = "declare_tests.rs"]
mod declare_tests;
