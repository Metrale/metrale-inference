// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The union report's long tables, split from `union_render.rs` for length: the
//! shape union per family (the envelope-sweep grid), the deduplicated parameterization plan and
//! the novel-kernel list, each ranked by the step share it covers.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: see [`super::union_render`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::Class;
use super::union::{SiteUse, UnionReport};
use super::union_render::{family_of, key_runs, letter, pct, point_text, run_name, short};

/// 2026-10-10: Every point and shape each family is asked to run, by whom and at which rows,
/// with its envelope status per run.
pub(super) fn shapes(s: &mut String, r: &UnionReport) {
    type Key = (String, String);
    let mut by: BTreeMap<String, BTreeMap<Key, Cell>> = BTreeMap::new();
    for u in &r.uses {
        let shape = u
            .shape
            .map_or_else(|| "-".to_string(), |(n, k)| format!("{n} x {k}"));
        let cell = by
            .entry(family_of(u))
            .or_default()
            .entry((point_text(u), shape))
            .or_default();
        cell.models.insert(u.model);
        let st = cell.runs.entry(run_name(u.run)).or_insert("N");
        if order(letter(u)) < order(st) {
            *st = letter(u);
        }
        cell.share += u.share;
    }
    let runs: Vec<String> = r.runs.iter().map(|x| run_name(*x)).collect();
    s.push_str(
        "\n## Shape union per family\n\nThe distinct (point, weight shape) pairs the models ask of \
         each family: the compile-time and policy point the Venn reads (head_dim, formats, group \
         size, epilogue, ...) and, for weight-reading ops, the weight's N x K. Per run, the best \
         status over the pair's sites (E measured at this shape, M point measured elsewhere, U/O/V \
         not measured, N novel). Families ordered by the total step share they carry.\n",
    );
    let mut fams: Vec<(&String, f64)> = by
        .iter()
        .map(|(f, m)| (f, m.values().map(|c| c.share).sum()))
        .collect();
    fams.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (fam, total) in fams {
        let cells = &by[fam];
        let measured = cells
            .values()
            .filter(|c| c.runs.values().any(|v| *v == "E"))
            .count();
        let _ = writeln!(
            s,
            "\n### `{fam}`: {} points, {measured} measured at their shape somewhere, {:.2} \
             model-steps summed over runs\n\n| Point | N x K | Models | {} |\n|---|---|---|{}",
            cells.len(),
            total,
            runs.join(" | "),
            "---|".repeat(runs.len())
        );
        for ((point, shape), c) in cells {
            let models: Vec<String> = c.models.iter().map(|m| format!("M{}", m + 1)).collect();
            let st: Vec<&str> = runs
                .iter()
                .map(|x| c.runs.get(x).copied().unwrap_or(""))
                .collect();
            let _ = writeln!(
                s,
                "| {point} | {shape} | {} | {} |",
                models.join(" "),
                st.join(" | ")
            );
        }
    }
}

#[derive(Default)]
struct Cell {
    models: BTreeSet<usize>,
    runs: BTreeMap<String, &'static str>,
    share: f64,
}

fn order(l: &str) -> u8 {
    ["E", "S", "M", "U", "O", "V", "N"]
        .iter()
        .position(|x| *x == l)
        .unwrap_or(6) as u8
}

/// 2026-10-10: Per-key totals at the two quoted runs (C1, C16), and who contributes.
#[derive(Default)]
struct Ranked {
    share: BTreeMap<String, f64>,
    models: BTreeSet<usize>,
    sites: BTreeSet<String>,
}

fn ranked_rows<K: Ord + Clone>(
    r: &UnionReport,
    keep: impl Fn(&SiteUse) -> Option<K>,
) -> Vec<(K, Ranked)> {
    let runs = key_runs(r);
    let mut by: BTreeMap<K, Ranked> = BTreeMap::new();
    for u in r.uses.iter().filter(|u| runs.contains(&u.run)) {
        let Some(k) = keep(u) else { continue };
        let e = by.entry(k).or_default();
        *e.share.entry(run_name(u.run)).or_insert(0.0) += u.share;
        e.models.insert(u.model);
        e.sites.insert(format!("M{} {}", u.model + 1, u.site));
    }
    let mut rows: Vec<(K, Ranked)> = by.into_iter().collect();
    rows.sort_by(|a, b| {
        let t = |x: &Ranked| x.share.values().sum::<f64>();
        t(&b.1).total_cmp(&t(&a.1))
    });
    rows
}

fn cells(r: &UnionReport, x: &Ranked) -> String {
    let named: Vec<String> = key_runs(r)
        .iter()
        .map(|run| pct(x.share.get(&run_name(*run)).copied().unwrap_or(0.0)))
        .collect();
    let models: Vec<String> = x.models.iter().map(|m| format!("M{}", m + 1)).collect();
    format!("{} | {}", named.join(" | "), models.join(" "))
}

/// 2026-10-10: Every policy variant and parameterization opportunity, deduplicated by family and
/// the parameter values it asks for, ranked by the step share it would bring into the family.
pub(super) fn plan(s: &mut String, r: &UnionReport) {
    let rows = ranked_rows(r, |u| {
        matches!(
            u.class,
            Class::PolicyVariant | Class::ParameterizationOpportunity
        )
        .then(|| {
            let what: Vec<String> = u
                .diffs
                .iter()
                .map(|d| format!("{} {} (have {})", d.param, d.target, d.other))
                .collect();
            let kinds: BTreeSet<&str> = u.diffs.iter().map(|d| d.kind.name()).collect();
            (
                u.family.clone().unwrap_or_default(),
                what.join("; "),
                kinds.into_iter().collect::<Vec<_>>().join("+"),
            )
        })
    });
    let names: Vec<String> = key_runs(r).iter().map(|x| run_name(*x)).collect();
    let _ = writeln!(
        s,
        "\n## Consolidated parameterization plan\n\nEvery primary policy variant and \
         parameterization opportunity over all models, deduplicated by family and the parameter \
         values it asks for, ranked by the summed step share (model-steps) it moves onto the \
         family at {}.\n\n| # | Family | Parameter(s) to add | Kind | {} | Models | Sites |\n\
         |---|---|---|---|{}---|---|",
        names.join(" and "),
        names.join(" | "),
        "---:|".repeat(names.len())
    );
    for (i, ((fam, what, kind), x)) in rows.iter().enumerate() {
        let _ = writeln!(
            s,
            "| {} | `{fam}` | {what} | {kind} | {} | {} |",
            i + 1,
            cells(r, x),
            x.sites.len()
        );
    }
}

/// 2026-10-10: Every op no family implements at its formats, ranked by step share.
pub(super) fn novel(s: &mut String, r: &UnionReport) {
    let rows = ranked_rows(r, |u| {
        (u.class == Class::Novel).then(|| {
            (
                u.op.clone(),
                u.weight.clone().unwrap_or_else(|| "-".into()),
                u.activation.clone().unwrap_or_else(|| "-".into()),
            )
        })
    });
    let names: Vec<String> = key_runs(r).iter().map(|x| run_name(*x)).collect();
    let _ = writeln!(
        s,
        "\n## Novel kernels\n\nOps no family implements with these formats, ranked by summed \
         step share.\n\n| # | Op | Weight | Activation | {} | Models | Sites |\n|---|---|---|---|{}---|---|",
        names.join(" | "),
        "---:|".repeat(names.len())
    );
    for (i, ((op, w, a), x)) in rows.iter().enumerate() {
        let sites: Vec<String> = x.sites.iter().take(4).cloned().collect();
        let more = x.sites.len().saturating_sub(4);
        let _ = writeln!(
            s,
            "| {} | `{op}` | {w} | {a} | {} | {}{} |",
            i + 1,
            cells(r, x),
            sites.join(", "),
            if more > 0 {
                format!(" +{more}")
            } else {
                String::new()
            }
        );
    }
    let _ = writeln!(
        s,
        "\nModels: {}.",
        r.models
            .iter()
            .enumerate()
            .map(|(i, m)| format!("M{} `{}`", i + 1, short(&m.recipe)))
            .collect::<Vec<_>>()
            .join(", ")
    );
}
