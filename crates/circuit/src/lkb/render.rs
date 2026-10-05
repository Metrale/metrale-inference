// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `met circuit lkb` output: the Markdown report, and the campaign-ledger fields as
//! TOML (the names a hardware campaign's ledger uses).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Deterministic: the same [`Lkb`] renders the same bytes.
//! - Percentages are of the estimated step, one decimal, in report-run order (C1 decode,
//!   C16 and C128 multi-sequence).

use std::fmt::Write as _;

use super::Lkb;

fn pct(x: f64) -> String {
    // 2026-10-05: `+ 0.0` turns a sum of no terms (`-0.0`) into `0.0`.
    format!("{:.1}", 100.0 * x + 0.0)
}

fn joined(l: &Lkb, f: impl Fn(&super::Coverage) -> f64) -> String {
    l.coverage
        .iter()
        .map(|c| pct(f(c)))
        .collect::<Vec<_>>()
        .join("/")
}

fn list(items: impl IntoIterator<Item = String>) -> String {
    let v: Vec<String> = items.into_iter().map(|s| format!("`{s}`")).collect();
    if v.is_empty() {
        "none".to_string()
    } else {
        v.join(", ")
    }
}

/// 2026-10-05: The Markdown report.
pub fn render_markdown(l: &Lkb) -> String {
    let mut s = String::new();
    let class = &l.chain[0];
    let inherits = if l.chain.len() > 1 {
        format!(" (realization inherits {})", l.chain[1..].join(" → "))
    } else {
        String::new()
    };
    let _ = writeln!(
        s,
        "# LKB on {class}{inherits}: {} on {}\n",
        l.model, l.device
    );
    let _ = writeln!(s, "Regenerate: `{}`\n", l.command);
    let _ = writeln!(
        s,
        "Terms: book/src/architecture/lkb.md. Shares are of the estimated step (roofline).\n"
    );
    let _ = writeln!(s, "## Presentation on {class}\n");
    let _ = writeln!(s, "| families | points | instantiation | copy | branch |");
    let _ = writeln!(s, "|---:|---:|---:|---:|---:|");
    let n = |k: &str| l.points.get(k).copied().unwrap_or(0);
    let _ = writeln!(
        s,
        "| {} | {} | {} | {} | {} |\n",
        l.families,
        l.points.values().sum::<usize>(),
        n("instantiation"),
        n("copy"),
        n("branch")
    );
    let _ = writeln!(s, "## Coverage\n");
    let _ = writeln!(
        s,
        "| run | LKB coverage | measured on {class} | uncovered (novel) | groups: reference / bit-identical / differs / uncovered |"
    );
    let _ = writeln!(s, "|---|---:|---:|---:|---|");
    for c in &l.coverage {
        let g = |k: &str| c.groups.get(k).copied().unwrap_or(0);
        let _ = writeln!(
            s,
            "| {} n={} | {}% | {}% | {}% | {} / {} / {} / {} |",
            c.run.mode.name(),
            c.run.rows,
            pct(c.lkb),
            pct(c.measured),
            pct(c.uncovered),
            g("reference"),
            g("bit_identical"),
            g("differs"),
            g("uncovered")
        );
    }
    let _ = writeln!(
        s,
        "\nGenerators used: {}\n\nRelations used: {}\n",
        list(l.generators_used.iter().cloned()),
        list(l.relations_used.iter().cloned())
    );
    let _ = writeln!(
        s,
        "## LKB residual on {class}: {} sources, {} lines; {} copy points\n",
        l.residual.len(),
        l.residual_lines(),
        l.copy_points.len()
    );
    let _ = writeln!(
        s,
        "Compiled modules of `kernels/{class}/` that no family names, and the family points \
         realized as per-point copies. Their share of the step is not modelled: no lowering \
         rule names them, so the plan never runs them.\n"
    );
    if !l.residual.is_empty() {
        let _ = writeln!(s, "| module | source | lines | tier | shadows |");
        let _ = writeln!(s, "|---|---|---:|---|---|");
        for r in &l.residual {
            let _ = writeln!(
                s,
                "| {} | `{}` | {} | {} | {} |",
                r.module,
                r.path,
                r.lines,
                r.tier.name(),
                r.shadow.as_deref().unwrap_or("-")
            );
        }
        s.push('\n');
    }
    if !l.copy_points.is_empty() {
        let _ = writeln!(s, "| family | copy point | sources |");
        let _ = writeln!(s, "|---|---|---|");
        for c in &l.copy_points {
            let _ = writeln!(
                s,
                "| {} | {} | {} |",
                c.family,
                c.values,
                c.files.join(", ")
            );
        }
        s.push('\n');
    }
    s
}

/// 2026-10-05: The `## LKB on <class>` section of `met circuit plan`'s report: the coverage
/// line and the residual's size; `met circuit lkb` prints the details.
pub fn report_section(s: &mut String, r: &crate::hardware::HwReport) {
    let l = super::from_report(r, &std::collections::BTreeMap::new(), String::new());
    let class = &l.chain[0];
    let inherits = if l.chain.len() > 1 {
        format!(" (realization inherits {})", l.chain[1..].join(" → "))
    } else {
        String::new()
    };
    let _ = writeln!(
        s,
        "## LKB on {class}{inherits}

LKB coverage {}: {}% of the step, measured on this class \
         {}% (book/src/architecture/lkb.md; placeholder rows count as uncovered). Generators \
         used: {} families. Relations used: {}. LKB residual on {class}: {} sources, {} lines; \
         {} copy points. Details: `met circuit lkb --checkpoint {} --hardware {} --precision {}`.\n",
        runs(&l),
        joined(&l, |c| c.lkb).replace('/', " / "),
        joined(&l, |c| c.measured).replace('/', " / "),
        l.generators_used.len(),
        list(l.relations_used.iter().cloned()),
        l.residual.len(),
        l.residual_lines(),
        l.copy_points.len(),
        r.model.checkpoint,
        r.resolved.device.id,
        r.model.precision_choice.name()
    );
}

fn runs(l: &Lkb) -> String {
    l.coverage
        .iter()
        .map(|c| format!("{} n={}", c.run.mode.name(), c.run.rows))
        .collect::<Vec<_>>()
        .join(" / ")
}

/// 2026-10-05: The campaign-ledger fields, as TOML.
pub fn render_toml(l: &Lkb) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# {}", l.command);
    let _ = writeln!(s, "lkb_class = {:?}", l.chain[0]);
    let _ = writeln!(s, "lkb_device = {:?}", l.device);
    let _ = writeln!(s, "lkb_coverage_pct = {:?}", joined(l, |c| c.lkb));
    let _ = writeln!(s, "lkb_measured_pct = {:?}", joined(l, |c| c.measured));
    let _ = writeln!(s, "residual_count = {}", l.residual.len());
    let _ = writeln!(s, "residual_loc = {}", l.residual_lines());
    let _ = writeln!(s, "residual_copy_points = {}", l.copy_points.len());
    let _ = writeln!(
        s,
        "lkb_relations_used = [{}]",
        l.relations_used
            .iter()
            .map(|r| format!("{r:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    s
}
