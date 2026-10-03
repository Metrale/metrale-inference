// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The "Tensor-core policy" section of a hardware report: the class's requirements,
//! how many covered nodes run on tensor cores at each report run, and every covered site that
//! runs off them under a stated exemption (the backlog included). Violations never reach a
//! report: they refuse the plan ([`super::HwError::TensorCore`]).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: deterministic text from the report's values.

use std::fmt::Write as _;

use super::HwReport;
use super::tc_policy::{TcPolicy, Weights};

fn weights(w: &Weights) -> String {
    match w {
        Weights::Any => "any".into(),
        Weights::Only(set) => set.iter().map(|f| f.name()).collect::<Vec<_>>().join(", "),
    }
}

fn op_names(ops: &[super::tc_policy::OpMatch]) -> String {
    ops.iter()
        .map(|o| match o.role {
            Some(r) => format!("{}:{}", o.base, r.name()),
            None => o.base.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 2026-10-02: Append the section.
pub(super) fn section(s: &mut String, r: &HwReport) {
    let class = &r.resolved.chain[0];
    let _ = writeln!(s, "## Tensor-core policy\n");
    let Some(policy) = &class.tensor_core else {
        let _ = writeln!(
            s,
            "kernels/{}/HARDWARE.toml states no `[tensor_core_policy]`: nothing is enforced on this class. The gap report names the compute unit of every planned group.\n",
            class.name
        );
        return;
    };
    let _ = writeln!(
        s,
        "kernels/{}/HARDWARE.toml `[tensor_core_policy]`: a plan that runs one of these ops off tensor cores is refused unless an exemption lists the op, mode, rows and kernels. Compute units are the kernel families' (`compute`, `mma`).\n",
        class.name
    );
    requirements(s, policy);
    let _ = writeln!(
        s,
        "| run | covered nodes | on tensor cores | exempted sites |\n|---|---:|---:|---:|"
    );
    for (t, a) in r.tables.iter().zip(&r.tensor_core) {
        let _ = writeln!(
            s,
            "| {} n={} | {} | {} | {} |",
            t.planned.plan.mode.name(),
            t.planned.plan.rows,
            a.covered,
            a.on_tensor_cores,
            a.exempted.len()
        );
    }
    let _ = writeln!(s);
    let rows: Vec<String> = r
        .tensor_core
        .iter()
        .flat_map(|a| a.exempted.iter())
        .map(|(f, i)| {
            let e = &policy.exempt[*i];
            let k: Vec<String> = f.kernels.iter().map(|k| k.to_string()).collect();
            format!(
                "| {} n={} | `{}` | {} | {} | {} | `{}` | #{} {} |",
                f.mode.name(),
                f.rows,
                f.site,
                f.op,
                f.weight.as_deref().unwrap_or("-"),
                f.unit,
                k.join("` + `"),
                i + 1,
                e.kind.name()
            )
        })
        .collect();
    let gaps: Vec<String> = r
        .tensor_core
        .iter()
        .flat_map(|a| a.gaps.iter())
        .map(|f| {
            format!(
                "| {} n={} | `{}` | {} | {} |",
                f.mode.name(),
                f.rows,
                f.site,
                f.op,
                f.weight.as_deref().unwrap_or("-")
            )
        })
        .collect();
    if !gaps.is_empty() {
        let _ = writeln!(
            s,
            "Covered sites no kernel of this class plans (gaps; their kernel is tensor-core work):\n\n| run | site | op | weight |\n|---|---|---|---|"
        );
        for g in gaps {
            let _ = writeln!(s, "{g}");
        }
        let _ = writeln!(s);
    }
    if rows.is_empty() {
        let _ = writeln!(s, "Every planned covered node runs on tensor cores.\n");
        return;
    }
    let _ = writeln!(
        s,
        "Covered sites off tensor cores, each under an exemption (`backlog` is a known violation awaiting a tensor-core kernel):\n\n| run | site | op | weight | unit | kernels | exemption |\n|---|---|---|---|---|---|---|"
    );
    for row in rows {
        let _ = writeln!(s, "{row}");
    }
    let _ = writeln!(s);
    exemptions(s, policy);
}

/// 2026-10-02: The class's exemptions, numbered as the rows above cite them.
fn exemptions(s: &mut String, policy: &TcPolicy) {
    let _ = writeln!(
        s,
        "| # | kind | ops | rows | reason |\n|---:|---|---|---|---|"
    );
    for (i, e) in policy.exempt.iter().enumerate() {
        let reason = match &e.evidence {
            Some(ev) => format!("{} (evidence: {ev})", e.reason),
            None => e.reason.clone(),
        };
        let _ = writeln!(
            s,
            "| {} | {} | {} | {}-{} | {} |",
            i + 1,
            e.kind.name(),
            op_names(&e.ops),
            e.rows.0,
            e.rows.1,
            reason.replace('|', "\\|")
        );
    }
    let _ = writeln!(s);
}

fn requirements(s: &mut String, policy: &TcPolicy) {
    let _ = writeln!(
        s,
        "| ops | modes | from rows | weights |\n|---|---|---:|---|"
    );
    for q in &policy.require {
        let modes: Vec<&str> = q.modes.iter().map(|m| m.name()).collect();
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} |",
            op_names(&q.ops),
            modes.join(", "),
            q.min_rows,
            weights(&q.weights)
        );
    }
    let _ = writeln!(s);
}
