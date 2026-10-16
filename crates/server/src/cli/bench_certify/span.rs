// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: How many boxes a certify unit holds.
//!
//! Owner: server CLI (`met benchmark certify`).
//! A unit's width is the `min_nodes` of the recipe its gate's default entry serves on
//! this hardware class: the recipe renders `min_nodes` as the world size
//! (`Recipe::argv`), and `max_nodes` only bounds it (refused at parse when below). A
//! unit whose entry names no recipe is one box; the run itself refuses such an entry
//! when it must self-start (`bench_resolve`). The widths come in as a closure, so
//! nothing here reads the tree.
//! Invariants:
//! - A unit wider than the campaign's boxes is refused before anything runs: a 3-node
//!   recipe on a 2-box fleet would otherwise sit pending to the end of the campaign.

use anyhow::{Context, Result, bail};

use super::plan::Unit;

/// 2026-10-15: Set each unit's `nodes` from `width_of(unit.id)` and refuse a unit wider
/// than `boxes`, the boxes this campaign may use (`--with-nodes` plus this one unless
/// `--remote-only`).
///
/// # Errors
/// When `width_of` fails for a unit, or a unit needs more boxes than `boxes`; the
/// message names every such unit.
pub fn hold(
    mut units: Vec<Unit>,
    width_of: &dyn Fn(&'static str) -> Result<usize>,
    boxes: usize,
) -> Result<Vec<Unit>> {
    let mut wide = Vec::new();
    for u in &mut units {
        u.nodes = width_of(u.id).with_context(|| format!("how many nodes does {} hold?", u.id))?;
        if u.nodes > boxes {
            wide.push(format!("{} needs {} nodes", u.label(), u.nodes));
        }
    }
    if !wide.is_empty() {
        bail!(
            "{}; this campaign has {boxes} box(es). A multi-node recipe is certified as one \
             unit holding that many boxes: pass them with --with-nodes",
            wide.join(", ")
        );
    }
    Ok(units)
}

#[cfg(test)]
mod tests {
    use super::super::plan::Estimate;
    use super::*;
    use metrale_bench::hardware::policy::Sensitivity;

    fn unit(id: &'static str) -> Unit {
        Unit {
            id,
            group: None,
            shard: None,
            class: Sensitivity::Speed,
            estimate: Estimate::Declared(60),
            needs_confirmation: false,
            serve_allowance_s: 600,
            nodes: 1,
        }
    }

    fn widths(id: &str) -> Result<usize> {
        Ok(if id.ends_with("-glm-512k") { 3 } else { 1 })
    }

    /// 2026-10-15: Each unit gets its own recipe's width; single-box units stay 1.
    #[test]
    fn each_unit_holds_its_recipes_min_nodes() {
        let held = hold(
            vec![unit("ttft-cold-glm-512k"), unit("decode-floor")],
            &widths,
            3,
        )
        .unwrap();
        assert_eq!(
            held.iter().map(|u| (u.id, u.nodes)).collect::<Vec<_>>(),
            vec![("ttft-cold-glm-512k", 3), ("decode-floor", 1)]
        );
    }

    /// 2026-10-15: A 3-node unit on two boxes is refused, naming the unit and its
    /// width. Control: the same plan on three boxes is accepted, so the refusal is
    /// the comparison and not the unit.
    #[test]
    fn a_unit_wider_than_the_campaign_is_refused_before_anything_runs() {
        let plan = || vec![unit("ttft-cold-glm-512k"), unit("decode-floor")];
        let e = format!("{:#}", hold(plan(), &widths, 2).unwrap_err());
        assert!(e.contains("ttft-cold-glm-512k needs 3 nodes"), "{e}");
        assert!(e.contains("has 2 box(es)"), "{e}");
        assert!(!e.contains("decode-floor"), "{e}");
        assert!(hold(plan(), &widths, 3).is_ok());
    }

    /// 2026-10-15: A recipe that cannot be read stops the plan; it is never read as one
    /// node.
    #[test]
    fn an_unreadable_width_is_an_error_not_one_node() {
        let broken = |_: &str| -> Result<usize> { bail!("recipe missing from the tree") };
        let e = format!(
            "{:#}",
            hold(vec![unit("decode-floor")], &broken, 3).unwrap_err()
        );
        assert!(e.contains("recipe missing from the tree"), "{e}");
    }
}
