// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The shapes a tensor-parallel rank fuses its plans at under `--forward circuit`
//! (`impl_circuit_ranks`): the decode row, each verify width the serve's speculation runs, and
//! the draft row when the drafter's block installs one. One source for the verify widths: the
//! serve's draft depth (`ForwardSelect::Circuit::verify_rows`), checked against the shapes the
//! family's sites install (`CircuitBindings::plan_shapes`).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A verify width the serve runs that no installed site covers is refused with
//!   `ErrorCode::CircuitPlanUncovered` naming the width and the draft depth, never fused and
//!   run as the layers' own code under the name `circuit`.
//! - No shape is fused that no site installs: an installed shape the serve never runs (another
//!   draft depth) is not fused either.

use metrale_circuit::Mode;
use metrale_error::{Error, ErrorCode, error};

/// 2026-10-23: The verify widths a draft depth of `num_drafts` runs: `num_drafts + 1` rows, and
/// every shallower depth the MTP ladder may step down to (2 rows at one draft).
pub fn speculative_verify_rows(num_drafts: usize) -> Vec<u64> {
    (2..=num_drafts as u64 + 1).collect()
}

/// 2026-10-23: The shapes to fuse, in order: decode n=1, the serve's verify widths
/// (`verify_rows`, each `num_drafts + 1` of a depth it runs), then draft n=1 when `installed`
/// holds it. `installed` is the union of the shapes the rank's sites install.
pub fn rank_plan_shapes(
    installed: &[(Mode, u64)],
    verify_rows: &[u64],
) -> Result<Vec<(Mode, u64)>, Error> {
    let mut shapes = vec![(Mode::Decode, 1)];
    for &rows in verify_rows {
        if !installed.contains(&(Mode::Verify, rows)) {
            let why = uncovered_why(rows, installed);
            return Err(error!(ErrorCode::CircuitPlanUncovered, "verify", rows, why));
        }
        shapes.push((Mode::Verify, rows));
    }
    if installed.contains(&(Mode::Draft, 1)) {
        shapes.push((Mode::Draft, 1));
    }
    Ok(shapes)
}

fn uncovered_why(rows: u64, installed: &[(Mode, u64)]) -> String {
    let covered: Vec<String> = installed
        .iter()
        .filter(|(m, _)| *m == Mode::Verify)
        .map(|(_, r)| r.to_string())
        .collect();
    format!(
        "draft depth {} (--num-drafts {}) verifies {rows} rows; this rank's sites install verify \
         at [{}] rows",
        rows.saturating_sub(1),
        rows.saturating_sub(1),
        covered.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_TEXT: [(Mode, u64); 2] = [(Mode::Decode, 1), (Mode::Verify, 2)];

    /// 2026-10-23: One draft (verify 2) fuses decode 1 and verify 2 and, with the draft head's
    /// site, draft 1; nothing else.
    #[test]
    fn a_depth_verifies_every_width_up_to_it() {
        assert_eq!(speculative_verify_rows(1), [2]);
        assert_eq!(speculative_verify_rows(3), [2, 3, 4]);
        assert!(speculative_verify_rows(0).is_empty());
    }

    #[test]
    fn one_draft_fuses_only_verify_two() {
        assert_eq!(
            rank_plan_shapes(&GLM_TEXT, &[2]).unwrap(),
            [(Mode::Decode, 1), (Mode::Verify, 2)]
        );
        let mut with_draft = GLM_TEXT.to_vec();
        with_draft.push((Mode::Draft, 1));
        assert_eq!(
            rank_plan_shapes(&with_draft, &[2]).unwrap(),
            [(Mode::Decode, 1), (Mode::Verify, 2), (Mode::Draft, 1)]
        );
        assert_eq!(
            rank_plan_shapes(&with_draft, &[]).unwrap(),
            [(Mode::Decode, 1), (Mode::Draft, 1)],
            "no speculation: no verify shape"
        );
    }

    /// 2026-10-23: A depth no site covers is refused with the plan-coverage code, naming the
    /// width and the depth.
    #[test]
    fn an_uncovered_depth_is_refused_naming_it() {
        let e = rank_plan_shapes(&GLM_TEXT, &[2, 3]).unwrap_err();
        assert_eq!(e.code(), ErrorCode::CircuitPlanUncovered);
        let text = e.to_string();
        assert!(
            text.contains("verify at 3 rows") && text.contains("--num-drafts 2"),
            "{text}"
        );
        assert!(text.contains("[2]"), "{text}");
    }
}
