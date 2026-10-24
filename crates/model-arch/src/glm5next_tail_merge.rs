// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The sub-chunk plan of a GLM prefill pass under `--prefill-tail-mode merge`: the
//! pass's rows in sub-chunks from row 0, each in-pass capture point a cut of the KDA conv and
//! recurrence of the sub-chunk that holds it (`glm5next_kda/cut.rs`), instead of the split's
//! segments ending and starting at the points (`glm5next_layer/steps/prefill_capture.rs`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`merge_plan`] covers `0..n` once, in order, in sub-chunks of at most `rows`.
//! - The rows before a point start their sub-chunks where the split's segment ending there
//!   does (see [`merge_plan`]). 2026-10-14: The captured state is the split's bits only when
//!   that sub-chunk and the split's segment fall in the same row class of the row-count-dependent
//!   GEMMs (dequant W4A16 at 256 rows, FP8 at 1024). Across a class boundary it can differ in
//!   low bits (disclosure PREFILL-TAILMERGE-1).

/// 2026-10-14: The `(start, end, cuts)` sub-chunks of a pass of `n` rows, `rows` per sub-chunk
/// from row 0, with each capture point of `points` (sorted, strictly inside `0..n`, tagged with
/// its slot `W`) a cut `(point - start, slot)` of the sub-chunk holding it. A sub-chunk that holds
/// a point and would end before the next point ends at its last point instead, so the rows
/// between two points start a sub-chunk where the split's segment does, as the module's
/// invariant says (with its row-class limit). Pure.
pub fn merge_plan<W: Copy>(
    n: usize,
    rows: usize,
    points: &[(usize, W)],
) -> Vec<(usize, usize, Vec<(usize, W)>)> {
    let rows = rows.max(1);
    let mut out = Vec::with_capacity(n.div_ceil(rows) + points.len());
    let (mut t, mut i) = (0usize, 0usize);
    while t < n {
        let mut end = (t + rows).min(n);
        let mut j = i;
        while j < points.len() && points[j].0 <= end {
            j += 1;
        }
        if j > i && j < points.len() {
            end = points[j - 1].0;
        }
        let cuts = points[i..j]
            .iter()
            .filter(|&&(p, _)| p > t)
            .map(|&(p, w)| (p - t, w))
            .collect();
        out.push((t, end, cuts));
        (t, i) = (end, j);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::merge_plan;

    type Plan = Vec<(usize, usize, Vec<(usize, char)>)>;

    /// 2026-10-14: What a merged plan must be, checked independently of `merge_plan`: contiguous
    /// sub-chunks of at most `rows` covering `0..n`, each point a cut of the sub-chunk holding
    /// it, and a sub-chunk ending short of `rows` rows and of `n` only at a point whose next point
    /// lies past `rows` rows from its start (the one split that keeps the later point's captured
    /// state the split's). The split's segments break the last rule.
    fn check(plan: &Plan, n: usize, rows: usize, points: &[(usize, char)]) -> Result<(), String> {
        let mut t = 0;
        for (s, e, cuts) in plan {
            if *s != t || e <= s || e - s > rows {
                return Err(format!("sub-chunk [{s}, {e}) after row {t}"));
            }
            let want: Vec<(usize, char)> = points
                .iter()
                .filter(|&&(p, _)| p > *s && p <= *e)
                .map(|&(p, w)| (p - s, w))
                .collect();
            if *cuts != want {
                return Err(format!(
                    "sub-chunk [{s}, {e}) cuts {cuts:?}, points {want:?}"
                ));
            }
            if *e < (s + rows).min(n) {
                let next = points.iter().find(|&&(p, _)| p > *e).map(|&(p, _)| p);
                let at_point = cuts.last().is_some_and(|&(r, _)| s + r == *e);
                if !at_point || next.is_none_or(|p| p <= s + rows) {
                    return Err(format!("sub-chunk [{s}, {e}) split short of {rows} rows"));
                }
            }
            t = *e;
        }
        if t != n {
            return Err(format!("plan ends at {t}, not {n}"));
        }
        Ok(())
    }

    /// 2026-10-14: A short prompt's tail split (n = 256, point 240, 8208-row sub-chunks) is ONE
    /// sub-chunk cut at 240, where the split runs two; a long prompt keeps its sub-chunks from
    /// row 0 and cuts the one holding the point; no point is plain sub-chunks.
    #[test]
    fn the_tail_runs_inside_its_sub_chunk() {
        let pts = [(240, 'M')];
        let plan = merge_plan(256, 8208, &pts);
        assert_eq!(plan, vec![(0, 256, vec![(240, 'M')])]);
        check(&plan, 256, 8208, &pts).unwrap();
        let pts = [(19984, 'M')];
        let plan = merge_plan(20000, 8208, &pts);
        assert_eq!(
            plan,
            vec![
                (0, 8208, vec![]),
                (8208, 16416, vec![]),
                (16416, 20000, vec![(3568, 'M')])
            ]
        );
        check(&plan, 20000, 8208, &pts).unwrap();
        let plan = merge_plan::<char>(40, 16, &[]);
        assert_eq!(
            plan,
            vec![(0, 16, vec![]), (16, 32, vec![]), (32, 40, vec![])]
        );
        check(&plan, 40, 16, &[]).unwrap();
    }

    /// 2026-10-14: Two points with a sub-chunk boundary between them: the sub-chunk holding the
    /// early point ends there, so the rows to the main point start where the split's segment
    /// does; two points in one sub-chunk are two cuts of it; a point on a boundary is a cut at
    /// the end of the sub-chunk before it.
    #[test]
    fn a_sub_chunk_ends_at_a_point_before_a_boundary() {
        let pts = [(30, 'E'), (50, 'M')];
        let plan = merge_plan(100, 40, &pts);
        assert_eq!(
            plan,
            vec![
                (0, 30, vec![(30, 'E')]),
                (30, 70, vec![(20, 'M')]),
                (70, 100, vec![])
            ]
        );
        check(&plan, 100, 40, &pts).unwrap();
        let pts = [(3968, 'E'), (3984, 'M')];
        let plan = merge_plan(4000, 8208, &pts);
        assert_eq!(plan, vec![(0, 4000, vec![(3968, 'E'), (3984, 'M')])]);
        check(&plan, 4000, 8208, &pts).unwrap();
        let pts = [(32, 'M')];
        let plan = merge_plan(40, 32, &pts);
        assert_eq!(plan, vec![(0, 32, vec![(32, 'M')]), (32, 40, vec![])]);
        check(&plan, 40, 32, &pts).unwrap();
    }

    /// 2026-10-14: Negative control: the split's segments for the same pass, read as a plan,
    /// fail the check, so a merge path that fell back to the split would not pass for merged.
    #[test]
    fn the_split_segments_are_not_a_merged_plan() {
        let pts = [(240, 'M')];
        let split: Plan = vec![(0, 240, vec![(240, 'M')]), (240, 256, vec![])];
        let err = check(&split, 256, 8208, &pts).unwrap_err();
        assert!(err.contains("split short"), "{err}");
    }
}
