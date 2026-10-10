// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Selection keeps the latest non-throttled record, never lets a failing or
//! unavailable candidate win, keeps the default inside the noise band, calls a winner
//! bit-identical only on equal digests over equal class sets, marks default-less cells `new`,
//! and merges only identical decisions over adjacent swept rows.

use std::collections::BTreeMap;

use super::*;
use crate::envelope::schedules::Source;

const D: &str = "m::default";

fn cell(n: u64, rows: u64) -> Cell {
    Cell {
        op: "linear".into(),
        weight: "nvfp4/g16".into(),
        activation: "bf16".into(),
        k: 5120,
        n,
        rows,
    }
}

/// 2026-10-10: A passing record at shape n=17408 on dgx1 at 03:00Z, served cell with default
/// `D`, one digest per class unique to the kernel (so nothing is bit-identical unless a test
/// says so). Tests change the fields they are about.
fn rec(rows: u64, kernel: &str, times: &[f64]) -> Measurement {
    Measurement {
        schema: 1,
        hardware: "gb10".into(),
        host: "dgx1".into(),
        cell: cell(17408, rows),
        kernel: kernel.into(),
        family: format!("fam_{}", kernel.split("::").next().unwrap_or(kernel)),
        default: Some(D.into()),
        verdict: Verdict::Pass,
        detail: String::new(),
        digests: [("normal", kernel), ("outlier", kernel)]
            .iter()
            .map(|(c, k)| (c.to_string(), format!("sha-{k}-{c}")))
            .collect(),
        time_us: times.to_vec(),
        floor_us: 1.0,
        throttled: false,
        temp_c: 60.0,
        closure: "gb10/x/y=0".into(),
        at: "2026-10-11T03:00:00Z".into(),
    }
}

fn at(mut m: Measurement, at: &str, host: &str) -> Measurement {
    m.at = at.into();
    m.host = host.into();
    m
}

fn only(sel: &Selection) -> &Decision {
    assert_eq!(sel.decisions.len(), 1, "{sel:#?}");
    &sel.decisions[0]
}

#[test]
fn the_latest_non_throttled_record_counts_by_time_not_by_spelling() {
    let older = at(rec(1, "a::fast", &[10.0]), "2026-10-11T03:00:00Z", "dgx1");
    // Half a second later on the other box: string order would put it first.
    let newer = at(rec(1, "a::fast", &[20.0]), "2026-10-11T03:00:00.5Z", "dgx2");
    let mut hot = at(rec(1, "a::fast", &[1.0]), "2026-10-11T05:00:00Z", "dgx1");
    hot.throttled = true;
    let def = rec(1, D, &[15.0]);
    for order in [
        vec![older.clone(), newer.clone(), hot.clone(), def.clone()],
        vec![def.clone(), hot.clone(), newer.clone(), older.clone()],
    ] {
        let sel = select(&order).unwrap();
        let d = only(&sel);
        assert_eq!((d.kernel.as_str(), d.numerics), (D, Numerics::Same));
        assert!(
            sel.reruns.is_empty(),
            "a throttled rerun beside a counted record is no rerun"
        );
    }
    // Once the slow rerun is gone, the older fast record counts and wins.
    let sel = select(&[older, hot, def]).unwrap();
    assert_eq!(only(&sel).kernel, "a::fast");
    assert_eq!(only(&sel).measured, "dgx1 2026-10-11T03:00:00Z");
}

#[test]
fn a_throttled_only_candidate_is_a_rerun_and_never_wins() {
    let mut hot1 = rec(1, "a::fast", &[1.0]);
    hot1.throttled = true;
    let hot2 = at(hot1.clone(), "2026-10-11T04:00:00Z", "dgx2");
    let sel = select(&[hot1.clone(), hot2, rec(1, D, &[15.0])]).unwrap();
    assert_eq!(only(&sel).kernel, D);
    assert_eq!(
        sel.reruns,
        vec![Rerun {
            cell: cell(17408, 1),
            kernel: "a::fast".into(),
            hosts: vec!["dgx1".into(), "dgx2".into()],
        }]
    );
    // A throttled-only default leaves the cell undecided, even beside a fast passing candidate.
    let mut hot_def = rec(1, D, &[15.0]);
    hot_def.throttled = true;
    let sel = select(&[hot_def, rec(1, "b::other", &[5.0])]).unwrap();
    assert!(sel.decisions.is_empty());
    assert_eq!(sel.undecided[0].why, NoWinner::DefaultNeedsRerun(D.into()));
}

#[test]
fn a_failing_or_unavailable_candidate_never_wins_even_when_fastest() {
    let mut fail = rec(1, "f::fail", &[1.0]);
    fail.verdict = Verdict::Fail;
    fail.detail = "outside its contract on outlier".into();
    let mut unav = rec(1, "u::unav", &[0.5]);
    unav.verdict = Verdict::Unavailable;
    let sel = select(&[
        fail.clone(),
        unav.clone(),
        rec(1, "p::pass", &[10.0]),
        rec(1, D, &[20.0]),
    ])
    .unwrap();
    assert_eq!(only(&sel).kernel, "p::pass");
    assert_eq!(only(&sel).numerics, Numerics::Differs);

    // No passing candidate at a new cell: undecided, with every counted candidate's verdict.
    fail.default = None;
    unav.default = None;
    let sel = select(&[fail, unav]).unwrap();
    assert!(sel.decisions.is_empty());
    assert_eq!(
        sel.undecided[0].why,
        NoWinner::NoPassingCandidate(vec![
            (
                "f::fail".into(),
                Verdict::Fail,
                "outside its contract on outlier".into()
            ),
            ("u::unav".into(), Verdict::Unavailable, String::new()),
        ])
    );

    // A default that fails its contract keeps today's routing: no decision at all.
    let mut bad_def = rec(1, D, &[]);
    bad_def.verdict = Verdict::Fail;
    let sel = select(&[bad_def, rec(1, "p::pass", &[1.0])]).unwrap();
    assert!(matches!(
        sel.undecided[0].why,
        NoWinner::DefaultDidNotPass {
            verdict: Verdict::Fail,
            ..
        }
    ));
    // And a default with no record at all.
    let sel = select(&[rec(1, "p::pass", &[1.0])]).unwrap();
    assert_eq!(sel.undecided[0].why, NoWinner::DefaultNotMeasured(D.into()));
}

#[test]
fn the_noise_band_keeps_the_default() {
    let def = rec(1, D, &[100.0, 100.0, 100.0]);
    let win = |times: &[f64]| {
        let sel = select(&[def.clone(), rec(1, "w::win", times)]).unwrap();
        (only(&sel).kernel.clone(), only(&sel).median_us)
    };
    // 2.5% faster with tight repetitions: inside the 3% floor.
    assert_eq!(win(&[97.5, 97.5, 97.5]), (D.to_string(), 100.0));
    // Exactly 3% is "at least 3%": the winner.
    assert_eq!(win(&[97.0, 97.0, 97.0]), ("w::win".to_string(), 97.0));
    // 10% faster but the winner's spread is 12%: the margin is 24%, the default stays.
    assert_eq!(win(&[90.0, 85.0, 95.8]), (D.to_string(), 100.0));
    // The default's own spread counts as well.
    let noisy = rec(1, D, &[100.0, 80.0, 120.0]);
    let sel = select(&[noisy, rec(1, "w::win", &[60.0, 60.0, 60.0])]).unwrap();
    assert_eq!(only(&sel).kernel, D, "a 40% gain is inside 2 x 40% spread");
    // 10% faster with tight repetitions: the winner.
    assert_eq!(win(&[90.0, 90.0, 90.0]).0, "w::win");
}

#[test]
fn bit_identical_needs_equal_digests_on_every_class() {
    let def = rec(1, D, &[100.0]);
    let classify = |digests: &[(&str, &str)]| {
        let mut w = rec(1, "w::win", &[50.0]);
        w.digests = digests
            .iter()
            .map(|(c, d)| (c.to_string(), d.to_string()))
            .collect();
        let sel = select(&[def.clone(), w]).unwrap();
        (only(&sel).numerics, only(&sel).enabled)
    };
    let (n, o) = (format!("sha-{D}-normal"), format!("sha-{D}-outlier"));
    assert_eq!(
        classify(&[("normal", &n), ("outlier", &o)]),
        (Numerics::BitIdentical, Enabled::Default)
    );
    assert_eq!(
        classify(&[("normal", &n), ("outlier", "other")]),
        (Numerics::Differs, Enabled::OptIn),
        "one differing class"
    );
    assert_eq!(
        classify(&[("normal", &n)]),
        (Numerics::Differs, Enabled::OptIn),
        "a class the winner did not record"
    );
    let mut empty = rec(1, D, &[100.0]);
    empty.digests.clear();
    let mut w = rec(1, "w::win", &[50.0]);
    w.digests.clear();
    assert_eq!(
        only(&select(&[empty, w]).unwrap()).numerics,
        Numerics::Differs,
        "no classes"
    );
}

#[test]
fn a_cell_with_no_default_is_new_and_opt_in() {
    let mut a = rec(1, "a::a", &[10.0]);
    let mut b = rec(1, "b::b", &[12.0]);
    a.default = None;
    b.default = None;
    let sel = select(&[a, b]).unwrap();
    let d = only(&sel);
    assert_eq!(
        (d.kernel.as_str(), d.numerics, d.enabled),
        ("a::a", Numerics::New, Enabled::OptIn)
    );
    assert_eq!(d.default_us, None);
    let e = &merge_rows(&sel)[0];
    assert_eq!((e.default.as_str(), e.default_us), ("", 0.0));
}

#[test]
fn the_row_merge_joins_only_identical_decisions_on_adjacent_swept_rows() {
    let mut recs = Vec::new();
    for rows in [1, 2, 4, 16] {
        recs.push(rec(rows, D, &[100.0 * rows as f64]));
        recs.push(rec(rows, "x::x", &[50.0 * rows as f64]));
    }
    // Row 8 is swept but nothing passes there: it breaks the run.
    let mut fail = rec(8, "x::x", &[]);
    fail.verdict = Verdict::Fail;
    let mut fail_def = rec(8, D, &[]);
    fail_def.verdict = Verdict::Fail;
    recs.extend([fail, fail_def]);
    // A second shape: row 1 differs, row 2 is bit-identical: two entries.
    let mut other1 = rec(1, "x::x", &[1.0]);
    other1.cell.n = 4096;
    let mut def1 = rec(1, D, &[2.0]);
    def1.cell.n = 4096;
    let mut other2 = rec(2, "x::x", &[1.0]);
    other2.cell.n = 4096;
    let mut def2 = rec(2, D, &[2.0]);
    def2.cell.n = 4096;
    def2.digests = other2.digests.clone();
    recs.extend([other1, def1, other2, def2]);

    let sel = select(&recs).unwrap();
    let rows: Vec<(u64, [u64; 2], Numerics)> = merge_rows(&sel)
        .iter()
        .map(|e| (e.n, e.rows, e.numerics))
        .collect();
    assert_eq!(
        rows,
        vec![
            (4096, [1, 1], Numerics::Differs),
            (4096, [2, 2], Numerics::BitIdentical),
            (17408, [1, 4], Numerics::Differs),
            (17408, [16, 16], Numerics::Differs),
        ]
    );
    // A merged entry carries its largest launch's times.
    let merged = &merge_rows(&sel)[2];
    assert_eq!((merged.median_us, merged.default_us), (200.0, 400.0));
}

#[test]
fn records_that_cannot_be_selected_from_are_refused() {
    assert_eq!(select(&[]), Err(SelectError::Empty));
    let mut other_hw = rec(1, "a::a", &[1.0]);
    other_hw.hardware = "h100".into();
    assert!(matches!(
        select(&[rec(1, D, &[1.0]), other_hw]),
        Err(SelectError::MixedHardware(..))
    ));
    let mut other_def = rec(1, "a::a", &[1.0]);
    other_def.default = None;
    assert!(matches!(
        select(&[rec(1, D, &[1.0]), other_def]),
        Err(SelectError::DefaultDisagrees { .. })
    ));
    for bad in [
        "2026-10-11T03:00:00+00:00",
        "2026-10-11 03:00:00Z",
        "2026-10-11T03:00:00.Z",
    ] {
        assert!(
            matches!(
                select(&[at(rec(1, D, &[1.0]), bad, "dgx1")]),
                Err(SelectError::BadRecord { .. })
            ),
            "{bad}"
        );
    }
    let a = rec(1, D, &[1.0]);
    let b = at(rec(1, D, &[2.0]), &a.at, "dgx2");
    assert!(
        matches!(select(&[a, b]), Err(SelectError::BadRecord { .. })),
        "same instant"
    );
}

#[test]
fn the_file_needs_a_source_for_every_winning_family() {
    let sel = select(&[rec(1, D, &[100.0]), rec(1, "x::x", &[50.0])]).unwrap();
    assert_eq!(families(&sel), vec!["fam_x".to_string()]);
    let src = Source {
        files: vec!["kernels/gb10/common/x.cu".into()],
        sha256: "a".repeat(64),
    };
    assert!(schedules(&sel, "met envelope schedules", BTreeMap::new()).is_err());
    let s = schedules(
        &sel,
        "met envelope schedules",
        [("fam_x".to_string(), src)].into(),
    )
    .unwrap();
    assert_eq!(s.schedule.len(), 1);
}
