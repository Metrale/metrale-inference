// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The FUSIONS check flags a rule whose rows a faster bit-identical sibling or
//! another routed kernel contradicts, and proposes a range only when every shape agrees.

use super::*;
use crate::envelope::schedules::{Enabled, Numerics, SCHEMA, Source};

const TC8: &str = "w4a16_gemv_tc::w4a16_gemv_tc8";
const SIB: &str = "w4a16_gemv_tc::w4a16_gemv_tc8_k2";
const B2: &str = "w4a16_gemv::w4a16_gemv_batch2";

fn rule(id: &str, func: &str, rows: [u64; 2], numerics: &str) -> String {
    format!(
        r#"
[[rule]]
id = "{id}"
pattern = [{{ op = "linear", roles = ["o", "down"], weight = "nvfp4/g16" }}]
kernels = [{{ module = "w4a16_gemv_tc", func = "{func}" }}]
repeat = "once"
emitter = "w4a16_gemv_batchm"
rows = [{}, {}]
modes = ["multi_seq"]
numerics = "{numerics}"
{}priority = 10
cite = "t"
"#,
        rows[0],
        rows[1],
        if numerics == "differs" {
            "lever = \"l\"\n"
        } else {
            ""
        }
    )
}

fn fusions(rules: &[String]) -> String {
    format!("schema = 1\n{}", rules.concat())
}

fn e(n: u64, rows: [u64; 2], kernel: &str, default: &str, numerics: Numerics) -> Schedule {
    Schedule {
        op: "linear".into(),
        weight: "nvfp4/g16".into(),
        activation: "bf16".into(),
        k: 5120,
        n,
        rows,
        kernel: kernel.into(),
        family: "w4a16_tc".into(),
        default: default.into(),
        numerics,
        enabled: Enabled::of(numerics),
        median_us: 10.0,
        default_us: if default.is_empty() { 0.0 } else { 12.0 },
        floor_us: 1.0,
        measured: "dgx1 2026-10-11T03:00:00Z".into(),
    }
}

fn schedules(schedule: Vec<Schedule>) -> Schedules {
    let s = Schedules {
        schema: SCHEMA,
        hardware: "gb10".into(),
        generated_by: "t".into(),
        sources: [(
            "w4a16_tc".to_string(),
            Source {
                files: vec!["kernels/gb10/common/w4a16_gemv_tc.cu".into()],
                sha256: "a".repeat(64),
            },
        )]
        .into(),
        schedule,
    };
    crate::envelope::schedules::check(&s).unwrap();
    s
}

fn one(text: &str, s: &Schedules) -> RuleCheck {
    let r = check_fusions(text, s).unwrap();
    assert_eq!(r.rules.len(), 1, "{r:#?}");
    r.rules.into_iter().next().unwrap_or_else(|| unreachable!())
}

#[test]
fn a_faster_bit_identical_sibling_is_a_disagreement_and_blocks_the_proposal() {
    let s = schedules(vec![
        e(4096, [2, 4], TC8, TC8, Numerics::Same),
        e(4096, [5, 8], SIB, TC8, Numerics::BitIdentical),
        e(17408, [2, 8], TC8, TC8, Numerics::Same),
    ]);
    let c = one(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 8], "reference")]),
        &s,
    );
    assert_eq!(c.shapes.len(), 2);
    assert_eq!(c.disagreements.len(), 1);
    assert_eq!(c.disagreements[0].rows, [5, 8]);
    assert_eq!(c.disagreements[0].why, Why::FasterSibling(SIB.into()));
    assert!(
        matches!(c.proposal, Proposal::None(ref why) if why.contains("[5, 8]")),
        "{c:#?}"
    );
    let md = check_fusions(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 8], "reference")]),
        &s,
    )
    .unwrap()
    .render();
    assert!(
        md.contains("| tc8 | w4a16_gemv_tc::w4a16_gemv_tc8 | [2, 8] | 2 | 1 | none:"),
        "{md}"
    );
    assert!(
        md.contains("rows [5, 8]: bit-identical faster sibling"),
        "{md}"
    );
}

#[test]
fn another_routed_default_inside_the_rows_is_a_disagreement() {
    let s = schedules(vec![
        e(4096, [2, 2], B2, B2, Numerics::Same),
        e(4096, [3, 8], TC8, TC8, Numerics::Same),
        // An opt-in winner leaves today's default launched: no disagreement.
        e(17408, [2, 8], SIB, TC8, Numerics::Differs),
    ]);
    let c = one(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 8], "reference")]),
        &s,
    );
    assert_eq!(c.disagreements.len(), 1);
    assert_eq!(c.disagreements[0].why, Why::NotRouted(B2.into()));
    assert_eq!(c.disagreements[0].rows, [2, 2]);
    assert!(
        matches!(c.proposal, Proposal::None(_)),
        "the shapes disagree at row 2"
    );
}

#[test]
fn a_range_is_proposed_only_when_every_shape_agrees() {
    let wide = schedules(vec![
        e(4096, [2, 4], TC8, TC8, Numerics::Same),
        e(4096, [5, 16], SIB, B2, Numerics::BitIdentical),
        e(4096, [17, 32], B2, B2, Numerics::Same),
        e(17408, [2, 16], TC8, TC8, Numerics::Same),
        e(17408, [17, 32], TC8, B2, Numerics::Differs),
    ]);
    // `SIB` is bit-identical to batch2 at n=4096 rows 5..16, not to tc8: tc8 only holds 2..4
    // there, so the shapes disagree at 5..16 and no range is proven.
    let c = one(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 8], "reference")]),
        &wide,
    );
    assert!(matches!(c.proposal, Proposal::None(_)), "{c:#?}");

    let agree = schedules(vec![
        e(4096, [2, 4], TC8, TC8, Numerics::Same),
        e(4096, [5, 16], TC8, B2, Numerics::BitIdentical),
        e(4096, [17, 32], B2, B2, Numerics::Same),
        e(17408, [2, 16], TC8, TC8, Numerics::Same),
        e(17408, [17, 32], TC8, B2, Numerics::Differs),
    ]);
    let c = one(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 8], "reference")]),
        &agree,
    );
    assert!(c.disagreements.is_empty(), "{c:#?}");
    assert_eq!(
        c.proposal,
        Proposal::Range {
            rows: [2, 16],
            changes: true
        }
    );
    let c = one(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 16], "reference")]),
        &agree,
    );
    assert_eq!(
        c.proposal,
        Proposal::Range {
            rows: [2, 16],
            changes: false
        }
    );

    // A gap in one shape's coverage splits the agreed rows: no single range.
    let gap = schedules(vec![
        e(4096, [2, 4], TC8, TC8, Numerics::Same),
        e(4096, [9, 16], TC8, TC8, Numerics::Same),
        e(17408, [2, 16], TC8, TC8, Numerics::Same),
    ]);
    let c = one(
        &fusions(&[rule("tc8", "w4a16_gemv_tc8", [2, 8], "reference")]),
        &gap,
    );
    assert!(
        matches!(c.proposal, Proposal::None(ref w) if w.contains("not one range")),
        "{c:#?}"
    );
}

#[test]
fn opt_in_rules_and_rules_no_schedule_names_are_not_checked() {
    let s = schedules(vec![e(4096, [2, 8], TC8, TC8, Numerics::Same)]);
    let r = check_fusions(
        &fusions(&[
            rule("lever", "w4a16_gemv_tc8", [2, 8], "differs"),
            rule("unswept", "w4a16_gemv_other", [2, 8], "reference"),
        ]),
        &s,
    )
    .unwrap();
    assert!(r.rules.is_empty(), "{r:#?}");
    assert!(check_fusions("schema = 2\n", &s).is_err());
}
