// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan fidelity: the plan's launch sequence (collectives at their `at`, repetitions,
//! placeholders), the multiset and order scores with their residuals, and every oracle refusal.

use super::*;

use crate::rules::{CommApi, CommLaunch, KernelId, Numerics, Repeat};

fn k(func: &str) -> KernelId {
    KernelId {
        module: "m".into(),
        func: func.into(),
    }
}

fn group(rule: &str, kernels: &[&str], repeat: Repeat, comm: &[(&str, u64)], emitter: &str) -> Group {
    Group {
        rule: rule.into(),
        kernels: kernels.iter().map(|f| k(f)).collect(),
        repeat,
        copies: None,
        comm: comm
            .iter()
            .map(|&(kernel, at)| CommLaunch {
                api: CommApi::Exchange,
                kernel: kernel.into(),
                at,
            })
            .collect(),
        emitter: emitter.into(),
        numerics: Numerics::Reference,
        nodes: Vec::new(),
        runs: Vec::new(),
    }
}

fn plan(rows: u64, groups: Vec<Group>) -> FusionPlan {
    FusionPlan {
        arch: "toy".into(),
        mode: Mode::Verify,
        rows,
        table: None,
        groups,
        edge_states: Vec::new(),
        edge_formats: Vec::new(),
        digest: String::new(),
    }
}

fn names(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|x| x.to_string()).collect()
}

/// 2026-10-23: The screened head launches its exchange after its two scan kernels, and the
/// collective rule's sum follows the exchange; a per-row group repeats whole.
#[test]
fn the_sequence_places_collectives_and_repeats_groups() {
    let p = plan(
        2,
        vec![
            group("ar", &["rank_sum"], Repeat::Once, &[("SendRecv", 0)], "e"),
            group("row", &["a", "b"], Repeat::PerRow, &[], "e"),
            group("head", &["scan", "rescore", "resolve"], Repeat::Once, &[("SendRecv", 2)], "e"),
        ],
    );
    let (seq, novel) = plan_sequence(&p);
    assert_eq!(
        seq,
        names(&["SendRecv", "rank_sum", "a", "b", "a", "b", "scan", "rescore", "SendRecv", "resolve"])
    );
    assert!(novel.is_empty());
    assert_eq!(seq.len() as u64, p.launches(), "the sequence is what the plan counts");
}

#[test]
fn an_exact_trace_scores_one_and_leaves_no_residual() {
    let p = plan(1, vec![group("head", &["scan", "rescore", "resolve"], Repeat::Once, &[("SendRecv", 2)], "e")]);
    let f = compare(&p, &names(&["scan", "rescore", "SendRecv", "resolve"]));
    assert_eq!((f.planned, f.traced, f.matched, f.in_order), (4, 4, 4, 4));
    assert_eq!(f.share(), 1.0);
    assert_eq!(f.order_share(), 1.0);
    assert!(f.missing.is_empty() && f.extra.is_empty() && f.novel.is_empty());
}

/// 2026-10-23: The same multiset in another order keeps the multiset score and loses order: the
/// exchange placed first (`at = 0`) where the serve launches it third.
#[test]
fn a_misplaced_collective_costs_order_not_membership() {
    let p = plan(1, vec![group("head", &["scan", "rescore", "resolve"], Repeat::Once, &[("SendRecv", 0)], "e")]);
    let f = compare(&p, &names(&["scan", "rescore", "SendRecv", "resolve"]));
    assert_eq!(f.share(), 1.0);
    assert_eq!(f.in_order, 3);
    assert!(f.order_share() < 1.0);
}

/// 2026-10-23: A planned kernel the serve does not launch and a traced one the plan omits are
/// each named with their count, and the score is matched over the multiset union.
#[test]
fn residuals_name_each_kernel_and_its_count() {
    let p = plan(2, vec![group("row", &["a", "x"], Repeat::PerRow, &[], "e")]);
    let f = compare(&p, &names(&["a", "a", "y", "y", "y"]));
    assert_eq!(f.matched, 2);
    assert_eq!(f.extra, [("x".to_string(), 2)].into_iter().collect());
    assert_eq!(f.missing, [("y".to_string(), 3)].into_iter().collect());
    assert_eq!(f.share(), 2.0 / 7.0);
}

/// 2026-10-23: A placeholder launches nothing: what the serve ran at that site is missing and
/// the placeholder is listed.
#[test]
fn a_placeholder_is_listed_and_launches_nothing() {
    let p = plan(
        1,
        vec![
            group("ok", &["a"], Repeat::Once, &[], "e"),
            group("novel.attend", &["ghost"], Repeat::Once, &[], crate::hardware::plan::NOVEL_EMITTER),
        ],
    );
    let f = compare(&p, &names(&["a", "attend_kernel"]));
    assert_eq!(f.novel, names(&["novel.attend"]));
    assert_eq!(f.planned, 1);
    assert_eq!(f.missing, [("attend_kernel".to_string(), 1)].into_iter().collect());
    assert!(f.extra.is_empty(), "the placeholder's kernels are not planned launches");
    assert!(f.share() < 1.0);
}

#[test]
fn lcs_is_the_longest_common_subsequence() {
    assert_eq!(lcs(&names(&["a", "b", "c", "d"]), &names(&["b", "d", "a", "c"])), 2);
    assert_eq!(lcs(&names(&["a", "b"]), &names(&[])), 0);
    assert_eq!(lcs(&names(&["a", "b", "a"]), &names(&["a", "a"])), 2);
}

const GOOD: &str = r#"
schema = 1
source = "t"
sqlite_sha256 = "00"
rank = 0
step_kernel = "scan"
steps = 3
sequence = "ABA"
[[variant]]
name = "A"
steps = 2
[[variant.segment]]
name = "verify"
mode = "verify"
rows = 2
kernels = ["a"]
[[variant.segment]]
name = "commit"
kernels = []
[[variant]]
name = "B"
steps = 1
[[variant.segment]]
name = "verify"
mode = "verify"
rows = 2
kernels = ["a"]
"#;

#[test]
fn a_consistent_oracle_parses() {
    let t = parse_trace(GOOD).unwrap();
    assert_eq!(t.variant.len(), 2);
    assert_eq!(t.variant[0].segment[0].rows, Some(2));
}

#[test]
fn an_inconsistent_oracle_is_refused() {
    for (from, to, why) in [
        ("schema = 1", "schema = 2", "schema"),
        ("steps = 3", "steps = 4", "variant steps"),
        ("sequence = \"ABA\"", "sequence = \"ABC\"", "unstated variant"),
        ("sequence = \"ABA\"", "sequence = \"AB\"", "sequence length"),
        ("mode = \"verify\"\nrows = 2\nkernels = [\"a\"]\n[[variant.segment]]\nname = \"commit\"",
         "mode = \"verify\"\nkernels = [\"a\"]\n[[variant.segment]]\nname = \"commit\"", "mode without rows"),
        ("name = \"verify\"\nmode = \"verify\"", "name = \"verify\"\nmode = \"verfy\"", "unknown mode"),
        ("source = \"t\"", "source = \"t\"\nextra = 1", "unknown field"),
    ] {
        let text = GOOD.replacen(from, to, 1);
        assert_ne!(text, GOOD, "{why}: the mutation applied");
        assert!(parse_trace(&text).is_err(), "{why}");
    }
}
