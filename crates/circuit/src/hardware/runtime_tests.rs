// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Runtime routes on the in-memory tree: a route that applies is planned beside the
//! primary plan and reported with it, one that does not apply leaves the plan and report as they
//! were, and malformed routes are refused.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use super::class::{ClassRules, class_rules};
use super::hardware_tests::W4A16;
use super::test_fixture::{self as fx, Tree};
use super::{ModelUnderPlan, build_report, plan_one, plan_text, render_report};
use crate::rules::Mode;
use crate::venn::Run;

/// 2026-09-30: The child class runs `act` through `m::act` when `arm = wide`, and a route
/// re-plans multi-sequence steps under `arm = narrow`, where the child's own `act` rule
/// (`m::act_child`) runs instead.
const ROUTE: &str = r#"
[[rule]]
id = "act_wide"
pattern = [{ op = "silu_mul" }]
kernels = [{ module = "m", func = "act" }]
repeat = "once"
emitter = "act_wide"
rows = [2, 128]
modes = ["multi_seq"]
when = { arm = "wide" }
numerics = "reference"
priority = 20
cite = "test"

[[runtime]]
id = "narrow_fallback"
when = { arm = "wide" }
modes = ["multi_seq"]
rows = [2, 128]
plans_as = { arm = "narrow" }
why = "the rows are not contiguous"
cite = "test.rs:1"
"#;

fn setup(arm: &str, route: &str) -> (Tree, ModelUnderPlan) {
    let mut tree = fx::tree();
    tree.files
        .get_mut("kernels/child/common/FUSIONS.toml")
        .expect("child rules")
        .push_str(route);
    let mut model = fx::model(W4A16);
    model.policy.settings.insert("arm".into(), arm.into());
    (tree, model)
}

fn rules(p: &crate::fuser::FusionPlan) -> Vec<&str> {
    p.groups.iter().map(|g| g.rule.as_str()).collect()
}

fn run(mode: Mode, rows: u64) -> Run {
    Run { mode, rows }
}

// 2026-09-30: Mutation: dropping the route loop in `plan::fuse_on` empties `routes`; fusing the
// route under the primary policy (not `plans_as`) leaves `act_wide` in the route's plan, and the
// digest check then drops it.
#[test]
fn a_route_that_applies_is_planned_beside_the_primary_plan() {
    let (tree, model) = setup("wide", ROUTE);
    let one = plan_one(
        &fx::registry(),
        "nofp4",
        &tree,
        &model,
        run(Mode::MultiSeq, 16),
    )
    .unwrap();
    assert!(rules(&one.planned.plan).contains(&"act_wide"));
    let [route] = one.planned.routes.as_slice() else {
        panic!("routes: {:?}", one.planned.routes.len());
    };
    assert_eq!(route.route.id, "narrow_fallback");
    let alt = rules(&route.planned.plan);
    assert!(
        alt.contains(&"act") && !alt.contains(&"act_wide"),
        "{alt:?}"
    );
    assert_ne!(route.planned.plan.digest, one.planned.plan.digest);
    let text = plan_text(&model.circuit, &one);
    assert!(text.contains("# runtime route `narrow_fallback`: when the rows are not contiguous"));
    assert!(text.contains("planned as `arm=narrow`"), "{text}");
}

// 2026-09-30: Mutation: dropping the mode, row or `when` test in `RuntimeRoute::applies` records
// a route on a run the engine never checks.
#[test]
fn a_route_that_does_not_apply_leaves_the_plan_alone() {
    let reg = fx::registry();
    let (tree, model) = setup("wide", ROUTE);
    let decode = plan_one(&reg, "nofp4", &tree, &model, run(Mode::Decode, 1)).unwrap();
    assert!(
        decode.planned.routes.is_empty(),
        "decode is outside the route's modes"
    );
    // 2026-10-02: The plan alone, each group with its compute unit and each node with its
    // pipeline, and no route section.
    let families = &decode.resolved.families;
    let group = |g: &crate::fuser::Group| super::tc_policy::unit_tag(families, &g.kernels);
    let node = |n: crate::ir::NodeIdx| decode.planned.pipelines.line(n);
    let notes = crate::render::Notes {
        group: &group,
        node: &node,
    };
    assert_eq!(
        plan_text(&model.circuit, &decode),
        crate::render::render_noted(&model.circuit, &decode.planned.plan, &decode.header, &notes)
    );
    let short = ROUTE.replace("rows = [2, 128]\nplans_as", "rows = [32, 128]\nplans_as");
    let (tree, model) = setup("wide", &short);
    let n16 = plan_one(&reg, "nofp4", &tree, &model, run(Mode::MultiSeq, 16)).unwrap();
    assert!(
        n16.planned.routes.is_empty(),
        "16 rows is below the route's rows"
    );
    let (tree, model) = setup("narrow", ROUTE);
    let off = plan_one(&reg, "nofp4", &tree, &model, run(Mode::MultiSeq, 16)).unwrap();
    assert!(
        off.planned.routes.is_empty(),
        "the policy is not in the route's `when`"
    );
    // 2026-09-30: The plan above would hide an `applies` that ignores `when`: planned as `narrow`
    // under a `narrow` policy, the route's digest equals the primary's and is dropped.
    let ClassRules::Rules { runtime, .. } = class_rules(&tree, "child").unwrap() else {
        panic!("child inherits rules");
    };
    let [route] = runtime.as_slice() else {
        panic!("one route: {runtime:?}");
    };
    let mut wide = model.policy.clone();
    wide.settings.insert("arm".into(), "wide".into());
    assert!(route.applies(&wide, Mode::MultiSeq, 2));
    assert!(route.applies(&wide, Mode::MultiSeq, 128));
    assert!(!route.applies(&model.policy, Mode::MultiSeq, 16));
    assert!(!route.applies(&wide, Mode::MultiSeq, 1));
    assert!(!route.applies(&wide, Mode::Verify, 2));
    wide.settings.remove("arm");
    assert!(!route.applies(&wide, Mode::MultiSeq, 16));
}

// 2026-09-30: Mutation: not calling `routes_section` drops the section; rendering it
// unconditionally adds it to every report, including the base class's, which has no route.
#[test]
fn the_report_lists_routes_only_where_they_apply() {
    let reg = fx::registry();
    let (tree, model) = setup("wide", ROUTE);
    let report = build_report(&reg, "nofp4", &tree, model.clone(), "cmd".into()).unwrap();
    assert_eq!(report.routes.len(), 2, "multi_seq n=16 and n=128");
    let text = render_report(&report);
    assert!(text.contains("## Runtime routes"), "{text}");
    assert!(text.contains("| `narrow_fallback` |"));
    assert!(
        text.contains("| `act` | `act_wide` |"),
        "runs `act` instead of `act_wide`"
    );
    assert!(text.contains("1 runtime route from"));
    let base = build_report(&reg, "fp4dev", &tree, model, "cmd".into()).unwrap();
    assert!(!render_report(&base).contains("Runtime routes"));
}

// 2026-09-30: Mutation: dropping any of the checks in `parse_routes`, or the id clash check in
// `rules_of`, lets a route through that re-plans nothing or invents a setting.
#[test]
fn malformed_routes_are_refused() {
    let reg = fx::registry();
    for (from, to, says) in [
        (
            "plans_as = { arm = \"narrow\" }",
            "plans_as = { lane = \"x\" }",
            "does not fix",
        ),
        (
            "plans_as = { arm = \"narrow\" }",
            "plans_as = { arm = \"wide\" }",
            "keeps `arm`",
        ),
        (
            "plans_as = { arm = \"narrow\" }",
            "plans_as = {}",
            "`plans_as` is empty",
        ),
        (
            "modes = [\"multi_seq\"]\nrows",
            "modes = [\"prefill\"]\nrows",
            "unknown mode",
        ),
        (
            "rows = [2, 128]\nplans_as",
            "rows = [0, 128]\nplans_as",
            "rows",
        ),
        (
            "why = \"the rows are not contiguous\"",
            "why = \" \"",
            "must be stated",
        ),
        (
            "id = \"narrow_fallback\"",
            "id = \"act_wide\"",
            "shares its id with a rule",
        ),
    ] {
        assert!(ROUTE.contains(from), "{from}");
        let (tree, model) = setup("wide", &ROUTE.replace(from, to));
        let e = plan_one(&reg, "nofp4", &tree, &model, run(Mode::MultiSeq, 16))
            .err()
            .unwrap_or_else(|| panic!("accepted: {to}"))
            .to_string();
        assert!(e.contains(says), "{to}: {e}");
    }
}

// 2026-09-30: `parse_rules` refuses a routed file, so a caller that ignores routes cannot drop
// one unseen; `parse_rule_set` reads it. Mutation: returning the rules of a routed file from
// `parse_rules` passes the first assertion.
#[test]
fn only_the_rule_set_parser_reads_runtime_routes() {
    let text = format!("schema = 1\n{ROUTE}");
    let e = crate::rules::parse_rules(&text).unwrap_err().to_string();
    assert!(
        e.contains("runtime `narrow_fallback`") && e.contains("parse_rule_set"),
        "{e}"
    );
    let set = crate::runtime::parse_rule_set(&text).unwrap();
    assert_eq!(set.rules.len(), 1);
    assert_eq!(set.runtime.len(), 1);
    assert_eq!(set.runtime[0].plans_as_text(), "arm=narrow");
}
