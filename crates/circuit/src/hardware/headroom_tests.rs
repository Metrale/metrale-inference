// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Headroom on the toy model (`test_fixture`): floors and bound types, Amdahl upside
//! on hand-computed numbers, the step budget closing exactly, refusal of a capture that does not
//! match the plan, unmeasured sites never read as measured, and blank energy without energy.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every expected number is computed here from the toy's dims (hidden 64, inter 128,
//! vocab 256) and the `fp4dev` datasheet (1000 GB/s, BF16 100 TFLOPS), not read back from the
//! code under test.

use std::collections::BTreeMap;

use super::headroom::{Bound, Headroom, HeadroomError, Inputs, Measured, Outside, saved_us};
use super::headroom_trace::{Stated, Trace, TraceKernel, TracePlan, parse_trace};
use super::hardware_tests::W4A16;
use super::test_fixture as fx;
use crate::rules::Mode;
use crate::venn::Run;

/// 2026-10-22: lm_head at one row: BF16 weight 256x64, BF16 input row 64, BF16 output row 256.
const LM_HEAD_BYTES: f64 = (256.0 * 64.0 + 64.0 + 256.0) * 2.0;
/// 2026-10-22: `fp4dev`'s bandwidth, bytes per microsecond.
const BW: f64 = 1000.0 * 1e3;

fn stated(v: f64) -> Stated {
    Stated {
        value: v,
        basis: "test".into(),
    }
}

fn kernel(name: &str, site: Option<&str>, calls: u64, total: f64, excl: f64) -> TraceKernel {
    TraceKernel {
        kernel: name.into(),
        site: site.map(Into::into),
        label: None,
        note: None,
        calls,
        total_us: total,
        exclusive_us: excl,
        energy_uj: None,
    }
}

/// 2026-10-22: Two steps of 100 us: lm_head 10 us a step, all exclusive, and an unknown kernel.
fn trace(kernels: Vec<TraceKernel>) -> Trace {
    Trace {
        schema: 1,
        source: "test capture".into(),
        step_kernel: "lm_head".into(),
        steps: 2,
        window_us: 200.0,
        tokens_per_step: None,
        launch_floor_us: None,
        plan: TracePlan::default(),
        kernels,
    }
}

fn inputs(t: Option<&Trace>, launch: Option<f64>) -> Inputs<'_> {
    Inputs {
        model: "toy".into(),
        kv_tokens: None,
        tokens_per_step: stated(1.0),
        launch_floor_us: launch.map(stated),
        dims: (BTreeMap::new(), None),
        target_pct: 80.0,
        trace: t,
        energy: None,
    }
}

fn run(device: &str, mode: Mode, rows: u64, i: &Inputs<'_>) -> Result<Headroom, HeadroomError> {
    super::build_headroom(
        &fx::registry(),
        device,
        &fx::tree(),
        &fx::model(W4A16),
        &[Run { mode, rows }],
        i,
    )
}

fn site<'h>(h: &'h Headroom, label: &str) -> &'h super::headroom::SiteRow {
    h.sites
        .iter()
        .find(|r| r.site == label)
        .unwrap_or_else(|| panic!("no site `{label}` in {:?}", h.sites.iter().map(|r| &r.site).collect::<Vec<_>>()))
}

#[test]
fn saved_time_is_capped_by_the_floor_and_by_exclusive_time() {
    let m = |total, excl| Measured {
        calls_per_step: 1.0,
        total_us: total,
        exclusive_us: excl,
        kernels: vec![],
        drift: vec![],
    };
    // Falls 10 -> 5: saves 5, but only 4 of its 10 us were on the critical path alone.
    assert_eq!(saved_us(&m(10.0, 4.0), 5.0, 1.0), 4.0);
    assert_eq!(saved_us(&m(10.0, 10.0), 5.0, 1.0), 5.0);
    // At 50% of a 5 us floor the target is 10 us: nothing saved.
    assert_eq!(saved_us(&m(10.0, 10.0), 5.0, 0.5), 0.0);
    // At 80%: 10 - 5 / 0.8 = 3.75.
    assert_eq!(saved_us(&m(10.0, 10.0), 5.0, 0.8), 3.75);
    // Faster than the floor (another lowering): never a negative saving.
    assert_eq!(saved_us(&m(10.0, 10.0), 12.0, 1.0), 0.0);
}

#[test]
fn the_floor_is_bytes_over_bandwidth_and_the_upside_is_amdahl() {
    let t = trace(vec![kernel("lm_head", None, 2, 20.0, 20.0)]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    assert_eq!(h.ceilings.dram_gbps, 1000.0, "the fixture's datasheet");
    let r = site(&h, "head [lm_head]");
    assert_eq!(r.bytes, LM_HEAD_BYTES);
    assert_eq!(r.flops, 2.0 * 64.0 * 256.0);
    let floor = LM_HEAD_BYTES / BW;
    assert!((r.floor_us - floor).abs() < 1e-12, "{} vs {floor}", r.floor_us);
    assert_eq!(r.bound, Bound::Memory);
    let m = r.measured.as_ref().expect("captured");
    assert_eq!((m.total_us, m.exclusive_us, m.calls_per_step), (10.0, 10.0, 1.0));
    assert!((r.pct_of_floor.unwrap() - 100.0 * floor / 10.0).abs() < 1e-9);
    assert!((r.headroom_us.unwrap() - (10.0 - floor)).abs() < 1e-12);
    // Alone at its floor the 100 us step loses 10 - floor; at 80% of it, 10 - floor / 0.8.
    let full = r.upside_full.unwrap();
    assert!((full.step_us - (90.0 + floor)).abs() < 1e-9);
    assert!((full.tok_s - 1e6 / (90.0 + floor)).abs() < 1e-6);
    let target = r.upside_target.unwrap();
    assert!((target.step_us - (90.0 + floor / 0.8)).abs() < 1e-9);
    assert_eq!(h.curve.len(), 1);
    assert!((h.curve[0].full.saved_us - (10.0 - floor)).abs() < 1e-12);
}

#[test]
fn a_fused_launch_does_not_pay_for_the_edges_it_keeps_on_chip() {
    // A rule that runs final_norm and lm_head in one launch keeps `xn` (64 BF16 values) on chip:
    // the fused floor is the two nodes' bytes less xn's write and read.
    let mut tree = fx::tree();
    let rules = tree.files.get_mut("kernels/base/common/FUSIONS.toml").unwrap();
    rules.push_str(
        r#"
[[rule]]
id = "head_fused"
pattern = [{ op = "final_norm" }, { op = "lm_head" }]
kernels = [{ module = "m", func = "lm_head" }]
repeat = "once"
emitter = "head_fused"
rows = [1, 128]
modes = ["decode", "multi_seq", "verify"]
numerics = "reference"
priority = 90
cite = "test"
"#,
    );
    // The fused kernel's family declares both ops (crate::pipeline refuses it otherwise).
    let fams = tree.files.get_mut("kernels/base/common/KERNEL_FAMILIES.toml").unwrap();
    let head = r#"op = [{ op = "lm_head" }]"#;
    assert_eq!(fams.matches(head).count(), 1);
    *fams = fams.replace(
        head,
        &format!(
            "{}\nop = [{{ op = \"lm_head\" }}, {{ op = \"final_norm\" }}]",
            crate::test_toy::pipelines(&["final_norm"])
        ),
    );
    let h = super::build_headroom(
        &fx::registry(),
        "fp4dev",
        &tree,
        &fx::model(W4A16),
        &[Run { mode: Mode::Decode, rows: 1 }],
        &inputs(None, None),
    )
    .unwrap();
    let r = site(&h, "head [final_norm,lm_head]");
    assert!(r.fused);
    // final_norm reads its 64-value row and writes `xn` (the roofline counts a norm's edges, not
    // its weight); lm_head reads `xn` again. Fused, `xn`'s write and read are gone: what is left
    // is final_norm's input row plus lm_head without its input row.
    let unfused = LM_HEAD_BYTES + (64.0 + 64.0) * 2.0;
    assert_eq!(r.bytes, unfused - 2.0 * 64.0 * 2.0);
    assert_eq!(r.bytes, (256.0 * 64.0 + 64.0 + 256.0) * 2.0);
    assert!((r.node_sum_us - unfused / BW).abs() < 1e-12, "{}", r.node_sum_us);
    assert!((r.floor_us - r.bytes / BW).abs() < 1e-12);
}

#[test]
fn bound_is_memory_compute_or_launch_by_the_larger_term() {
    let mem = run("fp4dev", Mode::Decode, 1, &inputs(None, None)).unwrap();
    assert_eq!(site(&mem, "head [lm_head]").bound, Bound::Memory);
    // `slowmath`: 1e6 GB/s, 1e-6 TFLOPS: the head's 32768 FLOPs take 32768 / (1e-6 * 1e6) us.
    let cmp = run("slowmath", Mode::Decode, 1, &inputs(None, None)).unwrap();
    let r = site(&cmp, "head [lm_head]");
    assert_eq!(r.bound, Bound::Compute);
    assert!((r.floor_us - 32768.0).abs() < 1e-6, "{}", r.floor_us);
    // A 1 us launch floor is above the head's 0.033 us roofline floor.
    let l = run("fp4dev", Mode::Decode, 1, &inputs(None, Some(1.0))).unwrap();
    let r = site(&l, "head [lm_head]");
    assert_eq!((r.bound, r.floor_us), (Bound::Launch, 1.0));
    // ...but not above a compute floor of 32768 us.
    let lc = run("slowmath", Mode::Decode, 1, &inputs(None, Some(1.0))).unwrap();
    assert_eq!(site(&lc, "head [lm_head]").bound, Bound::Compute);
}

#[test]
fn the_step_budget_closes_exactly_with_the_residual() {
    let t = trace(vec![
        kernel("lm_head", None, 2, 20.0, 20.0),
        kernel("norm", None, 4, 8.0, 6.0),
        kernel("mystery", None, 2, 30.0, 24.0),
    ]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    let tt = &h.totals;
    assert_eq!(tt.step_us, Some(100.0));
    assert_eq!(tt.sites_exclusive_us, Some(10.0 + 3.0));
    assert_eq!(tt.outside_exclusive_us, Some(12.0));
    assert_eq!(tt.residual_us, Some(100.0 - 13.0 - 12.0));
    let sum = tt.sites_exclusive_us.unwrap() + tt.outside_exclusive_us.unwrap() + tt.residual_us.unwrap();
    assert_eq!(sum, tt.step_us.unwrap(), "sites + outside + residual = measured step");
    assert_eq!(tt.overlap_us, Some((10.0 + 4.0 + 15.0) - (10.0 + 3.0 + 12.0)));
    assert_eq!(h.outside.len(), 1);
    assert_eq!((h.outside[0].kernel.as_str(), &h.outside[0].why), ("mystery", &Outside::Unknown));
}

#[test]
fn a_capture_that_does_not_match_the_plan_is_refused_with_every_mismatch() {
    let t = trace(vec![
        // lm_head's family implements lm_head, not the gate_up linear.
        kernel("lm_head", Some("ffn.up"), 2, 20.0, 20.0),
        kernel("norm", Some("nowhere.x"), 2, 2.0, 2.0),
    ]);
    match run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)) {
        Err(HeadroomError::Mismatch(m)) => {
            assert_eq!(m.len(), 2, "{m:?}");
            assert!(m[0].contains("`lm_head` is attributed to site `ffn.up`"), "{}", m[0]);
            assert!(m[1].contains("`nowhere.x`, which no launch site"), "{}", m[1]);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    // The same kernel attributed to its own site is accepted.
    let ok = trace(vec![kernel("lm_head", Some("head.lm_head"), 2, 20.0, 20.0)]);
    assert!(run("fp4dev", Mode::Decode, 1, &inputs(Some(&ok), None)).is_ok());
}

#[test]
fn a_kernel_of_the_planned_family_is_measured_with_a_drift_note() {
    // `final_norm` is in family f_norm with `norm`: attributed to ffn.norm it is that site's.
    let t = trace(vec![kernel("final_norm", Some("ffn.norm"), 4, 8.0, 8.0)]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    let m = site(&h, "ffn [norm]").measured.as_ref().unwrap();
    assert_eq!(m.drift, vec!["runs `final_norm`, the plan names `m::norm` (family `f_norm`)".to_string()]);
}

#[test]
fn unmeasured_sites_are_never_shown_as_measured() {
    let none = run("fp4dev", Mode::Decode, 1, &inputs(None, None)).unwrap();
    for r in &none.sites {
        assert!(r.measured.is_none() && r.pct_of_floor.is_none() && r.headroom_us.is_none());
        assert!(r.upside_full.is_none() && r.upside_target.is_none());
    }
    assert_eq!((none.totals.step_us, none.totals.residual_us), (None, None));
    assert!(none.curve.is_empty());
    let md = super::render_headroom(&none, "test");
    assert!(md.contains("unmeasured (projection"), "{md}");
    // With a capture of lm_head only, every other site stays unmeasured and only lm_head's floor
    // counts as a measured site's.
    let t = trace(vec![kernel("lm_head", None, 2, 20.0, 20.0)]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    let measured: Vec<&str> = h.sites.iter().filter(|r| r.measured.is_some()).map(|r| r.site.as_str()).collect();
    assert_eq!(measured, vec!["head [lm_head]"]);
    let lm = site(&h, "head [lm_head]").floor_us;
    assert!((h.totals.measured_floor_us.unwrap() - lm).abs() < 1e-12);
    let others: f64 = h.sites.iter().filter(|r| r.measured.is_none()).map(|r| r.floor_us).sum();
    assert!((h.totals.unmeasured_projection_us - others).abs() < 1e-12);
    assert!((h.totals.ceiling_us - (lm + others)).abs() < 1e-12);
}

#[test]
fn energy_columns_are_blank_without_energy_and_filled_from_captured_energy() {
    let t = trace(vec![kernel("lm_head", None, 2, 20.0, 20.0)]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    for r in &h.sites {
        assert_eq!(r.energy, Default::default(), "{}", r.site);
    }
    let md = super::render_headroom(&h, "test");
    let row = md.lines().find(|l| l.starts_with("| 1 | decode n=1 | head [lm_head]")).expect("top row");
    assert!(row.ends_with("| – | – | – |"), "{row}");
    // 500 uJ over 2 steps of one launch: 250 uJ per launch; no energy roofline, so no floor.
    let mut k = kernel("lm_head", None, 2, 20.0, 20.0);
    k.energy_uj = Some(500.0);
    let t = trace(vec![k]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    let e = site(&h, "head [lm_head]").energy;
    assert_eq!((e.uj_per_launch, e.floor_uj, e.pct_of_floor), (Some(250.0), None, None));
}

#[test]
fn a_trace_is_parsed_strictly_and_its_window_checked() {
    let good = r#"
schema = 1
source = "nsys x.sqlite @ abc, box, date"
step_kernel = "lm_head"
steps = 2
window_us = 200.0
[[kernel]]
kernel = "lm_head"
calls = 2
total_us = 20.0
exclusive_us = 20.0
"#;
    assert_eq!(parse_trace(good).unwrap().step_us(), 100.0);
    let bad = [
        (good.replace("exclusive_us = 20.0", "exclusive_us = 30.0"), "must lie in"),
        (good.replace("window_us = 200.0", "window_us = 10.0"), "exceeds its window"),
        (good.replace("steps = 2", "steps = 0"), "no whole step"),
        (good.replace("source = \"nsys x.sqlite @ abc, box, date\"", "source = \" \""), "no `source`"),
        (good.replace("calls = 2", "calls = 2\nguess = 1"), "unknown field"),
    ];
    for (text, why) in bad {
        let e = parse_trace(&text).unwrap_err();
        assert!(e.contains(why), "{why}: {e}");
    }
}
