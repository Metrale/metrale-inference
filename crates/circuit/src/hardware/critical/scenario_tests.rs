// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Rewrites on the diamond fixture, and the refusals of rewrites and backtests that
//! cannot mean anything.

use alloc::collections::BTreeMap;

use metrale_error::ErrorCode;

use super::analyze::{RewriteResult, WhatIf};
use super::backtest::{Measured, Set, Spec, Verdict, parse_spec};
use super::capture::Capture;
use super::dag::build;
use super::fixture::diamond;
use super::maxplus::{Latency, fit_latency, longest, step_us, work_in_context};
use super::scenario::{Rewrite, Scenario, Sel};
use super::sim::{DramCalibration, Resources, SmLimits};

fn scenario(c: &Capture) -> (Scenario, Latency) {
    let g = build(c, 1_000, 3_000).unwrap();
    let lat = fit_latency(&g);
    let work = work_in_context(&g);
    let site = g
        .nodes
        .iter()
        .map(|n| n.group.and_then(|k| c.groups[k as usize].1.clone()))
        .collect();
    let name = g
        .nodes
        .iter()
        .map(|n| {
            n.group
                .map_or_else(String::new, |k| c.groups[k as usize].0.clone())
        })
        .collect();
    let n = g.nodes.len();
    (
        Scenario {
            g,
            work0: work.clone(),
            solo: work.clone(),
            work,
            bytes: vec![0.0; n],
            cap: Vec::new(),
            over: BTreeMap::new(),
            site,
            name,
        },
        lat,
    )
}

fn res() -> Resources {
    Resources {
        sms: 48.0,
        sm: SmLimits {
            regs: 65_536,
            threads: 1_536,
            smem: 102_400,
            ctas: 32,
        },
        dram_bytes_per_us: 273_000.0,
        dram_peak_bytes_per_us: 273_000.0,
    }
}

fn shrink(site: Option<&str>, kernel: Option<&str>, factor: f64) -> Rewrite {
    Rewrite::Shrink {
        select: Sel {
            site: site.map(Into::into),
            kernel: kernel.map(Into::into),
        },
        factor,
        basis: "test".into(),
    }
}

fn step(sc: &Scenario, lat: &Latency) -> f64 {
    step_us(&sc.g, &longest(&sc.g, &sc.weights(lat)))
}

#[test]
fn shrinking_moves_the_step_by_the_subgradient() {
    let c = diamond(4, 11_000);
    let (base, lat) = scenario(&c);
    assert!((step(&base, &lat) - 23.0).abs() < 1e-9);
    let mut sc = base.clone();
    assert_eq!(
        sc.apply(&shrink(Some("blk.b1"), None, 0.5), &lat, &res())
            .unwrap(),
        4,
        "one b1 per window step"
    );
    assert!(
        (step(&sc, &lat) - 14.0).abs() < 1e-9,
        "b1 18 -> 9 on the chain: {}",
        step(&sc, &lat)
    );
    let mut sc = base.clone();
    sc.apply(&shrink(None, Some("a1"), 0.0), &lat, &res())
        .unwrap();
    assert!((step(&sc, &lat) - 23.0).abs() < 1e-9, "a1 is off the chain");
}

#[test]
fn a_rewrite_that_selects_nothing_or_everything_is_refused() {
    let c = diamond(2, 11_000);
    let (sc, lat) = scenario(&c);
    for (rw, why) in [
        (shrink(Some("blk.nope"), None, 0.5), "matches nothing"),
        (shrink(None, None, 0.5), "names neither"),
        (shrink(Some("blk.b1"), Some("a1"), 0.5), "matches nothing"),
        (shrink(Some("blk.b1"), None, -1.0), "must be >= 0"),
        (
            Rewrite::Fuse {
                chain: vec![
                    Sel {
                        site: None,
                        kernel: Some("a1".into()),
                    },
                    Sel {
                        site: None,
                        kernel: Some("b1".into()),
                    },
                ],
                work_ratio: 1.0,
                basis: "a1 and b1 are not on one stream".into(),
            },
            "matches nothing",
        ),
    ] {
        let e = sc.clone().apply(&rw, &lat, &res()).unwrap_err();
        assert_eq!(e.code(), ErrorCode::WhatIfInvalid);
        assert!(alloc::format!("{e}").contains(why), "{e}");
    }
}

/// 2026-10-23: An analysis of a `step` µs step whose only rewrite `id` simulates to `delta`.
fn whatif(step: f64, id: &str, delta: f64) -> WhatIf {
    let dram = DramCalibration {
        bytes_per_us: 0.0,
        overlapped_intervals: 0,
        overlapped_us: 0.0,
        solo_max_bytes_per_us: 0.0,
        clipped: 0,
        scaled: 0,
        basis: String::new(),
    };
    let rewrites = vec![RewriteResult {
        id: id.into(),
        places: vec![1],
        maxplus_delta_us: delta,
        sim_delta_us: delta,
        shifts: vec![],
    }];
    WhatIf {
        schema: "whatif/v1".into(),
        source: String::new(),
        steps: 1,
        kv_tokens: 0.0,
        measured_step_us: step,
        replay_step_us: step,
        model_step_us: step,
        sim_step_us: step,
        sim_sm_step_us: step,
        latency: vec![],
        dropped: vec![],
        without: vec![],
        dram,
        critical_by_kind: vec![],
        chain_step: vec![],
        sites: vec![],
        rewrites,
    }
}

#[test]
fn the_sign_rule_follows_the_measured_interval() {
    let v = |p: f64, lo: f64, hi: f64| {
        let spec = Spec {
            schema: "whatif-spec/v1".into(),
            set: vec![Set {
                id: "x".into(),
                depends_on: vec![],
                rewrite: vec![],
                measured: vec![Measured {
                    tag: "t".into(),
                    us: (lo + hi) / 2.0,
                    lo,
                    hi,
                    basis: String::new(),
                }],
            }],
        };
        let w = whatif(1_000.0, "x", p);
        super::backtest::verdicts(&spec, &w, "t").pop().unwrap()
    };
    let ok = |x: &Verdict| (x.sign_ok, x.error_ok);
    assert_eq!(ok(&v(10.0, 5.0, 15.0)), (true, true));
    assert_eq!(
        ok(&v(-10.0, 5.0, 15.0)),
        (false, false),
        "interval excludes zero: the sign must match"
    );
    assert_eq!(
        ok(&v(100.0, 5.0, 15.0)),
        (true, false),
        "right sign, 9% of the step off"
    );
    assert_eq!(
        ok(&v(4.0, -5.0, 15.0)),
        (true, true),
        "interval includes zero: inside it"
    );
    assert!(
        !ok(&v(-30.0, -5.0, 15.0)).0,
        "interval includes zero: outside it"
    );
    assert!(ok(&v(-10.0, -15.0, -5.0)).0);
}

#[test]
fn a_spec_with_unknown_fields_or_schema_is_refused() {
    let good = "schema = \"whatif-spec/v1\"\n[[set]]\nid = \"a\"\ndepends_on = []\nrewrite = [{ kind = \"pdl_off\", prologue = \"zero\", basis = \"b\" }]\nmeasured = []\n";
    assert!(parse_spec(good).is_ok());
    assert_eq!(
        parse_spec(&good.replace("v1", "v0")).unwrap_err().code(),
        ErrorCode::WhatIfInvalid
    );
    assert_eq!(
        parse_spec(&good.replace("basis = \"b\"", "basis = \"b\", factr = 1"))
            .unwrap_err()
            .code(),
        ErrorCode::WhatIfInvalid
    );
}
