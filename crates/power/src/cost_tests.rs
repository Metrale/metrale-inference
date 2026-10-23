// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Cost per Mtok: the design's worked example by hand, every input required, the
//! delta-method and bootstrap intervals agreeing on a well-measured input and disagreeing on a
//! noisy reciprocal.

use metrale_error::ErrorCode;
use metrale_stats::interval::Summary;

use crate::cost::{CostInputs, cost, per_token};

const INPUTS: CostInputs = CostInputs {
    electricity_usd_per_kwh: Some(0.30),
    hardware_usd: Some(4000.0),
    lifetime_hours: Some(3.0 * 8760.0),
    utilization: Some(1.0),
};

fn s(mean: f64, se: f64) -> Summary {
    Summary { mean, se, df: 30.0 }
}

#[test]
fn the_design_example_by_hand() {
    // 2026-10-22: 0.44 J/tok at $0.30/kWh = $0.0367/Mtok; $4000 over 3 y at 100 tok/s = $0.423/Mtok.
    let c = cost(s(0.44, 0.004), s(100.0, 0.5), &INPUTS, 0.95, 4000, 1).unwrap();
    assert!((c.electricity.usd_per_mtok - 0.44 * 0.30 / 3.6).abs() < 1e-12);
    let hw = 4000.0 / (3.0 * 8760.0 * 3600.0) / 100.0 * 1e6;
    assert!(
        (c.hardware.usd_per_mtok - hw).abs() < 1e-12,
        "{}",
        c.hardware.usd_per_mtok
    );
    assert!((c.total_usd_per_mtok - (0.44 * 0.30 / 3.6 + hw)).abs() < 1e-12);
    assert!(
        c.electricity.methods_agree && c.hardware.methods_agree,
        "{c:?}"
    );
    assert!(c.electricity.delta.contains(c.electricity.usd_per_mtok));
}

#[test]
fn every_input_is_required() {
    for drop in 0..4 {
        let mut i = INPUTS;
        match drop {
            0 => i.electricity_usd_per_kwh = None,
            1 => i.hardware_usd = None,
            2 => i.lifetime_hours = None,
            _ => i.utilization = None,
        }
        let e = cost(s(0.44, 0.0), s(100.0, 0.0), &i, 0.95, 100, 1).unwrap_err();
        assert_eq!(e.code(), ErrorCode::CostInputMissing, "{drop}");
    }
    let bad = CostInputs {
        utilization: Some(1.5),
        ..INPUTS
    };
    assert_eq!(
        cost(s(0.44, 0.0), s(100.0, 0.0), &bad, 0.95, 100, 1)
            .unwrap_err()
            .code(),
        ErrorCode::CostInputInvalid
    );
}

#[test]
fn a_noisy_reciprocal_makes_the_two_methods_disagree() {
    // 2026-10-22: tok/s 10 ± 4: 1/x is strongly skewed, the delta method's symmetric interval
    // misses it, and the result says so.
    let c = cost(s(0.44, 0.004), s(10.0, 4.0), &INPUTS, 0.95, 4000, 1).unwrap();
    assert!(!c.hardware.methods_agree, "{c:?}");
}

#[test]
fn per_token_divides_by_rows() {
    let t = per_token(s(0.08, 0.002), 16);
    assert!((t.mean - 0.005).abs() < 1e-15 && (t.se - 0.000125).abs() < 1e-15);
}
