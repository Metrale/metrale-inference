// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: A box plan exists only at a util its class allows and with its terms inside the
//! budget; each term reads back as given, and the driver's is derived from the budget.
//!
//! Owner: metrale-circuit (memory).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-18: Driver terms with a 0.85 ceiling and a fixed share only, so the driver term is
/// a known constant (the per-mille share is exercised by `budget_tests`).
const DRIVER: DriverTerms = DriverTerms {
    driver_fixed_bytes: 1000,
    driver_budget_per_mille: 0,
    unified: true,
    util_ceiling: 0.85,
};

fn inputs() -> BoxInputs {
    BoxInputs {
        device_bytes: 100_000,
        util: Util::new(0.85, &DRIVER).unwrap(),
        driver: DRIVER,
        weights: 50_000,
        drafter: 2_000,
        comm: 300,
        workspace: 4_000,
        staging: 70,
    }
}

#[test]
fn a_util_outside_zero_to_the_ceiling_is_refused() {
    assert_eq!(Util::new(0.85, &DRIVER).map(Util::get), Ok(0.85));
    for bad in [0.86, 0.0, -0.1, f64::NAN] {
        assert!(
            matches!(Util::new(bad, &DRIVER), Err(BoxPlanError::Util { ceiling, .. }) if ceiling == 0.85),
            "{bad}"
        );
    }
}

#[test]
fn every_term_reads_back_and_the_room_is_the_rest_of_the_budget() {
    let p = BoxPlan::new(&inputs()).unwrap();
    assert_eq!(p.budget_bytes, 85_000);
    for (term, want) in [
        (BoxTerm::Weights, 50_000),
        (BoxTerm::Drafter, 2_000),
        (BoxTerm::Driver, 1_000),
        (BoxTerm::Comm, 300),
        (BoxTerm::Workspace, 4_000),
        (BoxTerm::Staging, 70),
    ] {
        assert_eq!(p.term(term), want, "{term:?}");
    }
    assert_eq!(p.kv_room, 85_000 - 57_370);
    assert_eq!(p.sequences(1_000), Some(27));
    assert_eq!(p.sequences(0), None);
}

#[test]
fn terms_over_the_budget_or_past_u64_are_refused() {
    let exact = BoxInputs {
        weights: 85_000 - 7_370,
        ..inputs()
    };
    assert_eq!(BoxPlan::new(&exact).unwrap().kv_room, 0);
    let over = BoxInputs {
        weights: 85_000 - 7_369,
        ..inputs()
    };
    assert_eq!(
        BoxPlan::new(&over),
        Err(BoxPlanError::Over {
            resident: 85_001,
            budget: 85_000
        })
    );
    let wrap = BoxInputs {
        weights: u64::MAX,
        ..inputs()
    };
    assert!(matches!(
        BoxPlan::new(&wrap),
        Err(BoxPlanError::Over { .. })
    ));
}
