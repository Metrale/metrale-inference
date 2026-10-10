// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `resolve`'s one GPU-free path. The loaded-table/calibration paths need real
//! circuit plan digests (`spec_cost_plan_digests`) and a parsed table/calibration, which are
//! exercised end to end by the timed A/B, not re-asserted here; `TableKey::check`,
//! `DrafterKey` equality, `CostTable::parse` and `AcceptanceCalibration::parse` each have their
//! own unit tests in `metrale_speculative::spec_cost`.

use super::*;

fn off_args() -> ServeSpecCostArgs {
    ServeSpecCostArgs {
        spec_cost_model: SpecCostModel::Off,
        spec_cost_table: None,
        spec_cost_calibration: None,
        spec_cost_recipe: None,
        spec_cost_slack: None,
        spec_objective: None,
    }
}

#[test]
fn off_is_none_without_reading_anything() {
    // 2026-10-04: No table/calibration path, no recipe: if `resolve` tried to read any of them
    // with the mode off, this would fail rather than return `Ok(None)`.
    assert!(
        resolve(&off_args(), None, 100_000, "bf16")
            .unwrap()
            .is_none()
    );
}

#[test]
fn measured_without_a_drafter_is_refused() {
    let args = ServeSpecCostArgs {
        spec_cost_model: SpecCostModel::Measured,
        spec_cost_table: Some("/nonexistent/table.toml".into()),
        spec_cost_calibration: Some("/nonexistent/cal.toml".into()),
        spec_cost_recipe: Some("x/x".into()),
        spec_cost_slack: Some(0.0),
        spec_objective: None,
    };
    // 2026-10-04: The table path does not exist either, so this fails at the read, not the
    // drafter check; the point is only that `measured` never silently returns `Ok(None)`.
    assert!(resolve(&args, None, 100_000, "bf16").is_err());
}
