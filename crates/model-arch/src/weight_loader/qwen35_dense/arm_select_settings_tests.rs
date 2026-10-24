// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The qwen35_dense loader levers read from the settings against the environment
//! readers they replaced, frozen here: the arm predicates in `arm_select.rs`, the load-site
//! reads in `qwen35_dense.rs`, `gdn_layer.rs`, `attn_layer.rs` and `gdn_dequant.rs`, and the
//! two `Fp8RouteInputs` fields the preflight prices with.
//!
//! Owner: model-arch (qwen35_dense).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use super::dense_fp8_enabled;
use crate::weight_loader::qwen35_dense::predicted_residency::Fp8RouteInputs;
use metrale_config::ModelConfig;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};
use metrale_settings::InferenceServerSettings;

fn route(s: &InferenceServerSettings) -> Fp8RouteInputs {
    Fp8RouteInputs::from_settings(&ModelConfig::qwen3_next_80b_nvfp4(), s, true)
}

/// 2026-10-17: The `== "1"` readers of `METRALE_DENSE_FP8` (`dense_fp8_enabled` and the
/// `Fp8RouteInputs` copy) and `METRALE_FP8_ROWWISE` (`rowwise_fp8_enabled`) become switches.
#[test]
fn exact_one_readers_become_switches() {
    let switch_changes = |env| {
        [
            (env, Some("true"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected),
        ]
    };
    let old = |p: Option<&str>| p == Some("1");
    let dense = "METRALE_DENSE_FP8";
    assert_eq!(
        check(dense, Old::OptIn, old, dense_fp8_enabled),
        switch_changes(dense)
    );
    assert_eq!(
        check(dense, Old::OptIn, old, |s| route(s).dense_fp8),
        switch_changes(dense)
    );
    let rowwise = "METRALE_FP8_ROWWISE";
    assert_eq!(
        check(rowwise, Old::OptIn, old, |s| s.kernels.fp8_rowwise),
        switch_changes(rowwise)
    );
    for env in [dense, rowwise] {
        assert!(
            settings_from(&[(env, "")]).is_err(),
            "{env}=\"\" was off, now refused"
        );
    }
}

/// 2026-10-17: The `var_os(..).is_none()` readers stay presence levers: any value, the empty
/// one included, turns the arm off, as before.
#[test]
fn presence_readers_read_as_before() {
    let off_when_set = |p: Option<&str>| p.is_none();
    let rows: [(&'static str, fn(&InferenceServerSettings) -> bool); 5] = [
        ("METRALE_NO_Q2_GDN", |s| !s.kernels.no_q2_gdn),
        ("METRALE_NO_GDN_FP8", |s| !s.kernels.no_gdn_fp8),
        ("METRALE_NO_GDN_FP8_PREFILL", |s| {
            !s.scheduler.no_gdn_fp8_prefill
        }),
        ("METRALE_NO_Q2_ATTN", |s| !s.kernels.no_q2_attn),
        ("METRALE_NO_GDN_FP8", |s| route(s).gdn_fp8),
    ];
    for (env, new) in rows {
        assert_eq!(check(env, Old::Presence, off_when_set, new), []);
        let empty = settings_from(&[(env, "")]).expect("a presence lever accepts any value");
        assert!(!new(&empty), "{env}=\"\" still turns the arm off");
    }
}
