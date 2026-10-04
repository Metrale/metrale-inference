// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The fit reproduces a skewed load on fresh noise, the bias channel normalises as
//! stated, top-k ordering, and profile refusals.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;

use crate::testkit::SKEWED;

#[test]
fn the_fit_reproduces_a_skewed_load_on_independent_noise() {
    let s = Stream::for_tensor(3, "router", "BF16", &[8, 256]);
    let f = fit(&SKEWED, 2, s).unwrap();
    assert!(f.tv < 0.02, "fit tv {}", f.tv);
    assert_eq!(f.floored, 0);
    let fresh = noise(s.derive(99), 20_000, 8);
    let got = sample_counts(&f.bias, 2, &fresh);
    let tv = total_variation(&SKEWED, &got);
    assert!(tv < 0.03, "fresh-noise tv {tv}");
    let uniform = sample_counts(&[0.0; 8], 2, &fresh);
    assert!(
        total_variation(&SKEWED, &uniform) > 0.3,
        "control: random routers are far from the target"
    );
    assert_eq!(f, fit(&SKEWED, 2, s).unwrap(), "deterministic");
}

#[test]
fn a_never_selected_expert_is_floored_not_unreachable() {
    let s = Stream::for_tensor(4, "router", "BF16", &[8, 256]);
    let f = fit(&[100, 100, 100, 100, 100, 100, 100, 0], 2, s).unwrap();
    assert_eq!(f.floored, 1);
    assert!(f.bias.iter().all(|b| b.is_finite()));
}

#[test]
fn the_bias_channel_normalises_to_unit_noise() {
    for h in [256u64, 2048, 5120] {
        let c = bias_channel(h);
        assert_eq!(c.channel, h - 1);
        let rms = ((c.k * c.k + (h as f32 - 1.0)) / h as f32).sqrt();
        assert!((c.s - c.k / rms).abs() < 1e-5);
        assert!((c.sigma * c.sigma * (h as f32 - 1.0) / (rms * rms) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn top_k_is_largest_first_with_low_index_ties() {
    assert_eq!(top_k(&[0.1, 0.9, 0.5, 0.9], 3), vec![1, 3, 2]);
    assert_eq!(top_k(&[1.0, 2.0], 2), vec![1, 0]);
}

#[test]
fn malformed_profiles_are_refused() {
    let ok = r#"{"schema":1,"source":"s","experts":2,"top_k":1,"layers":[[1,2]]}"#;
    assert_eq!(RoutingProfile::parse(ok).unwrap().layers, vec![vec![1, 2]]);
    for bad in [
        r#"{"schema":2,"source":"s","experts":2,"top_k":1,"layers":[[1,2]]}"#,
        r#"{"schema":1,"source":"s","experts":2,"top_k":3,"layers":[[1,2]]}"#,
        r#"{"schema":1,"source":"s","experts":2,"top_k":1,"layers":[[1,2,3]]}"#,
        r#"{"schema":1,"source":"s","experts":2,"top_k":1,"layers":[[0,0]]}"#,
        r#"{"schema":1,"source":"s","experts":2,"top_k":1,"layers":[[1,2]],"extra":1}"#,
    ] {
        assert!(RoutingProfile::parse(bad).is_err(), "{bad}");
    }
}
