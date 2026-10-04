// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the spec-cost driver's parameters and request pins.
//!
//! Owner: bench, spec-cost.
//! Invariants: none beyond the types.

use super::*;

fn configured(overrides: &[(&str, ParamValue)]) -> Result<SpecCost> {
    let mut b = SpecCost::default();
    let mut values = ParamValues::defaults(&b.parameters());
    for (key, value) in overrides {
        values.set(*key, value.clone());
    }
    b.configure(&values)?;
    Ok(b)
}

#[test]
fn k_has_no_usable_default() {
    let err = configured(&[]).err().expect("an unstated k is refused");
    assert!(format!("{err:#}").contains("--param k="), "{err:#}");
}

#[test]
fn a_stated_k_and_the_default_widths_configure() {
    for k in 0..=MAX_K {
        let b = configured(&[("k", ParamValue::Int(k))]).unwrap();
        assert_eq!(i64::from(b.k), k);
        assert_eq!(b.widths, [1, 2, 4, 8, 16]);
        assert_eq!(b.osl, 1024);
        assert_eq!(b.window, Duration::from_secs(8));
        assert_eq!(b.settle, Duration::from_secs(3));
    }
    assert!(configured(&[("k", ParamValue::Int(MAX_K + 1))]).is_err());
}

#[test]
fn widths_must_be_strictly_ascending() {
    for widths in [vec![1, 4, 2], vec![2, 2]] {
        let err = configured(&[
            ("k", ParamValue::Int(3)),
            ("widths", ParamValue::IntList(widths.clone())),
        ])
        .err()
        .unwrap_or_else(|| panic!("{widths:?} was accepted"));
        assert!(format!("{err:#}").contains("ascending"), "{err:#}");
    }
}

#[test]
fn request_pins_thinking_off_greedy_and_the_output_budget() {
    let b = configured(&[("k", ParamValue::Int(2)), ("osl", ParamValue::Int(777))]).unwrap();
    let body = b.request_body("m", "spec-cost-n4-0-1-1");
    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["seed"], 0);
    assert_eq!(body["max_tokens"], 777);
    assert_eq!(body["stream"], true);
    assert_eq!(body["presence_penalty"], 0.0);
    assert_eq!(body["frequency_penalty"], 0.0);
    let prompt = body["messages"][0]["content"].as_str().unwrap();
    assert!(prompt.ends_with(crate::benchmarks::concurrency::ESSAY_TASK));
    let other = b.request_body("m", "spec-cost-n4-1-1-1");
    assert_ne!(
        body["messages"], other["messages"],
        "distinct tags must give distinct prompts, so no stream hits another's cached prefix"
    );
}
