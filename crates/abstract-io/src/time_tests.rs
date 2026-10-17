// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MonoNs`/`DurNs` arithmetic saturates instead of wrapping or panicking.

use super::{DurNs, MonoNs};

#[test]
fn since_is_the_forward_gap_and_zero_backwards() {
    assert_eq!(MonoNs(1_500).since(MonoNs(500)), DurNs(1_000));
    assert_eq!(MonoNs(500).since(MonoNs(1_500)), DurNs(0));
}

#[test]
fn plus_saturates_at_the_top() {
    assert_eq!(MonoNs(10).plus(DurNs(5)), MonoNs(15));
    assert_eq!(MonoNs(u64::MAX - 1).plus(DurNs(5)), MonoNs(u64::MAX));
}

#[test]
fn unit_constructors_scale_and_saturate() {
    assert_eq!(DurNs::from_ms(3), DurNs(3_000_000));
    assert_eq!(DurNs::from_us(3), DurNs(3_000));
    assert_eq!(DurNs::from_ms(u64::MAX), DurNs(u64::MAX));
    assert_eq!(DurNs::from_us(u64::MAX), DurNs(u64::MAX));
}

#[test]
fn as_ms_f64_keeps_the_fraction() {
    assert_eq!(DurNs(1_500_000).as_ms_f64(), 1.5);
    assert_eq!(DurNs(250).as_ms_f64(), 0.000_25);
}
