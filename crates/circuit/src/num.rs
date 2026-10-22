// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The one float-to-integer conversion the estimates use, named so no other cast
//! of an `f64` hides in the crate (BU64: the cast lints are denied crate-wide).
//!
//! Owner: metrale-circuit.
//! Invariants: [`sat_u64`] is exactly Rust's `f64 as u64`: toward zero, saturating at both
//! ends, NaN to 0, on every target.

/// 2026-10-22: `x` toward zero, saturating at `0` and `u64::MAX`; NaN is `0`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "2026-10-22: the saturating `as` is the documented semantics of this helper"
)]
pub(crate) fn sat_u64(x: f64) -> u64 {
    x as u64
}

#[cfg(test)]
mod tests {
    use super::sat_u64;

    #[test]
    fn truncates_toward_zero_and_saturates() {
        assert_eq!(sat_u64(2.9), 2);
        assert_eq!(sat_u64(-3.5), 0);
        assert_eq!(sat_u64(f64::NAN), 0);
        assert_eq!(sat_u64(1e30), u64::MAX);
        assert_eq!(sat_u64(f64::INFINITY), u64::MAX);
    }
}
