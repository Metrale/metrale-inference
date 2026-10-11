// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Provenance on every value that may be unmeasured. An unmeasured value is
//! `{"absent": "<reason>"}`, never zero; a value computed from other values is `derived`.
//! Serialized externally tagged: `{"measured": 42}`, `{"derived": 42}`,
//! `{"absent": "unsupported"}`.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

/// 2026-10-11: Why a value is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbsentReason {
    /// 2026-10-11: The source cannot produce this quantity (no counter, or the telemetry
    /// layer that holds it is switched off).
    Unsupported,
    /// 2026-10-11: The source exists but gave no usable reading.
    Unavailable,
    /// 2026-10-11: The quantity has no meaning for this record (TPOT of a one-token answer).
    NotApplicable,
    /// 2026-10-11: The value existed but could not be recorded.
    Lost,
    /// 2026-10-11: Only known later in the record's life.
    NotYet,
}

/// 2026-10-11: A value with its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Field<T> {
    Measured(T),
    Derived(T),
    Absent(AbsentReason),
}

impl<T> Field<T> {
    /// 2026-10-11: The value, measured or derived.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Measured(v) | Self::Derived(v) => Some(v),
            Self::Absent(_) => None,
        }
    }
}

impl Field<u32> {
    /// 2026-10-11: A measured count. A count past `u32::MAX` cannot be written into this
    /// field, so it is `lost`, not clamped.
    pub fn count(n: usize) -> Self {
        u32::try_from(n).map_or(Self::Absent(AbsentReason::Lost), Self::Measured)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_is_not_zero_on_the_wire() {
        let zero = serde_json::to_string(&Field::Measured(0u32)).unwrap();
        let absent =
            serde_json::to_string(&Field::<u32>::Absent(AbsentReason::Unsupported)).unwrap();
        assert_eq!(zero, r#"{"measured":0}"#);
        assert_eq!(absent, r#"{"absent":"unsupported"}"#);
        assert_eq!(
            Field::<u32>::Absent(AbsentReason::NotApplicable).value(),
            None
        );
        assert_eq!(Field::Derived(7u32).value(), Some(&7));
    }

    #[test]
    fn count_past_u32_is_lost_not_clamped() {
        assert_eq!(Field::count(5), Field::Measured(5));
        if let Ok(big) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert_eq!(Field::count(big), Field::Absent(AbsentReason::Lost));
        }
    }
}
