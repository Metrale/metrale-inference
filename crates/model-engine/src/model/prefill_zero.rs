// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Opt-in levers for the zeroing a prefill does before its first launch, measured on
//! GLM-5.3 TP3 (cold-4k nsys pf4kZ): with a communicator every prefill chunk zeroed the WHOLE
//! arena (`BufferArena::zero_all`, 2.66 GB at 8208 arena tokens, 14 ms of GPU time whatever the
//! prompt length). It sits on every cold TTFT.
//!
//! Owner: model-engine (prefill).
//! Invariants:
//! - Unset or `0` keeps today's behaviour exactly; another value than the ones named is an
//!   error at first use, never a silent default.

use std::sync::OnceLock;

use anyhow::Result;

/// 2026-10-12: `METRALE_PREFILL_ZERO_ROWS=1`: with a communicator, a prefill chunk zeroes the
/// rows it will touch (`BufferArena::zero_all_rows` at the chunk length rounded up to 16, the
/// cuBLASLt pad) instead of the whole arena. Read once.
pub(crate) fn prefill_zero_rows() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_switch(
            "METRALE_PREFILL_ZERO_ROWS",
            std::env::var("METRALE_PREFILL_ZERO_ROWS").ok().as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-12: A 0/1 switch, pure.
pub(crate) fn parse_switch(name: &str, v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{name}={other:?}: expected 0 or 1")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-12: Both levers parse strictly: the switch takes 0/1, the block mode takes
    /// 0/block/range; anything else is an error naming the variable.
    #[test]
    fn the_zeroing_levers_parse_strictly() {
        assert_eq!(parse_switch("X", None), Ok(false));
        assert_eq!(parse_switch("X", Some("1")), Ok(true));
        assert!(parse_switch("X", Some("yes")).unwrap_err().contains("X="));
    }
}
