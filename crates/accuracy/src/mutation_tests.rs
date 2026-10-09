// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Every mutation spelling round-trips; unknown spellings are refused.

use super::*;

#[test]
fn spellings_round_trip() {
    for s in [
        "corrupt_block_scale",
        "swap_scale_granularity",
        "wrong_rope_base",
        "split_off_by_one",
        "zero_expert",
        "kv_page_swap",
        "state_stale",
        "accumulate:bf16",
        "accumulate:f16",
        "symbol:w4a16_gemv::w4a16_gemv_qg",
    ] {
        let m = Mutation::parse(s).unwrap_or_else(|| panic!("{s}"));
        assert_eq!(m.name(), s);
    }
    assert!(
        Mutation::parse("accumulate:f32").is_none(),
        "f32 is the declared accumulator, not a mutation"
    );
    assert!(Mutation::parse("symbol:nomodule").is_none());
    assert!(Mutation::parse("corrupt").is_none());
    assert!(Mutation::parse("accumulate:bf16").unwrap().emulated());
}
