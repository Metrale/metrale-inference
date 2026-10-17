// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`LoadLevers::from_settings`] against the load-phase readers it replaced
//! (`maybe_run_ep_worker`, `decode_ring::requested_slots`, `headroom::kv_floor_tokens`,
//! `ssm_h_fp16_preconditions`, `resolve_prefill_budget`, `resolve_vision_max_pixels`,
//! `load_weight_store`), resolved over `MapEnv`.
//!
//! Owner: server startup.
//! Invariants: none beyond the types.

use std::borrow::Cow;

use super::{DEFAULT_KV_FLOOR_TOKENS, LoadLevers};
use crate::lever_equivalence::{Old, Out, Row, assert_old_readers_hold, num, settings_from};

fn resolved(env: &[(&str, &str)]) -> anyhow::Result<LoadLevers> {
    LoadLevers::from_settings(&settings_from(env)?)
}

#[test]
fn an_empty_environment_gives_the_shipped_load() {
    assert_eq!(
        resolved(&[]).expect("defaults resolve"),
        LoadLevers {
            allow_spec_mismatch: false,
            ssm_reserve_ring_full: false,
            kv_floor_tokens: DEFAULT_KV_FLOOR_TOKENS,
            gdn_fused_conv: false,
            max_batch_tokens: None,
            vision_max_pixels: None,
            fast_load: true,
            fast_load_prefetch_shards: false,
            kv_paging: false,
        }
    );
}

#[test]
fn kv_paging_keeps_the_storage_grammar() {
    let paging = |v: &str| resolved(&[("METRALE_KV_PAGING", v)]).map(|l| l.kv_paging);
    assert!(paging(" 1 ").expect("on"));
    assert!(!paging("0").expect("off"));
    for junk in ["true", "", "2"] {
        assert!(paging(junk).is_err(), "{junk:?}");
    }
}

#[test]
fn a_zero_floor_disables_it_and_a_zero_pixel_cap_is_unset() {
    let l = resolved(&[
        ("METRALE_KV_FLOOR_TOKENS", "0"),
        ("METRALE_VISION_MAX_PIXELS", "0"),
    ])
    .expect("resolves");
    assert_eq!((l.kv_floor_tokens, l.vision_max_pixels), (0, None));
    let l = resolved(&[("METRALE_VISION_MAX_PIXELS", "1048576")]).expect("resolves");
    assert_eq!(l.vision_max_pixels, Some(1_048_576));
}

fn rows() -> Vec<Row<LoadLevers>> {
    vec![
        ("METRALE_ALLOW_SPEC_MISMATCH", Old::OptInOneOrTrue, |l| {
            Out::B(l.allow_spec_mismatch)
        }),
        ("METRALE_SSM_RESERVE_RING_FULL", Old::Present, |l| {
            Out::B(l.ssm_reserve_ring_full)
        }),
        (
            "METRALE_KV_FLOOR_TOKENS",
            Old::UintTrimmed(DEFAULT_KV_FLOOR_TOKENS as u64),
            |l| num(l.kv_floor_tokens),
        ),
        ("METRALE_GDN_FUSED_CONV", Old::OptIn, |l| {
            Out::B(l.gdn_fused_conv)
        }),
        // 2026-10-17: `None` printed as 0: the old reader's fall-through to the default budget.
        ("METRALE_MAX_BATCH_TOKENS", Old::NumUint(0), |l| {
            num(l.max_batch_tokens.unwrap_or(0))
        }),
        ("METRALE_VISION_MAX_PIXELS", Old::UintBlankZeroUnset, |l| {
            num(l.vision_max_pixels.unwrap_or(0))
        }),
        ("METRALE_FAST_LOAD", Old::OnUnlessZero, |l| {
            Out::B(l.fast_load)
        }),
        ("METRALE_FAST_LOAD_PREFETCH_SHARDS", Old::OptInWord, |l| {
            Out::B(l.fast_load_prefetch_shards)
        }),
    ]
}

/// 2026-10-17: Each reader now uses the settings crate's switch or integer grammar: switches
/// accept `1|true|on|yes` / `0|false|off|no` (trimmed, any case) and refuse anything else;
/// counts are trimmed and refuse anything that is not a non-negative integer. Every value
/// that was silently read as "off" or as the default now stops the load.
const CHANGES: &[(&str, &str, Out)] = &[
    ("opt_in_one_or_true", "TRUE", Out::B(true)),
    ("opt_in_one_or_true", "yes", Out::B(true)),
    ("opt_in_one_or_true", "", Out::Err),
    ("opt_in_one_or_true", " 1 ", Out::B(true)),
    ("opt_in_one_or_true", "garbage", Out::Err),
    ("opt_in_one_or_true", "7", Out::Err),
    ("opt_in_one_or_true", "-1", Out::Err),
    ("uint_trimmed", "true", Out::Err),
    ("uint_trimmed", "TRUE", Out::Err),
    ("uint_trimmed", "yes", Out::Err),
    ("uint_trimmed", "false", Out::Err),
    ("uint_trimmed", "off", Out::Err),
    ("uint_trimmed", "no", Out::Err),
    ("uint_trimmed", "", Out::Err),
    ("uint_trimmed", "garbage", Out::Err),
    ("uint_trimmed", "-1", Out::Err),
    ("opt_in", "true", Out::B(true)),
    ("opt_in", "TRUE", Out::B(true)),
    ("opt_in", "yes", Out::B(true)),
    ("opt_in", "", Out::Err),
    ("opt_in", " 1 ", Out::B(true)),
    ("opt_in", "garbage", Out::Err),
    ("opt_in", "7", Out::Err),
    ("opt_in", "-1", Out::Err),
    ("uint", "true", Out::Err),
    ("uint", "TRUE", Out::Err),
    ("uint", "yes", Out::Err),
    ("uint", "false", Out::Err),
    ("uint", "off", Out::Err),
    ("uint", "no", Out::Err),
    ("uint", "", Out::Err),
    ("uint", " 1 ", Out::N(Cow::Borrowed("1"))),
    ("uint", "garbage", Out::Err),
    ("uint", "-1", Out::Err),
    ("uint_blank_zero_unset", "", Out::Err),
    ("on_unless_zero", "false", Out::B(false)),
    ("on_unless_zero", "off", Out::B(false)),
    ("on_unless_zero", "no", Out::B(false)),
    ("on_unless_zero", "", Out::Err),
    ("on_unless_zero", "garbage", Out::Err),
    ("on_unless_zero", "7", Out::Err),
    ("on_unless_zero", "-1", Out::Err),
    ("opt_in_word", "yes", Out::B(true)),
    ("opt_in_word", "", Out::Err),
    ("opt_in_word", " 1 ", Out::B(true)),
    ("opt_in_word", "garbage", Out::Err),
    ("opt_in_word", "7", Out::Err),
    ("opt_in_word", "-1", Out::Err),
];

#[test]
fn the_load_levers_match_their_old_readers_except_the_recorded_changes() {
    assert_old_readers_hold(&rows(), resolved, CHANGES);
}
