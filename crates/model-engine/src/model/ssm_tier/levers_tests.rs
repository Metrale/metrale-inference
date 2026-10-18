// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `SsmTierLevers` against frozen copies of the environment readers it replaced,
//! through `MapEnv`-resolved settings (never the process environment).
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants: every probe that reads differently from the old reader is asserted by name,
//! and every row whose old reader cared about an empty value is probed with `""` too.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

use super::SsmTierLevers;
use crate::model::ssm_tier::unified::unified_flag_truthy;

fn levers(s: &metrale_settings::InferenceServerSettings) -> SsmTierLevers {
    SsmTierLevers::from_settings(s)
}

/// 2026-10-17: Asserts the new reader agrees with `old` on every probe and on `""`.
fn unchanged<T: PartialEq + core::fmt::Debug>(
    env: &'static str,
    kind: Old,
    old: impl Fn(Option<&str>) -> T,
    new: impl Fn(&SsmTierLevers) -> T,
) {
    assert_eq!(check(env, kind, &old, |s| new(&levers(s))), vec![], "{env}");
    let s = settings_from(&[(env, "")]).unwrap();
    assert_eq!(new(&levers(&s)), old(Some("")), "{env}=\"\"");
}

fn non_empty(v: Option<&str>) -> Option<String> {
    v.filter(|s| !s.is_empty()).map(str::to_owned)
}

/// 2026-10-17: Rows whose readers keep their exact old behaviour (presence rows read by
/// `var_os`, text and path rows read verbatim or filtered for emptiness).
#[test]
fn presence_and_text_rows_read_exactly_as_before() {
    let present = |v: Option<&str>| v.is_some();
    unchanged("METRALE_SSM_TIER", Old::Presence, present, |l| l.enabled);
    unchanged("METRALE_SSM_SWAP_NS", Old::Presence, present, |l| {
        l.ns_overridden
    });
    unchanged("METRALE_SSM_DECODE_NS", Old::Presence, present, |l| {
        l.ns_overridden
    });
    unchanged(
        "METRALE_SSM_DECODE_CLIENT_ID",
        Old::Presence,
        present,
        |l| l.decode_client_id_set,
    );
    unchanged(
        "METRALE_SSM_DECODE_RING_ROLL",
        Old::Presence,
        present,
        |l| l.decode_ring_roll_set,
    );
    unchanged("METRALE_SSM_RDMA_TIER", Old::Verbatim, non_empty, |l| {
        l.rdma_peer().map(str::to_owned)
    });
    unchanged(
        "METRALE_SSM_RDMA_ARENA_SLOTS",
        Old::Verbatim,
        |v| v.and_then(|s| s.parse::<usize>().ok()).unwrap_or(512),
        |l| l.rdma_arena_slots,
    );
    unchanged(
        "METRALE_SSM_DECODE_TIER",
        Old::Verbatim,
        |v| v.map(str::to_owned),
        |l| l.decode_tier.clone(),
    );
    unchanged(
        "METRALE_SSM_DECODE_NVME_DIR",
        Old::Verbatim,
        non_empty,
        |l| non_empty(l.decode_nvme_dir.as_deref()),
    );
    unchanged(
        "METRALE_SSM_DECODE_RDMA_TIER",
        Old::Verbatim,
        non_empty,
        |l| non_empty(l.decode_rdma_tier.as_deref()),
    );
    unchanged(
        "METRALE_SSM_TIER_UNIFIED",
        Old::Verbatim,
        unified_flag_truthy,
        |l| l.unified,
    );
    unchanged("METRALE_SSM_TIER_SWAP_DIR", Old::Verbatim, non_empty, |l| {
        non_empty(l.swap_dir.as_deref())
    });
    unchanged(
        "METRALE_SSM_TIER_DISK_GB",
        Old::Verbatim,
        |v| (v.is_some_and(|s| !s.is_empty()), v.map(str::to_owned)),
        |l| (l.disk_gb_requested(), l.disk_gb.clone()),
    );
    unchanged(
        "METRALE_MODEL_ID",
        Old::Verbatim,
        |v| v.unwrap_or_default().to_owned(),
        |l| l.model_id.clone(),
    );
}

/// 2026-10-17: The selector and the fingerprint warning read `== "1"`; `true` now arms paging
/// mode and garbage is a startup error.
#[test]
fn swap_changes_only_where_the_switch_rule_says() {
    let changes = check(
        "METRALE_SSM_SWAP",
        Old::OptIn,
        |v: Option<&str>| v == Some("1"),
        |s| levers(s).swap,
    );
    assert_eq!(
        changes,
        vec![
            ("METRALE_SSM_SWAP", Some("true"), Outcome::Changed),
            ("METRALE_SSM_SWAP", Some("garbage"), Outcome::Rejected),
        ]
    );
}

/// 2026-10-17: The capability gate counted `METRALE_SSM_SWAP` by presence; an explicit off
/// value (`0`, `off`) is no longer a tier request on a model without recurrent state.
#[test]
fn capability_gate_counts_swap_only_when_it_is_on() {
    let changes = check(
        "METRALE_SSM_SWAP",
        Old::Presence,
        |v: Option<&str>| v.is_some(),
        |s| {
            levers(s)
                .requested_tier_vars()
                .contains(&"METRALE_SSM_SWAP")
        },
    );
    assert_eq!(
        changes,
        vec![
            ("METRALE_SSM_SWAP", Some("0"), Outcome::Changed),
            ("METRALE_SSM_SWAP", Some("off"), Outcome::Changed),
            ("METRALE_SSM_SWAP", Some("garbage"), Outcome::Rejected),
        ]
    );
}

/// 2026-10-17: The gate lists the requested vars in the old `SSM_TIER_VARS` order, and an
/// empty text value still counts as a request.
#[test]
fn capability_gate_lists_requests_in_the_old_order() {
    let s = settings_from(&[
        ("METRALE_SSM_DECODE_RING_ROLL", ""),
        ("METRALE_SSM_DECODE_TIER", "nvme"),
        ("METRALE_SSM_SWAP", "1"),
        ("METRALE_SSM_RDMA_TIER", ""),
        ("METRALE_SSM_TIER", "0"),
    ])
    .unwrap();
    assert_eq!(
        levers(&s).requested_tier_vars(),
        [
            "METRALE_SSM_TIER",
            "METRALE_SSM_RDMA_TIER",
            "METRALE_SSM_SWAP",
            "METRALE_SSM_DECODE_TIER",
            "METRALE_SSM_DECODE_RING_ROLL",
        ]
    );
    let none = settings_from(&[("METRALE_SSM_SWAP_NS", "7")]).unwrap();
    assert!(levers(&none).requested_tier_vars().is_empty());
}

/// 2026-10-17: Every integer keeps its slot count (a negative one the default, `0` the floor
/// of 1, as before); a non-integer, `""` included (was 64), is now a startup error, and
/// surrounding spaces parse.
#[test]
fn hot_slots_match_the_old_reader_where_they_parse() {
    let old = |v: Option<&str>| v.and_then(|s| s.parse::<usize>().ok()).unwrap_or(64).max(1);
    let changes = check("METRALE_SSM_TIER_SLOTS", Old::Number, old, |s| {
        levers(s).hot_slots
    });
    let rejected =
        ["true", "off", "garbage"].map(|p| ("METRALE_SSM_TIER_SLOTS", Some(p), Outcome::Rejected));
    assert_eq!(changes, rejected);
    for v in ["0", "16", "-4"] {
        let s = settings_from(&[("METRALE_SSM_TIER_SLOTS", v)]).unwrap();
        assert_eq!(levers(&s).hot_slots, old(Some(v)), "{v:?}");
    }
    let s = settings_from(&[("METRALE_SSM_TIER_SLOTS", " 16 ")]).unwrap();
    assert_eq!(levers(&s).hot_slots, 16);
    assert!(settings_from(&[("METRALE_SSM_TIER_SLOTS", "")]).is_err());
}
