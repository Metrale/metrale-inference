// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `WeightsSettings` against the model-weights readers it replaced, including the
//! `gpu-sys` fallback-chain helpers the RDMA rails used (copied below over a pair list instead
//! of the process environment).
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{Rail, WeightsSettings};
use crate::groups::device::ProjectionError;
use crate::groups::equivalence::{
    VALUES, Want, check, check_values, eq_one, eq_one_to_switch, resolved,
};

fn weights(s: &crate::InferenceServerSettings) -> Result<WeightsSettings, ProjectionError> {
    Ok(WeightsSettings::from(s))
}

type Pairs<'a> = &'a [(&'a str, &'a str)];

fn get(pairs: Pairs<'_>, k: &str) -> Option<String> {
    pairs
        .iter()
        .find(|(n, _)| *n == k)
        .map(|(_, v)| String::from(*v))
}

/// 2026-10-17: `gpu_sys::env::first_set`.
fn old_first_set(pairs: Pairs<'_>, keys: &[&str], default: &str) -> String {
    keys.iter()
        .find_map(|k| get(pairs, k))
        .unwrap_or_else(|| default.into())
}

/// 2026-10-17: `gpu_sys::env::first_nonempty`.
fn old_first_nonempty(pairs: Pairs<'_>, keys: &[&str], default: &str) -> String {
    keys.iter()
        .find_map(|k| get(pairs, k).filter(|v| !v.is_empty()))
        .unwrap_or_else(|| default.into())
}

/// 2026-10-17: `gpu_sys::env::first_set_u32`.
fn old_first_set_u32(pairs: Pairs<'_>, keys: &[&str], default: u32) -> u32 {
    keys.iter()
        .find_map(|k| get(pairs, k).and_then(|v| v.parse().ok()))
        .unwrap_or(default)
}

const W_DEV: &str = "METRALE_WEIGHT_RDMA_DEV";
const E_DEV: &str = "METRALE_EXPERT_RDMA_DEV";
const L_DEV: &str = "METRALE_LORA_RDMA_DEV";
const W_GID: &str = "METRALE_WEIGHT_RDMA_GID";
const E_GID: &str = "METRALE_EXPERT_RDMA_GID";
const L_GID: &str = "METRALE_LORA_RDMA_GID";
const W2_DEV: &str = "METRALE_WEIGHT_RAIL2_DEV";
const E2_DEV: &str = "METRALE_EXPERT_RAIL2_DEV";
const W2_GID: &str = "METRALE_WEIGHT_RAIL2_GID";
const E2_GID: &str = "METRALE_EXPERT_RAIL2_GID";

/// 2026-10-17: The three rails exactly as `weight_tier_rdma` and `weight_lora_rdma` built them.
fn old_rails(pairs: Pairs<'_>) -> (Rail, Option<Rail>, Rail) {
    let rail0 = Rail {
        dev: old_first_nonempty(pairs, &[W_DEV, E_DEV], "roceP2p1s0f1"),
        gid: old_first_set_u32(pairs, &[W_GID, E_GID], 3),
    };
    let dual = get(pairs, "METRALE_WEIGHT_DUAL_RAIL").as_deref() == Some("1");
    let rail1 = dual.then(|| Rail {
        dev: old_first_nonempty(pairs, &[W2_DEV, E2_DEV], "rocep1s0f1"),
        gid: old_first_set_u32(pairs, &[W2_GID, E2_GID], 3),
    });
    let lora = Rail {
        dev: old_first_set(pairs, &[L_DEV, W_DEV, E_DEV], "roceP2p1s0f1"),
        gid: old_first_set_u32(pairs, &[L_GID], 3),
    };
    (rail0, rail1, lora)
}

/// 2026-10-18: The GID rows that are `Int` settings rows (the storage lane's retype): the value
/// is trimmed, and one that is not an integer fails to resolve instead of falling back.
const INT_GIDS: [&str; 3] = [W_GID, E_GID, E2_GID];

fn new_rails(pairs: Pairs<'_>) -> Option<(Rail, Option<Rail>, Rail)> {
    let w = WeightsSettings::from(&resolved(pairs).ok()?);
    Some((w.weight_rail0, w.weight_rail1, w.lora_rail))
}

/// 2026-10-18: What the old chains give for `pairs` under the `Int` grammar of [`INT_GIDS`]:
/// `None` where an `Int` GID does not parse, else the old rails over the trimmed values.
fn old_rails_int_gids(pairs: Pairs<'_>) -> Option<(Rail, Option<Rail>, Rail)> {
    let mut trimmed = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        if INT_GIDS.contains(k) {
            v.trim_ascii().parse::<i64>().ok()?;
            trimmed.push((*k, v.trim_ascii()));
        } else {
            trimmed.push((*k, *v));
        }
    }
    Some(old_rails(&trimmed))
}

#[test]
fn every_rdma_lever_alone_matches_the_old_chains() {
    let dual = ("METRALE_WEIGHT_DUAL_RAIL", "1");
    for key in [
        W_DEV, E_DEV, L_DEV, W_GID, E_GID, L_GID, W2_DEV, E2_DEV, W2_GID, E2_GID,
    ] {
        for v in VALUES.iter().flatten().chain(&["", "7", " 7", "-1"]) {
            let pairs = [(key, *v), dual];
            assert_eq!(new_rails(&pairs), old_rails_int_gids(&pairs), "{key}={v:?}");
        }
        assert_eq!(new_rails(&[dual]), Some(old_rails(&[dual])), "{key} unset");
    }
}

#[test]
fn int_gid_rows_trim_and_refuse_garbage() {
    for key in INT_GIDS {
        assert_eq!(new_rails(&[(key, "x")]), None, "{key}=x");
        assert_eq!(new_rails(&[(key, "")]), None, "{key} empty");
        let spaced =
            new_rails(&[(key, " 7"), ("METRALE_WEIGHT_DUAL_RAIL", "1")]).expect(" 7 resolves");
        let gid = if key == E2_GID {
            spaced.1.expect("dual rail").gid
        } else {
            spaced.0.gid
        };
        assert_eq!(gid, 7, "{key}");
    }
}

#[test]
fn rdma_chains_fall_back_in_the_old_order() {
    let cases: [Pairs<'_>; 7] = [
        &[(W_DEV, ""), (E_DEV, "e0")],
        &[(W_DEV, "w0"), (E_DEV, "e0")],
        &[(L_DEV, ""), (W_DEV, "w0")],
        &[(W_GID, "-1"), (E_GID, "5")],
        &[(W_GID, "4"), (E_GID, "5"), (L_GID, "6")],
        &[(W2_DEV, ""), (E2_DEV, "e2"), (W2_GID, "bad"), (E2_GID, "9")],
        &[
            (W2_DEV, "w2"),
            (E2_GID, "9"),
            ("METRALE_WEIGHT_DUAL_RAIL", "0"),
        ],
    ];
    for pairs in cases {
        let with_dual: Vec<_> = pairs
            .iter()
            .copied()
            .chain([("METRALE_WEIGHT_DUAL_RAIL", "1")])
            .collect();
        assert_eq!(new_rails(pairs), old_rails_int_gids(pairs), "{pairs:?}");
        if !pairs.iter().any(|(k, _)| *k == "METRALE_WEIGHT_DUAL_RAIL") {
            assert_eq!(
                new_rails(&with_dual),
                old_rails_int_gids(&with_dual),
                "{with_dual:?}"
            );
        }
    }
}

#[test]
fn dual_rail_moves_onto_a_switch() {
    check(
        "METRALE_WEIGHT_DUAL_RAIL",
        eq_one,
        |s| weights(s).map(|w| w.weight_rail1.is_some()),
        &eq_one_to_switch(),
    );
}

#[test]
fn eq_one_switches_match_the_old_readers_except_true_and_garbage() {
    type Field = fn(&WeightsSettings) -> bool;
    let switches: [(&str, Field); 3] = [
        ("METRALE_GGUF_NATIVE_Q2", |w| w.gguf_native_q2),
        ("METRALE_GGUF_FORCE_CPU", |w| w.gguf_force_cpu),
        ("METRALE_DS41_KEEP_PAGE_CACHE", |w| w.keep_page_cache),
    ];
    for (env, field) in switches {
        check(
            env,
            eq_one,
            |s| weights(s).map(|w| field(&w)),
            &eq_one_to_switch(),
        );
    }
}

#[test]
fn direct_reads_turn_off_for_every_off_spelling_and_reject_garbage() {
    check(
        "METRALE_DS41_DIRECT_READS",
        |v| v != Some("0"),
        |s| weights(s).map(|w| w.direct_reads),
        &[
            (Some("off"), Want::New(false)),
            (Some("garbage"), Want::Err),
        ],
    );
}

#[test]
fn kquant_texts_turn_off_only_for_zero() {
    type Field = fn(&WeightsSettings) -> bool;
    let texts: [(&str, Field); 2] = [
        ("METRALE_DS41_KQUANT_RESIDENT", |w| w.kquant_resident),
        ("METRALE_DS41_Q6K_HEAD", |w| w.q6k_head),
    ];
    for (env, field) in texts {
        check(
            env,
            |v| v != Some("0"),
            |s| weights(s).map(|w| field(&w)),
            &[],
        );
    }
}

#[test]
fn q2_group_and_evict_pct_keep_their_lenient_parses() {
    let extra = |x: &[Option<&'static str>]| -> Vec<_> {
        VALUES.iter().copied().chain(x.iter().copied()).collect()
    };
    check_values(
        "METRALE_GGUF_Q2_GROUP",
        &extra(&[Some("64"), Some(" 64"), Some("128")]),
        |v| if v == Some("64") { 64 } else { 128 },
        |s| weights(s).map(|w| w.q2_group),
        &[],
    );
    check_values(
        "METRALE_DS41_EVICT_RANDOM_PCT",
        &extra(&[Some("100"), Some("101"), Some("-1")]),
        |v| {
            v.and_then(|v| v.parse::<usize>().ok())
                .filter(|p| *p <= 100)
                .unwrap_or(8)
        },
        |s| weights(s).map(|w| w.evict_random_pct),
        &[],
    );
}
