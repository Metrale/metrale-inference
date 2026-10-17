// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`legacy_env_weights_settings_from`] against the `WeightsSettings` projection that
//! replaces it, over every value in `VALUES` (plus edges), and the RDMA rail chains over the
//! fallback combinations: equal except for the listed deviations, which are the behaviour change
//! the 2c-device-wire commit makes when it deletes the legacy reader.
//!
//! Owner: model-weights.
//! Invariants: never reads or writes the process environment.

use metrale_settings::InferenceServerSettings;
use metrale_settings::groups::device::ProjectionError;
use metrale_settings::groups::device_weights::{Rail, WeightsSettings};
use metrale_settings::groups::equivalence::{
    VALUES, Want, check, check_values, eq_one_to_switch, resolved, var_in, var_one,
};

use super::legacy_env_weights_settings_from;

fn weights(s: &InferenceServerSettings) -> Result<WeightsSettings, ProjectionError> {
    Ok(WeightsSettings::from(s))
}

/// 2026-10-17: The legacy reader's view of one field with only `env` set to the value.
fn legacy<T>(
    env: &'static str,
    field: impl Fn(&WeightsSettings) -> T,
) -> impl Fn(Option<&str>) -> T {
    move |v| field(&legacy_env_weights_settings_from(var_one(env, v)))
}

type Pairs<'a> = &'a [(&'a str, &'a str)];
type Rails = (Rail, Option<Rail>, Rail);

fn rails(w: WeightsSettings) -> Rails {
    (w.weight_rail0, w.weight_rail1, w.lora_rail)
}

fn old_rails(pairs: Pairs<'_>) -> Rails {
    rails(legacy_env_weights_settings_from(var_in(pairs)))
}

const DUAL: &str = "METRALE_WEIGHT_DUAL_RAIL";
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

#[test]
fn an_empty_environment_reads_as_the_declared_settings() {
    assert_eq!(
        legacy_env_weights_settings_from(var_in(&[])),
        WeightsSettings::declared().expect("declared defaults resolve")
    );
}

fn new_rails(pairs: Pairs<'_>) -> Option<Rails> {
    Some(rails(WeightsSettings::from(&resolved(pairs).ok()?)))
}

/// 2026-10-18: The GID rows that are `Int` settings rows (the storage lane's retype): the value
/// is trimmed, and one that is not an integer fails to resolve instead of falling back.
const INT_GIDS: [&str; 3] = [W_GID, E_GID, E2_GID];

/// 2026-10-18: The legacy rails for `pairs` under the `Int` grammar of [`INT_GIDS`]: `None` where
/// an `Int` GID does not parse, else the legacy rails over the trimmed values.
fn old_rails_int_gids(pairs: Pairs<'_>) -> Option<Rails> {
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
    let dual = (DUAL, "1");
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
        let spaced = new_rails(&[(key, " 7"), (DUAL, "1")]).expect(" 7 resolves");
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
        &[(W2_DEV, "w2"), (E2_GID, "9"), (DUAL, "0")],
    ];
    for pairs in cases {
        assert_eq!(new_rails(pairs), old_rails_int_gids(pairs), "{pairs:?}");
        if !pairs.iter().any(|(k, _)| *k == DUAL) {
            let with_dual: Vec<_> = pairs.iter().copied().chain([(DUAL, "1")]).collect();
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
    let field = |w: &WeightsSettings| w.weight_rail1.is_some();
    check(
        DUAL,
        legacy(DUAL, field),
        |s| weights(s).map(|w| field(&w)),
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
            legacy(env, field),
            |s| weights(s).map(|w| field(&w)),
            &eq_one_to_switch(),
        );
    }
}

#[test]
fn direct_reads_turn_off_for_every_off_spelling_and_reject_garbage() {
    const ENV: &str = "METRALE_DS41_DIRECT_READS";
    let field = |w: &WeightsSettings| w.direct_reads;
    check(
        ENV,
        legacy(ENV, field),
        |s| weights(s).map(|w| field(&w)),
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
        check_values(
            env,
            &[VALUES.as_slice(), &[Some(""), Some(" 0")]].concat(),
            legacy(env, field),
            |s| weights(s).map(|w| field(&w)),
            &[],
        );
    }
}

#[test]
fn q2_group_and_evict_pct_keep_their_lenient_parses() {
    const GROUP: &str = "METRALE_GGUF_Q2_GROUP";
    let group = |w: &WeightsSettings| w.q2_group;
    check_values(
        GROUP,
        &[VALUES.as_slice(), &[Some("64"), Some(" 64"), Some("128")]].concat(),
        legacy(GROUP, group),
        |s| weights(s).map(|w| group(&w)),
        &[],
    );
    const PCT: &str = "METRALE_DS41_EVICT_RANDOM_PCT";
    let pct = |w: &WeightsSettings| w.evict_random_pct;
    check_values(
        PCT,
        &[VALUES.as_slice(), &[Some("100"), Some("101"), Some("-1")]].concat(),
        legacy(PCT, pct),
        |s| weights(s).map(|w| pct(&w)),
        &[],
    );
}
