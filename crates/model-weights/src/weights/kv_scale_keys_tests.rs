// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Tests for the FP8 KV scale key resolver and census.

use std::collections::BTreeSet;

use super::{KvScaleSpelling, kv_scale_census, resolve_kv_scale_keys};

const P: &str = "model.language_model.layers.3.self_attn";

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn census_of(names: &BTreeSet<String>) -> anyhow::Result<super::KvScaleCensus> {
    kv_scale_census(names.iter().map(String::as_str), |n| names.contains(n))
}

/// 2026-09-28: Path A: each exporter spelling resolves to its own two keys, and
/// the census files it under the attention prefix the loaders pass.
#[test]
fn every_spelling_resolves_to_its_keys_under_the_attention_prefix() {
    let cases = [
        (
            KvScaleSpelling::KProjOutput,
            "k_proj.k_scale",
            "v_proj.v_scale",
        ),
        (KvScaleSpelling::AttnModule, "attn.k_scale", "attn.v_scale"),
        (KvScaleSpelling::Bare, "k_scale", "v_scale"),
    ];
    for (spelling, k, v) in cases {
        let (k, v) = (format!("{P}.{k}"), format!("{P}.{v}"));
        let names = set(&[&k, &v, &format!("{P}.q_proj.weight")]);
        let keys = resolve_kv_scale_keys(|n| names.contains(n), P)
            .expect("resolves")
            .expect("present");
        assert_eq!((keys.spelling, &keys.k, &keys.v), (spelling, &k, &v));
        let census = census_of(&names).expect("census");
        assert_eq!(
            census.layers.keys().collect::<Vec<_>>(),
            [P],
            "{spelling:?}: filed under the loader's prefix, not a deeper one"
        );
    }
}

/// 2026-09-28: Path B, regression: the unsloth Qwen3.6/3.8-27B spelling. The old
/// loader asked only for `k_proj.k_scale`, so these 16 layers ran at 1.0 while
/// the serve log counted them as loaded.
#[test]
fn the_bare_self_attn_spelling_is_found() {
    let names = set(&[&format!("{P}.k_scale"), &format!("{P}.v_scale")]);
    let keys = resolve_kv_scale_keys(|n| names.contains(n), P)
        .expect("resolves")
        .expect("the bare spelling must be found, not treated as absent");
    assert_eq!(keys.k, format!("{P}.k_scale"));
}

/// 2026-09-28: Path B: two spellings for one layer is an error naming both, not
/// a silent pick.
#[test]
fn two_spellings_for_one_layer_is_an_error() {
    let names = set(&[
        &format!("{P}.k_proj.k_scale"),
        &format!("{P}.v_proj.v_scale"),
        &format!("{P}.k_scale"),
        &format!("{P}.v_scale"),
    ]);
    let err = resolve_kv_scale_keys(|n| names.contains(n), P).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("ambiguous"), "{msg}");
    assert!(msg.contains("k_proj.k_scale") && msg.contains(&format!("{P}.k_scale")));
    assert!(census_of(&names).is_err(), "the census refuses it too");
}

/// 2026-09-28: Path B: a K without its V (and the reverse) is an error in every
/// spelling, including across spellings (K in one, V in another).
#[test]
fn a_half_pair_is_an_error() {
    for names in [
        set(&[&format!("{P}.k_scale")]),
        set(&[&format!("{P}.v_proj.v_scale")]),
        set(&[&format!("{P}.attn.k_scale")]),
        set(&[&format!("{P}.k_proj.k_scale"), &format!("{P}.v_scale")]),
    ] {
        let err = resolve_kv_scale_keys(|n| names.contains(n), P).unwrap_err();
        assert!(
            format!("{err:#}").contains("incomplete"),
            "{names:?}: {err:#}"
        );
        assert!(census_of(&names).is_err(), "{names:?}");
    }
}

/// 2026-09-28: No key in any spelling is `None`, and lookalikes are not scales.
#[test]
fn absent_and_lookalike_names_resolve_to_none() {
    let names = set(&[
        &format!("{P}.k_proj.weight_scale"),
        &format!("{P}.k_proj.input_scale"),
        &format!("{P}.attnk_scale"),
        "model.language_model.layers.30.self_attn.k_scale",
        "model.language_model.layers.30.self_attn.v_scale",
    ]);
    assert_eq!(
        resolve_kv_scale_keys(|n| names.contains(n), P).expect("ok"),
        None,
        "layer 30's keys are not layer 3's"
    );
    let census = census_of(&names).expect("census");
    assert_eq!(
        census.layers.keys().collect::<Vec<_>>(),
        ["model.language_model.layers.30.self_attn"]
    );
}

/// 2026-09-28: A partial set across layers (some layers ship scales, some do
/// not) is not a resolver error; the census counts only the layers that do,
/// which is what serve compares with the attention-layer count.
#[test]
fn the_census_counts_only_layers_with_scales_and_reports_their_spellings() {
    let names = set(&[
        "m.layers.3.self_attn.k_scale",
        "m.layers.3.self_attn.v_scale",
        "m.layers.7.self_attn.k_proj.k_scale",
        "m.layers.7.self_attn.v_proj.v_scale",
        "m.layers.11.self_attn.q_proj.weight",
    ]);
    let census = census_of(&names).expect("census");
    assert_eq!(census.len(), 2);
    assert_eq!(census.spellings(), ["k_proj.k_scale", "k_scale"]);
}
