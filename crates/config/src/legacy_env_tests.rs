// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Each [`crate::legacy_env`] reader (and the server's old prefix-cache reader)
//! against the settings field that replaces it, over the same values; every difference must be
//! listed in the lever's `changes` table.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use metrale_io_host_mem::MapEnv;
use metrale_settings::InferenceServerSettings;
use metrale_settings::sources::{CliLayer, RecipeLayer};

use crate::PrefixCacheGate;
use crate::legacy_env::{
    allow_partial_targets_from, gemma4_lmhead_nvfp4_from, glm_dsa_indexer_paged_from,
    glm_vision_enabled_env, softcap_override_from,
};

const VALUES: [Option<&str>; 15] = [
    None,
    Some("1"),
    Some("true"),
    Some("TRUE"),
    Some("yes"),
    Some("on"),
    Some("0"),
    Some("false"),
    Some("off"),
    Some(""),
    Some(" 1 "),
    Some("garbage"),
    Some("25.0"),
    Some("nan"),
    Some("-1"),
];

/// 2026-10-17: A lever value as its reader used it; `Err` stops the serve at resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Out {
    B(bool),
    /// 2026-10-17: An `f32`, by its bits so `NaN` compares.
    F(Option<u32>),
    Err,
}

fn f(v: Option<f32>) -> Out {
    Out::F(v.map(f32::to_bits))
}

fn settings(var: &str, value: Option<&str>) -> Option<InferenceServerSettings> {
    let env = MapEnv::new(value.map(|v| (var, v)));
    metrale_settings::resolve(&env, &CliLayer::empty(), &RecipeLayer::empty()).ok()
}

/// 2026-10-17: Resolve `var` over [`VALUES`]; the outcomes that differ from `old` must be
/// exactly `changes`.
fn assert_old_reader_holds(
    var: &str,
    old: impl Fn(&dyn Fn(&str) -> Option<String>) -> Out,
    new: impl Fn(&InferenceServerSettings) -> Out,
    changes: &[(&str, Out)],
) {
    let actual: Vec<(&str, Out)> = VALUES
        .iter()
        .filter_map(|&v| {
            let lookup = |name: &str| v.filter(|_| name == var).map(str::to_string);
            let got = settings(var, v).map_or(Out::Err, |s| new(&s));
            (got != old(&lookup)).then(|| (v.unwrap_or("<unset>"), got))
        })
        .collect();
    assert_eq!(
        actual, changes,
        "{var}: the changed values differ from the changes table"
    );
}

const SWITCH_ONLY_ERRORS: [(&str, Out); 5] = [
    ("", Out::Err),
    ("garbage", Out::Err),
    ("25.0", Out::Err),
    ("nan", Out::Err),
    ("-1", Out::Err),
];

fn with_errors(words: &[(&'static str, Out)]) -> Vec<(&'static str, Out)> {
    let mut all: Vec<(&str, Out)> = words.to_vec();
    all.extend(SWITCH_ONLY_ERRORS);
    let order = |w: &str| VALUES.iter().position(|v| v.unwrap_or("<unset>") == w);
    all.sort_by_key(|(w, _)| order(w));
    all
}

#[test]
fn the_indexer_layout_takes_the_switch_grammar() {
    assert_old_reader_holds(
        "METRALE_GLM_DSA_INDEXER_FLAT",
        |var| Out::B(glm_dsa_indexer_paged_from(var)),
        |s| Out::B(PrefixCacheGate::from_settings(&s.cache).glm_indexer_paged),
        &with_errors(&[
            ("true", Out::B(false)),
            ("TRUE", Out::B(false)),
            ("yes", Out::B(false)),
            ("on", Out::B(false)),
            (" 1 ", Out::B(false)),
        ]),
    );
}

#[test]
fn the_validation_override_stays_a_presence_switch() {
    // 2026-10-17: The old reader: `var_os(..).is_some()`, `0` and empty included.
    assert_old_reader_holds(
        "METRALE_GLM53_PREFIX_CACHE_UNPROVEN",
        |var| Out::B(var("METRALE_GLM53_PREFIX_CACHE_UNPROVEN").is_some()),
        |s| Out::B(PrefixCacheGate::from_settings(&s.cache).validation_override),
        &[],
    );
}

#[test]
fn the_declared_gate_is_the_empty_environment() {
    let s = settings("METRALE_UNUSED", None).expect("an empty environment resolves");
    assert_eq!(
        PrefixCacheGate::from_settings(&s.cache),
        PrefixCacheGate::DECLARED
    );
}

#[test]
fn glm_vision_keeps_its_value_verbatim() {
    assert_old_reader_holds(
        "METRALE_GLM_VISION",
        |var| Out::B(glm_vision_enabled_env(var)),
        |s| {
            Out::B(crate::glm_vision_enabled_from(
                s.model.glm_vision.as_deref(),
            ))
        },
        &[],
    );
}

#[test]
fn lora_allow_partial_takes_the_switch_grammar() {
    assert_old_reader_holds(
        "METRALE_LORA_ALLOW_PARTIAL",
        |var| Out::B(allow_partial_targets_from(var)),
        |s| Out::B(s.model.lora_allow_partial),
        &with_errors(&[
            ("yes", Out::B(true)),
            ("on", Out::B(true)),
            (" 1 ", Out::B(true)),
        ]),
    );
}

#[test]
fn gemma4_lm_head_takes_the_switch_grammar() {
    assert_old_reader_holds(
        "METRALE_GEMMA4_LMHEAD_NVFP4",
        |var| Out::B(gemma4_lmhead_nvfp4_from(var)),
        |s| Out::B(s.kernels.gemma4_lmhead_nvfp4),
        &with_errors(&[
            ("true", Out::B(true)),
            ("TRUE", Out::B(true)),
            ("yes", Out::B(true)),
            ("on", Out::B(true)),
            (" 1 ", Out::B(true)),
        ]),
    );
}

#[test]
fn the_softcap_override_is_a_finite_trimmed_float() {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the parser stores the softcap as f32, as the old reader parsed it"
    )]
    let as_f32 = |v: f64| v as f32;
    assert_old_reader_holds(
        "METRALE_SOFTCAP_OVERRIDE",
        |var| f(softcap_override_from(var)),
        |s| f(s.runtime.softcap_override.map(as_f32)),
        &[
            ("true", Out::Err),
            ("TRUE", Out::Err),
            ("yes", Out::Err),
            ("on", Out::Err),
            ("false", Out::Err),
            ("off", Out::Err),
            ("", Out::Err),
            (" 1 ", f(Some(1.0))),
            ("garbage", Out::Err),
            ("nan", Out::Err),
        ],
    );
}
