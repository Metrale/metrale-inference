// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the lever contract: what is a lever, what a
//! declaration means, and what the harness may inherit (nothing it did not
//! declare).
//!
//! Owner: serve identity (serve environment).
//! Invariants: none beyond the types.

use super::*;

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn a_lever_is_any_metrale_name_that_is_not_a_harness_variable() {
    assert!(is_lever("METRALE_FP8_ROWWISE"));
    assert!(is_lever("METRALE_MTP_K_LADDER"));
    assert!(is_lever("METRALE_PREFILL_CODISPATCH"));
    // 2026-09-26: A name the lever table does not declare is still a lever.
    assert!(is_lever("METRALE_SOMETHING_NEW"));
    let harness: Vec<&str> = metrale_settings::groups::by_name()
        .into_iter()
        .filter(|l| l.class == Class::Harness)
        .map(|l| l.env)
        .collect();
    assert!(harness.contains(&"METRALE_HOME"), "{harness:?}");
    for name in harness {
        assert!(!is_lever(name), "{name} is the box's, not a lever");
    }
    // 2026-09-26: A name outside the prefix is not a lever.
    assert!(!is_lever("METRALECTL_CONFIG_DIR"));
    assert!(!is_lever("PATH"));
    assert!(!is_lever("CUDARC_CUDA_VERSION"));
}

#[test]
fn levers_filters_an_environment_and_keeps_only_what_the_server_reads() {
    let env = [
        ("PATH", "/usr/bin"),
        ("METRALE_HOME", "/workspace/.metrale"),
        ("METRALE_FP8_ROWWISE", "1"),
        ("CUDARC_CUDA_VERSION", "13000"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
    ];
    assert_eq!(
        levers(env),
        map(&[
            ("METRALE_FP8_ROWWISE", "1"),
            ("METRALE_MTP_DCUT_RATIO", "1.0")
        ])
    );
}

/// 2026-09-26: The fingerprint is a function of the set: order-free,
/// value-sensitive, and it tells `A=1B=2` from `A=1 B=2`.
#[test]
fn the_fingerprint_separates_sets_that_differ_and_nothing_else() {
    let dense = map(&[
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
    ]);
    let same = map(&[
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_FP8_ROWWISE", "1"),
    ]);
    assert_eq!(fingerprint(&dense), fingerprint(&same));
    assert_eq!(fingerprint(&dense).len(), 64);
    let mut off = dense.clone();
    off.insert("METRALE_FP8_ROWWISE".into(), "0".into());
    assert_ne!(fingerprint(&dense), fingerprint(&off), "a value moved");
    let mut fewer = dense.clone();
    fewer.remove("METRALE_MTP_K_LADDER");
    assert_ne!(fingerprint(&dense), fingerprint(&fewer), "a lever dropped");
    assert_ne!(
        fingerprint(&BTreeMap::new()),
        fingerprint(&map(&[("METRALE_X", "")])),
        "an empty value is not an absent lever"
    );
    assert_ne!(
        fingerprint(&map(&[("METRALE_A", "1B=2")])),
        fingerprint(&map(&[("METRALE_A", "1"), ("B", "2")])),
    );
}

#[test]
fn a_declaration_is_validated() {
    let ok = declared(
        "recipe q/x",
        &map(&[
            ("METRALE_FP8_ROWWISE", "1"),
            ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ]),
    )
    .unwrap();
    assert_eq!(
        ok,
        map(&[
            ("METRALE_FP8_ROWWISE", "1"),
            ("METRALE_MTP_DCUT_RATIO", "1.0")
        ])
    );

    for (bad, why, code) in [
        (
            ("CUDA_VISIBLE_DEVICES", "0"),
            "not a METRALE_* serve lever",
            ErrorCode::ServeEnvNotALever,
        ),
        (
            ("METRALE_HOME", "/tmp/x"),
            "harness variable",
            ErrorCode::ServeEnvHarness,
        ),
        (
            ("METRALE_FP8_ROWWISE", "  "),
            "empty value",
            ErrorCode::ServeEnvEmpty,
        ),
        // 2026-09-26: A name the lever table does not declare.
        (
            ("METRALE_FOO", "1"),
            "not a declared lever",
            ErrorCode::ServeEnvUndeclared,
        ),
        (
            ("METRALE_SKIP_BUILD", "1"),
            "harness variable",
            ErrorCode::ServeEnvHarness,
        ),
        (
            ("METRALE_BENCH_ITERS", "3"),
            "never reads (it is a dev variable)",
            ErrorCode::ServeEnvNotRead,
        ),
    ] {
        let err = declared("recipe q/x", &map(&[bad])).unwrap_err();
        assert_eq!(err.code(), code, "{bad:?}: {err}");
        let e = err.to_string();
        assert!(e.contains(why), "{bad:?}: {e}");
        assert!(e.contains(bad.0), "names the key: {e}");
    }
}

#[test]
fn the_gate_pin_wins_over_the_recipe_on_a_clash_and_adds_otherwise() {
    let merged = merge_declared(
        map(&[("METRALE_A", "recipe"), ("METRALE_B", "recipe")]),
        map(&[("METRALE_B", "pin"), ("METRALE_C", "pin")]),
    );
    assert_eq!(
        merged,
        map(&[
            ("METRALE_A", "recipe"),
            ("METRALE_B", "pin"),
            ("METRALE_C", "pin")
        ])
    );
}

/// 2026-09-26: The harness carries three levers and the recipe declares
/// none: all three are refused by name.
#[test]
fn an_undeclared_lever_in_the_harness_is_refused_by_name() {
    let node = map(&[
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
    ]);
    let e = reconcile(
        "recipe qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl",
        &BTreeMap::new(),
        &node,
    )
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::ServeEnvInherited, "{e}");
    let e = e.to_string();
    assert!(e.contains("3 serve lever(s)"), "{e}");
    assert!(e.contains("METRALE_FP8_ROWWISE=1"), "{e}");
    assert!(
        e.contains("METRALE_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1"),
        "{e}"
    );
    assert!(
        e.contains("recipe qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl"),
        "names the recipe: {e}"
    );
    assert!(e.contains("bench.yaml"), "names the remedy: {e}");
    // 2026-09-26: One stray lever beside a fully declared set is still
    // refused.
    let declared = map(&[("METRALE_FP8_ROWWISE", "1")]);
    let mut present = declared.clone();
    present.insert("METRALE_PREFILL_CODISPATCH".into(), "1".into());
    let e = reconcile("recipe r", &declared, &present)
        .unwrap_err()
        .to_string();
    assert!(e.contains("1 serve lever(s)"), "{e}");
    // 2026-09-26: Assert the list of refused levers, not the whole message:
    // its fixed prose names METRALE_FP8_ROWWISE=1 as an example whatever was
    // refused.
    let blamed = e
        .split("does not declare: ")
        .nth(1)
        .and_then(|rest| rest.split(". A lever").next())
        .unwrap_or_else(|| panic!("the refusal names what it refuses: {e}"));
    assert_eq!(
        blamed, "METRALE_PREFILL_CODISPATCH=1",
        "only the undeclared lever is blamed, never the declared one: {e}"
    );
}

/// 2026-09-26: A declared lever is applied: absent from the harness it is
/// what the child must be given; present at the declared value there is
/// nothing to give; present at another value it is refused.
#[test]
fn a_declared_lever_is_applied_and_a_contradicted_one_is_refused() {
    let declared = map(&[
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
    ]);
    let fresh = reconcile("recipe r", &declared, &BTreeMap::new()).unwrap();
    assert_eq!(
        fresh.env, declared,
        "the server runs under the whole declaration"
    );
    assert_eq!(fresh.missing, declared, "and none of it was already there");

    let half = reconcile("recipe r", &declared, &map(&[("METRALE_FP8_ROWWISE", "1")])).unwrap();
    assert_eq!(half.env, declared);
    assert_eq!(half.missing, map(&[("METRALE_MTP_DCUT_RATIO", "1.0")]));

    let all = reconcile("recipe r", &declared, &declared).unwrap();
    assert!(all.missing.is_empty(), "nothing to apply, nothing refused");

    let e = reconcile(
        "recipe r",
        &declared,
        &map(&[
            ("METRALE_FP8_ROWWISE", "0"),
            ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ]),
    )
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::ServeEnvContradicted, "{e}");
    let e = e.to_string();
    assert!(e.contains("contradicts recipe r"), "{e}");
    assert!(
        e.contains("METRALE_FP8_ROWWISE: harness \"0\", recipe r declares \"1\""),
        "{e}"
    );
    assert!(
        !e.contains("METRALE_MTP_DCUT_RATIO"),
        "the agreeing one is not blamed: {e}"
    );
}

/// 2026-09-26: No levers on either side: no refusal, nothing to apply, the
/// empty digest.
#[test]
fn no_levers_anywhere_reconciles_to_nothing() {
    let r = reconcile("recipe r", &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert!(r.env.is_empty() && r.missing.is_empty());
    assert_eq!(fingerprint(&r.env), fingerprint(&BTreeMap::new()));
}

/// 2026-10-17: Moving the refusals onto the registry kept every message byte-identical to the
/// text `EnvError` printed (frozen here from the pre-move build).
#[test]
fn every_refusal_renders_the_text_it_printed_before_the_registry_move() {
    for ((name, value), want) in [
        (
            ("CUDA_VISIBLE_DEVICES", "0"),
            "recipe r declares env CUDA_VISIBLE_DEVICES, which is not a METRALE_* serve lever. The gate applies exactly the levers a recipe declares and nothing else, so a variable the server does not read cannot be declared here.",
        ),
        (
            ("METRALE_HOME", "/tmp/x"),
            "recipe r declares env METRALE_HOME, which places, logs or builds rather than measuring (METRALE_HOME is a harness variable, not a serve lever) — it is the box's to set, never the recipe's.",
        ),
        (
            ("METRALE_BENCH_ITERS", "3"),
            "recipe r declares env METRALE_BENCH_ITERS, which the server never reads (it is a dev variable), so declaring it here would change nothing that is measured.",
        ),
        (
            ("METRALE_FOO", "1"),
            "recipe r declares env METRALE_FOO, which is not a declared lever: nothing reads it, so it is a typo or a lever that no longer exists, and `met serve` refuses to start with it set. `met dump-serve-options` lists every declared lever.",
        ),
        (
            ("METRALE_FP8_ROWWISE", "  "),
            "recipe r declares env METRALE_FP8_ROWWISE with an empty value. An exported-but-empty variable is how a shell spells \"unset\": declare a value or drop the key.",
        ),
    ] {
        let got = declared("recipe r", &map(&[(name, value)])).unwrap_err();
        assert_eq!(got.to_string(), want, "{name}");
    }
    let present = map(&[("METRALE_A", "1"), ("METRALE_B", "2")]);
    let got = reconcile("recipe r", &BTreeMap::new(), &present).unwrap_err();
    assert_eq!(
        got.to_string(),
        "the harness environment carries 2 serve lever(s) that recipe r does not declare: METRALE_A=1, METRALE_B=2. A lever the recipe did not ask for changes what the gate measures (METRALE_FP8_ROWWISE=1 builds FP8 twin weights that a 0.70-util serve cannot boot with — #1242), so it is refused rather than inherited. Unset it — on a node agent, remove it from bench.yaml `env:`, which reaches EVERY gate on the box — or declare it in the recipe's `env:` (or the gate's [benchmarks.serve_env]) if the gate is measured under it."
    );
    let got = reconcile(
        "recipe r",
        &map(&[("METRALE_A", "1")]),
        &map(&[("METRALE_A", "0")]),
    )
    .unwrap_err();
    assert_eq!(
        got.to_string(),
        "the harness environment contradicts recipe r on 1 serve lever(s): METRALE_A: harness \"0\", recipe r declares \"1\". Neither value wins silently: unset the variable so the declaration applies, or change the declaration."
    );
}
