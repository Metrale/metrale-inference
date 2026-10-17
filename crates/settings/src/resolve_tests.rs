// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The resolver on an in-memory environment: parse rules, precedence, provenance
//! and every error variant. Test levers are chosen from the table at run time, never by name.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use metrale_io_host_mem::MapEnv;

use crate::groups::harness::HarnessSettings;
use crate::groups::{by_name, rule_of};
use crate::provenance::{Provenance, Source};
use crate::resolve::{FromLever, Layer, pick};
use crate::sources::{CliLayer, RecipeLayer, Sources};
use crate::spec::{Class, LeverSpec, PREFIX, Rule};
use crate::{Env, EnvError, SettingsError, resolve};

fn no_env() -> MapEnv {
    MapEnv::new([] as [(&str, &str); 0])
}

fn first(what: &str, pred: impl Fn(&LeverSpec, Rule) -> bool) -> &'static LeverSpec {
    by_name()
        .into_iter()
        .find(|l| pred(l, rule_of(l.env).expect("every lever has a rule")))
        .unwrap_or_else(|| panic!("the table has no {what} lever"))
}

fn runtime(rule: Rule, default_ok: impl Fn(&str) -> bool) -> &'static LeverSpec {
    first(rule.as_str(), |l, r| {
        l.class == Class::Runtime && r == rule && default_ok(l.default)
    })
}

/// 2026-10-17: One `pick` through the given layers; returns the value and its recorded source.
fn pick_in<T: FromLever>(
    l: &'static LeverSpec,
    env: &MapEnv,
    cli: &CliLayer,
    recipe: &RecipeLayer,
) -> Result<(T, Option<Source>), SettingsError> {
    let rule = rule_of(l.env).expect("declared");
    let mut prov = Provenance::default();
    let v = pick::<T, MapEnv>(
        &Sources { env, cli, recipe },
        &mut prov,
        l.env,
        rule,
        l.default,
    )?;
    assert_eq!(prov.len(), 1, "{}: one provenance entry per pick", l.env);
    Ok((v, prov.source_of(l.env)))
}

fn from_env<T: FromLever>(l: &'static LeverSpec, raw: &str) -> Result<T, SettingsError> {
    let env = MapEnv::new([(l.env, raw)]);
    pick_in::<T>(l, &env, &CliLayer::empty(), &RecipeLayer::empty()).map(|(v, _)| v)
}

fn assert_malformed_env<T: FromLever + core::fmt::Debug>(l: &'static LeverSpec, raw: &str) {
    match from_env::<T>(l, raw) {
        Err(SettingsError::Malformed {
            env,
            from: Source::Env,
            value,
            ..
        }) => {
            assert_eq!(env, l.env);
            assert_eq!(value, raw);
        }
        other => panic!(
            "{} = {raw:?}: expected Malformed from env, got {other:?}",
            l.env
        ),
    }
}

#[test]
fn empty_sources_resolve_every_lever_to_its_declared_default() {
    let env = no_env();
    let (cli, recipe) = (CliLayer::empty(), RecipeLayer::empty());
    let s = resolve(&env, &cli, &recipe).expect("every Runtime default parses");
    let runtime_levers = by_name()
        .iter()
        .filter(|l| l.class == Class::Runtime)
        .count();
    assert_eq!(
        s.provenance.len(),
        runtime_levers,
        "one entry per Runtime lever"
    );
    assert!(s.provenance.iter().all(|(_, src)| src == Source::Default));

    let mut prov = Provenance::default();
    HarnessSettings::resolve(
        &Sources {
            env: &env,
            cli: &cli,
            recipe: &recipe,
        },
        &mut prov,
    )
    .expect("every Harness default parses");
    let harness = by_name()
        .iter()
        .filter(|l| l.class == Class::Harness)
        .count();
    assert_eq!(prov.len(), harness);
    assert!(prov.iter().all(|(_, src)| src == Source::Default));
}

#[test]
fn cli_beats_recipe_beats_env_beats_default() {
    let l = runtime(Rule::Int, |d| d.parse::<i64>().is_ok());
    let d: i64 = l.default.parse().expect("numeric default");
    let (e, r, c) = (
        (d + 1).to_string(),
        (d + 2).to_string(),
        (d + 3).to_string(),
    );
    let env = MapEnv::new([(l.env, e.as_str())]);
    let cli = CliLayer::from_pairs([(l.env, c.as_str())]).expect("runtime lever");
    let recipe = RecipeLayer::from_pairs([(l.env, r.as_str())]).expect("runtime lever");
    let (none, no_cli, no_recipe) = (no_env(), CliLayer::empty(), RecipeLayer::empty());

    let ladder: [(&MapEnv, &CliLayer, &RecipeLayer, i64, Source); 4] = [
        (&none, &no_cli, &no_recipe, d, Source::Default),
        (&env, &no_cli, &no_recipe, d + 1, Source::Env),
        (&env, &no_cli, &recipe, d + 2, Source::Recipe),
        (&env, &cli, &recipe, d + 3, Source::Cli),
    ];
    for (env, cli, recipe, want, src) in ladder {
        let (v, got) = pick_in::<i64>(l, env, cli, recipe).expect("parses");
        assert_eq!((v, got), (want, Some(src)), "{}", l.env);
        let s = resolve(env, cli, recipe).expect("resolves");
        assert_eq!(
            s.provenance.source_of(l.env),
            Some(src),
            "{} via resolve",
            l.env
        );
    }

    // 2026-10-17: A higher layer shadows the env entirely: a malformed env value is never read.
    let bad = MapEnv::new([(l.env, "not a number")]);
    let (v, got) = pick_in::<i64>(l, &bad, &cli, &no_recipe).expect("CLI shadows env");
    assert_eq!((v, got), (d + 3, Some(Source::Cli)));
}

#[test]
fn an_unset_option_lever_is_none_and_a_set_one_is_some() {
    let l = runtime(Rule::Int, |d| d == "unset");
    let (cli, recipe) = (CliLayer::empty(), RecipeLayer::empty());
    let (v, src) = pick_in::<Option<i64>>(l, &no_env(), &cli, &recipe).expect("unset");
    assert_eq!((v, src), (None, Some(Source::Default)));
    let env = MapEnv::new([(l.env, " 7 ")]);
    let (v, src) = pick_in::<Option<i64>>(l, &env, &cli, &recipe).expect("set");
    assert_eq!((v, src), (Some(7), Some(Source::Env)));
}

#[test]
fn switch_accepts_every_spelling_and_rejects_anything_else() {
    let l = runtime(Rule::Switch, |d| d == "off" || d == "on");
    let on = ["1", "true", "on", "yes", "TRUE", " Yes ", "On\t", "\n1"];
    let off = ["0", "false", "off", "no", "FALSE", " No ", "oFf\t", "\n0"];
    for raw in on {
        assert_eq!(from_env::<bool>(l, raw), Ok(true), "{raw:?}");
    }
    for raw in off {
        assert_eq!(from_env::<bool>(l, raw), Ok(false), "{raw:?}");
    }
    for raw in ["", " ", "2", "enable", "y", "on1"] {
        assert_malformed_env::<bool>(l, raw);
    }
    let want = l.default == "on";
    let (v, _) = pick_in::<bool>(l, &no_env(), &CliLayer::empty(), &RecipeLayer::empty())
        .expect("default parses");
    assert_eq!(v, want, "{}: unset is the declared default", l.env);
}

#[test]
fn a_garbage_switch_fails_resolve_naming_the_lever() {
    let l = runtime(Rule::Switch, |d| d == "off" || d == "on");
    let env = MapEnv::new([(l.env, "maybe")]);
    let err = resolve(&env, &CliLayer::empty(), &RecipeLayer::empty())
        .expect_err("a malformed switch fails resolution");
    assert!(
        matches!(err, SettingsError::Malformed { env, from: Source::Env, rule: Rule::Switch, .. } if env == l.env),
        "{err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains(l.env) && msg.contains("env") && msg.contains("maybe"),
        "{msg}"
    );
}

#[test]
fn presence_is_on_for_any_set_value_and_off_when_unset() {
    let l = runtime(Rule::Presence, |d| d == "off");
    for raw in ["0", "", "off", "no", "anything"] {
        assert_eq!(from_env::<bool>(l, raw), Ok(true), "{raw:?}");
    }
    let (v, src) = pick_in::<bool>(l, &no_env(), &CliLayer::empty(), &RecipeLayer::empty())
        .expect("default parses");
    assert_eq!((v, src), (false, Some(Source::Default)));
}

#[test]
fn numbers_reject_non_numbers_and_non_finite_floats() {
    let int = runtime(Rule::Int, |d| d == "unset");
    for raw in ["abc", "1.5", "", "0x10"] {
        assert_malformed_env::<Option<i64>>(int, raw);
    }
    assert_eq!(from_env::<Option<i64>>(int, "\t-42 "), Ok(Some(-42)));

    let float = first("unset Float", |l, r| {
        r == Rule::Float && l.default == "unset"
    });
    for raw in ["nan", "inf", "-inf", "NaN", "abc", ""] {
        assert_malformed_env::<Option<f64>>(float, raw);
    }
    assert_eq!(from_env::<Option<f64>>(float, " 1.5 "), Ok(Some(1.5)));
}

#[test]
fn text_and_path_values_are_kept_verbatim() {
    for rule in [Rule::Text, Rule::Path] {
        let l = first(rule.as_str(), |l, r| r == rule && l.default == "unset");
        assert_eq!(
            from_env::<Option<String>>(l, "  a b\t"),
            Ok(Some(String::from("  a b\t")))
        );
        assert_eq!(from_env::<Option<String>>(l, ""), Ok(Some(String::new())));
    }
}

#[test]
fn malformed_display_truncates_a_long_value_on_a_char_boundary() {
    let l = runtime(Rule::Int, |d| d == "unset");
    let long = format!("a{}", "\u{e9}".repeat(100));
    let err = from_env::<Option<i64>>(l, &long).expect_err("not a number");
    let msg = err.to_string();
    assert!(msg.contains(l.env) && msg.contains("..."), "{msg}");
    assert!(!msg.contains(&long), "the full value is echoed: {msg}");
    assert!(msg.len() < long.len(), "{msg}");
}

#[test]
fn layers_refuse_undeclared_non_runtime_and_repeated_names() {
    let undeclared = format!("{PREFIX}NOT_A_DECLARED_LEVER_2026_10_17");
    assert!(crate::groups::lookup(&undeclared).is_none());
    let harness = first("Harness", |l, _| l.class == Class::Harness);
    let build = first("Build", |l, _| l.class == Class::Build);
    let rt = runtime(Rule::Switch, |_| true);

    for layer in [Layer::Cli, Layer::Recipe] {
        assert_eq!(
            build_layer(layer, &[(undeclared.as_str(), "1")]),
            Err(SettingsError::Undeclared {
                from: layer,
                name: undeclared.clone()
            })
        );
        for l in [harness, build] {
            assert_eq!(
                build_layer(layer, &[(l.env, "1")]),
                Err(SettingsError::NotRuntime {
                    from: layer,
                    env: l.env,
                    class: l.class
                })
            );
        }
        assert_eq!(
            build_layer(layer, &[(rt.env, "1"), (rt.env, "0")]),
            Err(SettingsError::Duplicate {
                from: layer,
                env: rt.env
            })
        );
        assert_eq!(build_layer(layer, &[(rt.env, "1")]), Ok(()));
    }
}

fn build_layer(layer: Layer, pairs: &[(&str, &str)]) -> Result<(), SettingsError> {
    let pairs = pairs.iter().copied();
    match layer {
        Layer::Cli => CliLayer::from_pairs(pairs).map(|_| ()),
        Layer::Recipe => RecipeLayer::from_pairs(pairs).map(|_| ()),
    }
}

/// 2026-10-17: An environment whose every variable is set but not UTF-8.
struct NotUnicodeEnv;

impl Env for NotUnicodeEnv {
    fn var(&self, name: &str) -> Result<Option<String>, EnvError> {
        Err(EnvError::NotUnicode {
            name: String::from(name),
        })
    }

    fn names(&self) -> Vec<String> {
        Vec::new()
    }
}

#[test]
fn an_unreadable_env_value_is_a_settings_error() {
    let err = resolve(&NotUnicodeEnv, &CliLayer::empty(), &RecipeLayer::empty())
        .expect_err("the env cannot be read");
    let first_runtime = crate::groups::cache::CacheSettings::LEVERS[0].env;
    assert_eq!(
        err,
        SettingsError::Env(EnvError::NotUnicode {
            name: String::from(first_runtime)
        })
    );
}
