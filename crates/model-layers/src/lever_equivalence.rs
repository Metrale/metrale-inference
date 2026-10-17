// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Test support for the move of the model levers onto `metrale-settings`: resolve the
//! settings from a fixed map (`MapEnv`, never the process environment) and check a migrated
//! lever's reading against the reader it replaced.
//!
//! Owner: model-layers (settings migration).
//! Invariants:
//! - [`expected`] is the whole behaviour-change rule: a probe may differ from the old reader only
//!   where the settings rule reads it differently ([`Outcome::Changed`]) or rejects it
//!   ([`Outcome::Rejected`]).
//! - [`check`] fails unless every probe's actual outcome is the expected one.

use metrale_settings::sources::{CliLayer, RecipeLayer};
use metrale_settings::spec::Rule;
use metrale_settings::{InferenceServerSettings, SettingsError};

/// 2026-10-17: The settings resolved from `pairs` alone, CLI and recipe layers empty.
///
/// # Errors
/// The first lever in `pairs` whose value does not parse under its rule.
pub fn settings_from(pairs: &[(&str, &str)]) -> Result<InferenceServerSettings, SettingsError> {
    let env = metrale_io_host_mem::MapEnv::new(pairs.iter().copied());
    metrale_settings::resolve(&env, &CliLayer::empty(), &RecipeLayer::empty())
}

/// 2026-10-17: Every lever at its declared default.
///
/// # Panics
/// When a declared default does not parse (a table bug the settings crate's tests also catch).
#[must_use]
pub fn defaults() -> InferenceServerSettings {
    settings_from(&[]).expect("the declared defaults resolve")
}

/// 2026-10-17: The settings and model levers at their defaults, owned, for a test that hands a
/// loader a [`LoadContext`](crate::load_context::LoadContext).
pub struct DefaultLoad {
    pub settings: InferenceServerSettings,
    pub levers: crate::layers::ops::ModelLevers,
}

impl DefaultLoad {
    #[must_use]
    pub fn at_defaults() -> Self {
        Self {
            settings: defaults(),
            levers: crate::layers::ops::ModelLevers::defaults(),
        }
    }

    #[must_use]
    pub fn lc(&self) -> crate::load_context::LoadContext<'_> {
        crate::load_context::LoadContext {
            settings: &self.settings,
            levers: &self.levers,
        }
    }
}

/// 2026-10-17: The values every migrated lever is probed with; `None` is unset.
pub const PROBES: [Option<&str>; 6] = [
    Some("1"),
    Some("true"),
    Some("0"),
    Some("off"),
    Some("garbage"),
    None,
];

/// 2026-10-17: The reader a lever had before it moved to a settings field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Old {
    /// 2026-10-17: `== "1"`.
    OptIn,
    /// 2026-10-17: `!= "0"`.
    OptOut,
    /// 2026-10-17: `"1"` or `true` in any case.
    Truthy,
    /// 2026-10-17: `"1"` or `"true"` exactly.
    TruthyExact,
    /// 2026-10-17: Set at any value.
    Presence,
    /// 2026-10-17: Kept as text; the call site applies its own predicate to it, as before.
    Verbatim,
    /// 2026-10-17: Parsed as a number; unparseable reads as unset.
    Number,
}

/// 2026-10-17: How the migrated reading of one probe compares with the old reader's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Same,
    Changed,
    /// 2026-10-17: The settings resolve refuses the value (`SettingsError::Malformed`); the old
    /// reader took it silently.
    Rejected,
}

fn switch_value(raw: &str) -> Option<bool> {
    let v = raw.trim_ascii();
    if ["1", "true", "on", "yes"]
        .iter()
        .any(|s| v.eq_ignore_ascii_case(s))
    {
        Some(true)
    } else if ["0", "false", "off", "no"]
        .iter()
        .any(|s| v.eq_ignore_ascii_case(s))
    {
        Some(false)
    } else {
        None
    }
}

fn old_armed(old: Old, probe: Option<&str>) -> bool {
    match old {
        Old::OptIn => probe == Some("1"),
        Old::OptOut => probe != Some("0"),
        Old::Truthy => probe.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
        Old::TruthyExact => matches!(probe, Some("1" | "true")),
        Old::Presence => probe.is_some(),
        Old::Verbatim | Old::Number => unreachable!("{old:?} is not a switch reader"),
    }
}

/// 2026-10-17: The outcome the settings rule of `env` gives `probe` against `old`.
///
/// # Panics
/// When `env` is not a declared lever.
#[must_use]
pub fn expected(old: Old, env: &str, probe: Option<&str>) -> Outcome {
    let spec = metrale_settings::groups::lookup(env).unwrap_or_else(|| panic!("{env} undeclared"));
    let rule = metrale_settings::groups::rule_of(env).expect("a declared lever has a rule");
    let same_if = |same: bool| {
        if same {
            Outcome::Same
        } else {
            Outcome::Changed
        }
    };
    match rule {
        Rule::Text | Rule::Path => Outcome::Same,
        Rule::Presence => same_if(old_armed(old, probe) == probe.is_some()),
        Rule::Switch => match probe {
            None => same_if(old_armed(old, None) == (spec.default == "on")),
            Some(raw) => match switch_value(raw) {
                None => Outcome::Rejected,
                Some(on) => same_if(old_armed(old, probe) == on),
            },
        },
        Rule::Int => match probe {
            Some(raw) if raw.trim_ascii().parse::<i64>().is_err() => Outcome::Rejected,
            _ => Outcome::Same,
        },
        Rule::Float => match probe {
            Some(raw) if !raw.trim_ascii().parse::<f64>().is_ok_and(f64::is_finite) => {
                Outcome::Rejected
            }
            _ => Outcome::Same,
        },
    }
}

/// 2026-10-17: Probe `env` with every [`PROBES`] value through the old reader and the settings
/// path, and assert each outcome is the [`expected`] one. Returns the probes that are not
/// [`Outcome::Same`], for the behaviour-change list.
///
/// # Panics
/// When an outcome differs from the expected one.
pub fn check<T: PartialEq + core::fmt::Debug>(
    env: &'static str,
    old_kind: Old,
    old: impl Fn(Option<&str>) -> T,
    new: impl Fn(&InferenceServerSettings) -> T,
) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    let mut changes = Vec::new();
    for probe in PROBES {
        let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
        let actual = match settings_from(&pairs) {
            Err(SettingsError::Malformed { env: bad, .. }) if bad == env => Outcome::Rejected,
            Err(e) => panic!("{env}={probe:?}: unexpected resolve error {e:?}"),
            Ok(s) => {
                if new(&s) == old(probe) {
                    Outcome::Same
                } else {
                    Outcome::Changed
                }
            }
        };
        assert_eq!(
            actual,
            expected(old_kind, env, probe),
            "{env}={probe:?} ({old_kind:?})"
        );
        if actual != Outcome::Same {
            changes.push((env, probe, actual));
        }
    }
    changes
}
