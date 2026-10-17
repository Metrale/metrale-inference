// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Parsing a lever value and resolving the runtime settings.
//!
//! Owner: settings.
//! Invariants: only a set value can be `Malformed`; only a declared default can be
//! `BadDefault`; every `pick` records exactly one provenance entry.

use alloc::string::String;
use core::fmt;

use crate::Env;
use crate::groups::cache::CacheSettings;
use crate::groups::kernels::KernelsSettings;
use crate::groups::model::ModelSettings;
use crate::groups::runtime::RuntimeSettings;
use crate::groups::sampling::SamplingSettings;
use crate::groups::scheduler::SchedulerSettings;
use crate::groups::server::ServerSettings;
use crate::groups::speculative::SpeculativeSettings;
use crate::groups::storage::StorageSettings;
use crate::groups::telemetry::TelemetrySettings;
use crate::provenance::{Provenance, Source};
use crate::sources::{CliLayer, RecipeLayer, Sources};
use crate::spec::{Class, Rule};
use metrale_error::Error;

/// 2026-10-17: A raw value or declared default did not parse; `pick` names the lever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseFail;

/// 2026-10-17: Defaults that mean "no value": the field is an `Option` and stays `None`.
const SENTINELS: [&str; 3] = ["unset", "built-in", "target-declared"];

pub trait FromLever: Sized {
    /// 2026-10-17: `raw` is a set value (CLI, recipe or env).
    ///
    /// # Errors
    /// `ParseFail` when `raw` is not a valid value under `rule`.
    fn parse(rule: Rule, raw: &str) -> Result<Self, ParseFail>;

    /// 2026-10-17: The value of the declared `default`.
    ///
    /// # Errors
    /// `ParseFail` when the default does not parse under `rule` for this type.
    fn absent(rule: Rule, default: &'static str) -> Result<Self, ParseFail>;
}

fn switch(raw: &str) -> Result<bool, ParseFail> {
    let v = raw.trim_ascii();
    if ["1", "true", "on", "yes"]
        .iter()
        .any(|s| v.eq_ignore_ascii_case(s))
    {
        Ok(true)
    } else if ["0", "false", "off", "no"]
        .iter()
        .any(|s| v.eq_ignore_ascii_case(s))
    {
        Ok(false)
    } else {
        Err(ParseFail)
    }
}

impl FromLever for bool {
    fn parse(rule: Rule, raw: &str) -> Result<Self, ParseFail> {
        match rule {
            Rule::Switch => switch(raw),
            Rule::Presence => Ok(true),
            Rule::Int | Rule::Float | Rule::Text | Rule::Path => Err(ParseFail),
        }
    }

    fn absent(rule: Rule, default: &'static str) -> Result<Self, ParseFail> {
        match (rule, default) {
            (Rule::Switch, d) => switch(d),
            (Rule::Presence, "off") => Ok(false),
            _ => Err(ParseFail),
        }
    }
}

impl FromLever for i64 {
    fn parse(rule: Rule, raw: &str) -> Result<Self, ParseFail> {
        match rule {
            Rule::Int => raw.trim_ascii().parse::<i64>().map_err(|_| ParseFail),
            Rule::Switch | Rule::Presence | Rule::Float | Rule::Text | Rule::Path => Err(ParseFail),
        }
    }

    fn absent(rule: Rule, default: &'static str) -> Result<Self, ParseFail> {
        Self::parse(rule, default)
    }
}

impl FromLever for f64 {
    fn parse(rule: Rule, raw: &str) -> Result<Self, ParseFail> {
        match rule {
            Rule::Float => match raw.trim_ascii().parse::<f64>() {
                Ok(v) if v.is_finite() => Ok(v),
                _ => Err(ParseFail),
            },
            Rule::Switch | Rule::Presence | Rule::Int | Rule::Text | Rule::Path => Err(ParseFail),
        }
    }

    fn absent(rule: Rule, default: &'static str) -> Result<Self, ParseFail> {
        Self::parse(rule, default)
    }
}

impl FromLever for String {
    fn parse(rule: Rule, raw: &str) -> Result<Self, ParseFail> {
        match rule {
            Rule::Text | Rule::Path => Ok(Self::from(raw)),
            Rule::Switch | Rule::Presence | Rule::Int | Rule::Float => Err(ParseFail),
        }
    }

    fn absent(rule: Rule, default: &'static str) -> Result<Self, ParseFail> {
        if SENTINELS.contains(&default) {
            return Err(ParseFail);
        }
        Self::parse(rule, default)
    }
}

impl<T: FromLever> FromLever for Option<T> {
    fn parse(rule: Rule, raw: &str) -> Result<Self, ParseFail> {
        T::parse(rule, raw).map(Some)
    }

    fn absent(rule: Rule, default: &'static str) -> Result<Self, ParseFail> {
        if SENTINELS.contains(&default) {
            Ok(None)
        } else {
            T::absent(rule, default).map(Some)
        }
    }
}

/// 2026-10-17: One lever's value: CLI, then recipe, then env, then the declared default.
///
/// # Errors
/// `Malformed` for a set value that does not parse, `BadDefault` for a default that does not,
/// `Env` when the environment cannot be read.
pub fn pick<T: FromLever, E: Env>(
    src: &Sources<'_, E>,
    prov: &mut Provenance,
    env: &'static str,
    rule: Rule,
    default: &'static str,
) -> Result<T, SettingsError> {
    let from_env;
    let (from, raw) =
        if let Some(v) = src.cli.get(env) {
            (Source::Cli, v)
        } else if let Some(v) = src.recipe.get(env) {
            (Source::Recipe, v)
        } else {
            from_env = src.env.var(env)?;
            match from_env.as_deref() {
                Some(v) => (Source::Env, v),
                None => {
                    prov.record(env, Source::Default);
                    return T::absent(rule, default)
                        .map_err(|ParseFail| SettingsError::BadDefault { env, default, rule });
                }
            }
        };
    prov.record(env, from);
    T::parse(rule, raw).map_err(|ParseFail| SettingsError::Malformed {
        env,
        from,
        value: String::from(raw),
        rule,
    })
}

/// 2026-10-17: Every `Runtime` group, resolved, with where each value came from.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceServerSettings {
    pub cache: CacheSettings,
    pub kernels: KernelsSettings,
    pub model: ModelSettings,
    pub runtime: RuntimeSettings,
    pub sampling: SamplingSettings,
    pub scheduler: SchedulerSettings,
    pub server: ServerSettings,
    pub speculative: SpeculativeSettings,
    pub storage: StorageSettings,
    pub telemetry: TelemetrySettings,
    pub provenance: Provenance,
}

/// 2026-10-17: Precedence CLI > recipe > env > declared default. Runtime levers only;
/// HarnessSettings::resolve is the harness entry point.
///
/// # Errors
/// The first lever that fails, in field order (see [`SettingsError`]).
pub fn resolve(
    env: &impl Env,
    cli: &CliLayer,
    recipe: &RecipeLayer,
) -> Result<InferenceServerSettings, SettingsError> {
    let src = Sources { env, cli, recipe };
    let mut prov = Provenance::default();
    Ok(InferenceServerSettings {
        cache: CacheSettings::resolve(&src, &mut prov)?,
        kernels: KernelsSettings::resolve(&src, &mut prov)?,
        model: ModelSettings::resolve(&src, &mut prov)?,
        runtime: RuntimeSettings::resolve(&src, &mut prov)?,
        sampling: SamplingSettings::resolve(&src, &mut prov)?,
        scheduler: SchedulerSettings::resolve(&src, &mut prov)?,
        server: ServerSettings::resolve(&src, &mut prov)?,
        speculative: SpeculativeSettings::resolve(&src, &mut prov)?,
        storage: StorageSettings::resolve(&src, &mut prov)?,
        telemetry: TelemetrySettings::resolve(&src, &mut prov)?,
        provenance: prov,
    })
}

/// 2026-10-17: Which caller-built layer rejected a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Cli,
    Recipe,
}

impl Layer {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Recipe => "recipe",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SettingsError {
    Malformed {
        env: &'static str,
        from: Source,
        value: String,
        rule: Rule,
    },
    BadDefault {
        env: &'static str,
        default: &'static str,
        rule: Rule,
    },
    Undeclared {
        from: Layer,
        name: String,
    },
    NotRuntime {
        from: Layer,
        env: &'static str,
        class: Class,
    },
    Duplicate {
        from: Layer,
        env: &'static str,
    },
    Env(Error),
}

/// 2026-10-17: Values may be paths; a long one is shown by its first 64 bytes only.
const SHOWN_VALUE_BYTES: usize = 64;

fn shown(value: &str) -> (&str, &'static str) {
    if value.len() <= SHOWN_VALUE_BYTES {
        return (value, "");
    }
    let mut end = SHOWN_VALUE_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (&value[..end], "...")
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed {
                env,
                from,
                value,
                rule,
            } => {
                let (head, ellipsis) = shown(value);
                write!(
                    f,
                    "{env} (from {}): {head:?}{ellipsis} is not a valid {} value",
                    from.as_str(),
                    rule.as_str()
                )
            }
            Self::BadDefault { env, default, rule } => write!(
                f,
                "{env} (from default): declared default {default:?} is not a valid {} value",
                rule.as_str()
            ),
            Self::Undeclared { from, name } => {
                write!(f, "{name} (from {}): not a declared lever", from.as_str())
            }
            Self::NotRuntime { from, env, class } => write!(
                f,
                "{env} (from {}): a {} lever; only runtime levers may be set here",
                from.as_str(),
                class.as_str()
            ),
            Self::Duplicate { from, env } => {
                write!(f, "{env} (from {}): set more than once", from.as_str())
            }
            Self::Env(e) => write!(f, "{e} (from env)"),
        }
    }
}

impl core::error::Error for SettingsError {}

impl From<Error> for SettingsError {
    fn from(e: Error) -> Self {
        Self::Env(e)
    }
}
