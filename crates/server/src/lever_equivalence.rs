// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The harness behind every "old reader vs resolved settings" test. A migrated
//! lever's old reader is frozen as an [`Old`] rule; the test resolves the lever through
//! `metrale_settings` over each of [`VALUES`] and compares. Every difference must appear in
//! the caller's `changes` table, the record of the behaviour the migration changed on
//! purpose; an unlisted difference, or a listed one that no longer differs, fails.
//!
//! Owner: server (lever migration tests).
//! Invariants: a `changes` table lists exactly the (rule class, value) pairs whose outcome
//! changed, and a class's changes hold for every lever in it.

use std::borrow::Cow;

use metrale_io_host_mem::MapEnv;
use metrale_settings::InferenceServerSettings;
use metrale_settings::sources::{CliLayer, RecipeLayer};

/// 2026-10-17: One lever value: a switch, a number (as printed), or a load-stopping error.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Out {
    B(bool),
    N(Cow<'static, str>),
    Err,
}

/// 2026-10-17: The old readers, by name, with their shipped defaults.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Old {
    /// 2026-10-17: `== "1"`.
    OptIn,
    /// 2026-10-17: `!= "0"`.
    OnUnlessZero,
    /// 2026-10-17: `!= "1"`.
    OnUnless,
    /// 2026-10-17: Any value, `0` included.
    Present,
    /// 2026-10-17: `!present`: the kill switches spelled `NO_*`.
    Absent,
    /// 2026-10-17: `1`, or `true` in any ASCII case; untrimmed.
    OptInWord,
    /// 2026-10-17: `1`, or `true` after a Unicode lowercase; untrimmed.
    OptInLowercase,
    /// 2026-10-17: `helpers::parse_flag_default_on`: trimmed, off on `0`/`false` (any case).
    DefaultOn,
    /// 2026-10-17: `METRALE_BEAM_CODISPATCH`: off on `0`/`false` (any case); untrimmed.
    Beam,
    /// 2026-10-17: The former `helpers::parse_disable_watchdogs`: trimmed, on for `1`/`true`
    /// (any case).
    DisableWatchdogs,
    /// 2026-10-17: `helpers::parse_forced_token_fastpath`: trimmed, off for `1`/`true` (any case).
    ForcedToken,
    /// 2026-10-17: `num::<f32>`: the parse, or the default when unparsable.
    NumF32(f32),
    /// 2026-10-17: `num::<u32|u64|usize>`, widened: the parse, or the default when unparsable.
    NumUint(u64),
    /// 2026-10-17: `mtp_dcut::resolve_dcut_ratio(None, _)`: `f32` parse or 0.75, then snapped
    /// by the given `mtp_dcut::snap_ratio` (the scheduler's module is private to it).
    DcutRatio(fn(f32) -> f32),
    /// 2026-10-17: Exactly `1` or `true`; untrimmed.
    OptInOneOrTrue,
    /// 2026-10-17: `u32`/`u64` parse after a trim, or the default when unparsable.
    UintTrimmed(u64),
    /// 2026-10-17: Off on `0`, `false`, `off` or `no` (trimmed, any case); on otherwise,
    /// unset included.
    OnUnlessOffWord,
    /// 2026-10-17: Trimmed; blank and `0` are unset (shown as 0); a value that is not an
    /// unsigned integer stops the load.
    UintBlankZeroUnset,
}

impl Old {
    pub(crate) fn eval(self, v: Option<&str>) -> Out {
        let word = |v: &str, w: &str| v.eq_ignore_ascii_case(w);
        match self {
            Self::OptIn => Out::B(v == Some("1")),
            Self::OnUnlessZero => Out::B(v != Some("0")),
            Self::OnUnless => Out::B(v != Some("1")),
            Self::Present => Out::B(v.is_some()),
            Self::Absent => Out::B(v.is_none()),
            Self::OptInWord => Out::B(v.is_some_and(|v| v == "1" || word(v, "true"))),
            Self::OptInLowercase => {
                Out::B(v.is_some_and(|v| v == "1" || v.to_lowercase() == "true"))
            }
            Self::DefaultOn => Out::B(
                v.map(str::trim)
                    .is_none_or(|v| !(v == "0" || word(v, "false"))),
            ),
            Self::Beam => Out::B(v.is_none_or(|v| v != "0" && !word(v, "false"))),
            Self::DisableWatchdogs => Out::B(
                v.map(str::trim)
                    .is_some_and(|v| v == "1" || word(v, "true")),
            ),
            Self::ForcedToken => Out::B(
                v.map(str::trim)
                    .is_none_or(|v| !(v == "1" || word(v, "true"))),
            ),
            Self::NumF32(d) => num(v.and_then(|v| v.parse::<f32>().ok()).unwrap_or(d)),
            Self::NumUint(d) => num(v.and_then(|v| v.parse::<u64>().ok()).unwrap_or(d)),
            Self::DcutRatio(snap) => {
                num(snap(v.and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.75)))
            }
            Self::OptInOneOrTrue => Out::B(matches!(v, Some("1" | "true"))),
            Self::UintTrimmed(d) => num(v.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(d)),
            Self::OnUnlessOffWord => Out::B(!v.is_some_and(|v| {
                let v = v.trim();
                ["0", "false", "off", "no"].iter().any(|w| word(v, w))
            })),
            Self::UintBlankZeroUnset => match v.map(str::trim) {
                None | Some("" | "0") => num(0),
                Some(t) => t.parse::<u64>().map_or(Out::Err, num),
            },
        }
    }

    /// 2026-10-17: Rules whose changes are identical share a class.
    pub(crate) fn class(self) -> &'static str {
        match self {
            Self::NumUint(_) => "uint",
            Self::NumF32(_) => "f32",
            Self::OptIn => "opt_in",
            Self::OnUnlessZero => "on_unless_zero",
            Self::OnUnless => "on_unless",
            Self::Present => "present",
            Self::Absent => "absent",
            Self::OptInWord => "opt_in_word",
            Self::OptInLowercase => "opt_in_lowercase",
            Self::DefaultOn => "default_on",
            Self::Beam => "beam",
            Self::DisableWatchdogs => "disable_watchdogs",
            Self::ForcedToken => "forced_token",
            Self::DcutRatio(_) => "dcut_ratio",
            Self::OptInOneOrTrue => "opt_in_one_or_true",
            Self::UintTrimmed(_) => "uint_trimmed",
            Self::OnUnlessOffWord => "on_unless_off_word",
            Self::UintBlankZeroUnset => "uint_blank_zero_unset",
        }
    }
}

pub(crate) const VALUES: [Option<&str>; 14] = [
    None,
    Some("1"),
    Some("true"),
    Some("TRUE"),
    Some("yes"),
    Some("0"),
    Some("false"),
    Some("off"),
    Some("no"),
    Some(""),
    Some(" 1 "),
    Some("garbage"),
    Some("7"),
    Some("-1"),
];

/// 2026-10-17: One lever under test: its variable, the rule its old reader applied, and how to
/// read its new value from what `resolve` returns.
pub(crate) type Row<T> = (&'static str, Old, fn(&T) -> Out);

/// 2026-10-17: One lever's (value, new outcome) pairs that differ from its old rule.
type LeverDiffs<'a> = (&'a str, Old, Vec<(&'a str, Out)>);

/// 2026-10-17: Resolve each of `rows` over [`VALUES`] with `resolve` and compare against its
/// [`Old`] rule; the differences must be exactly `changes`.
pub(crate) fn assert_old_readers_hold<T>(
    rows: &[Row<T>],
    resolve: impl Fn(&[(&str, &str)]) -> anyhow::Result<T>,
    changes: &[(&str, &str, Out)],
) {
    let mut actual: Vec<(&str, &str, Out)> = Vec::new();
    let mut per_lever: Vec<LeverDiffs<'_>> = Vec::new();
    for &(var, old, get) in rows {
        let mut diffs = Vec::new();
        for value in VALUES {
            let env: Vec<(&str, &str)> = value.map(|v| (var, v)).into_iter().collect();
            let new = resolve(&env).map_or(Out::Err, |l| get(&l));
            let was = old.eval(value);
            if new != was {
                diffs.push((value.unwrap_or("<unset>"), new));
            }
        }
        for (value, new) in &diffs {
            if !actual
                .iter()
                .any(|(c, v, o)| *c == old.class() && v == value && o == new)
            {
                actual.push((old.class(), value, new.clone()));
            }
        }
        per_lever.push((var, old, diffs));
    }
    let report: String = actual
        .iter()
        .map(|(c, v, o)| format!("    ({c:?}, {v:?}, Out::{o:?}),\n"))
        .collect();
    assert_eq!(
        actual.len(),
        changes.len(),
        "the changed (rule, value) pairs differ from the changes table; actual:\n{report}"
    );
    for change in changes {
        assert!(
            actual.contains(change),
            "{change:?} is recorded but no longer differs; actual:\n{report}"
        );
    }
    // 2026-10-17: A class's changes hold for every lever in it, so one row per class is
    // the whole record.
    for (var, old, mut diffs) in per_lever {
        let mut expected: Vec<(&str, Out)> = changes
            .iter()
            .filter(|(c, _, _)| *c == old.class())
            .map(|(_, v, o)| (*v, o.clone()))
            .collect();
        diffs.sort_by_key(|(v, _)| *v);
        expected.sort_by_key(|(v, _)| *v);
        assert_eq!(diffs, expected, "{var}");
    }
}

pub(crate) fn num(v: impl ToString) -> Out {
    Out::N(Cow::Owned(v.to_string()))
}

/// 2026-10-17: The settings one load resolves from `env` alone, with no flag and no recipe.
///
/// # Errors
/// A value in `env` that does not parse as its lever's type.
pub(crate) fn settings_from(
    env: &[(&str, &str)],
) -> Result<InferenceServerSettings, metrale_settings::SettingsError> {
    metrale_settings::resolve(
        &MapEnv::new(env.iter().copied()),
        &CliLayer::empty(),
        &RecipeLayer::empty(),
    )
}

/// 2026-10-17: Every lever at its declared default.
pub(crate) fn default_settings() -> InferenceServerSettings {
    settings_from(&[]).expect("the declared defaults resolve")
}
