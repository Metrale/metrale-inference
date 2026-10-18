// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The resolution-equivalence harness for the device-side projections: each value
//! of a lever is resolved on an in-memory environment and projected, and the result is compared
//! with the reader the projection replaced (copied into the test as a closure). A value where the
//! two differ must be listed as a deviation, and a listed deviation must really differ, so the
//! deviation lists are the exact behaviour-change record.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use core::fmt::Debug;

use metrale_io_host_mem::MapEnv;

use crate::groups::device::ProjectionError;
use crate::resolve::InferenceServerSettings;
use crate::sources::{CliLayer, RecipeLayer};
use crate::{SettingsError, resolve};

/// 2026-10-17: The values every migrated lever is checked with (`None` = unset).
pub const VALUES: [Option<&str>; 6] = [
    Some("1"),
    Some("true"),
    Some("0"),
    Some("off"),
    Some("garbage"),
    None,
];

/// 2026-10-17: What the projection gives for one value, when it is not what the old reader gave.
#[derive(Debug)]
pub enum Want<T> {
    /// 2026-10-17: Resolution or projection fails (the old reader defaulted silently).
    Err,
    /// 2026-10-17: A different value than the old reader's.
    New(T),
}

/// 2026-10-17: Every runtime lever resolved from `pairs` alone (no CLI, no recipe).
///
/// # Errors
/// The resolver's error for a malformed value.
pub fn resolved(pairs: &[(&str, &str)]) -> Result<InferenceServerSettings, SettingsError> {
    let env = MapEnv::new(pairs.iter().copied());
    resolve(&env, &CliLayer::empty(), &RecipeLayer::empty())
}

/// 2026-10-17: [`check_values`] over [`VALUES`].
pub fn check<T: PartialEq + Debug>(
    env: &str,
    old: impl Fn(Option<&str>) -> T,
    new: impl Fn(&InferenceServerSettings) -> Result<T, ProjectionError>,
    deviations: &[(Option<&str>, Want<T>)],
) {
    check_values(env, &VALUES, old, new, deviations);
}

/// 2026-10-17: For each of `values`, `env` set to it (or unset): the projection `new` must give
/// what `old` gave, except where `deviations` says otherwise.
pub fn check_values<T: PartialEq + Debug>(
    env: &str,
    values: &[Option<&str>],
    old: impl Fn(Option<&str>) -> T,
    new: impl Fn(&InferenceServerSettings) -> Result<T, ProjectionError>,
    deviations: &[(Option<&str>, Want<T>)],
) {
    check_fallible(env, values, |v| Ok(old(v)), new, deviations);
}

/// 2026-10-17: [`check_values`] for an old reader that could reject a value (`Err(())`); a
/// rejection the projection keeps is not a deviation.
pub fn check_fallible<T: PartialEq + Debug>(
    env: &str,
    values: &[Option<&str>],
    old: impl Fn(Option<&str>) -> Result<T, ()>,
    new: impl Fn(&InferenceServerSettings) -> Result<T, ProjectionError>,
    deviations: &[(Option<&str>, Want<T>)],
) {
    for (v, _) in deviations {
        assert!(
            values.contains(v),
            "{env}: deviation for {v:?} names an unchecked value"
        );
    }
    for v in values {
        let pairs: &[(&str, &str)] = match v {
            Some(raw) => &[(env, raw)],
            None => &[],
        };
        let got = resolved(pairs)
            .map_err(|_| ())
            .and_then(|s| new(&s).map_err(|_| ()));
        let was = old(*v);
        match deviations.iter().find(|(d, _)| d == v).map(|(_, w)| w) {
            None => assert_eq!(
                got, was,
                "{env}={v:?}: the projection must equal the old reader"
            ),
            Some(Want::Err) => {
                assert!(
                    was.is_ok(),
                    "{env}={v:?}: the old reader already rejected it"
                );
                assert_eq!(got, Err(()), "{env}={v:?}: must now fail (was {was:?})");
            }
            Some(Want::New(t)) => {
                assert_ne!(
                    Ok(t),
                    was.as_ref(),
                    "{env}={v:?}: a listed deviation must differ"
                );
                assert_eq!(got.as_ref(), Ok(t), "{env}={v:?}: the deliberate new value");
            }
        }
    }
}

/// 2026-10-17: The old `== "1"` reader, which most device switches used.
#[must_use]
pub fn eq_one(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-17: The deviations of an `== "1"` reader moved onto a `Switch` row: `true` turns it
/// on and a garbage value fails instead of reading as off.
#[must_use]
pub fn eq_one_to_switch() -> [(Option<&'static str>, Want<bool>); 2] {
    [
        (Some("true"), Want::New(true)),
        (Some("garbage"), Want::Err),
    ]
}

/// 2026-10-17: A `std::env::var`-shaped lookup over `pairs`, for the device crates' legacy
/// readers (`legacy_env_*_from`), which take one in place of the process environment.
pub fn var_in(
    pairs: &[(&str, &str)],
) -> impl Fn(&str) -> Result<alloc::string::String, std::env::VarError> + use<> {
    let owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = pairs
        .iter()
        .map(|(k, v)| ((*k).into(), (*v).into()))
        .collect();
    move |name| {
        owned
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .ok_or(std::env::VarError::NotPresent)
    }
}

/// 2026-10-17: [`var_in`] with only `env` set to `v` (`None`: nothing set).
pub fn var_one(
    env: &str,
    v: Option<&str>,
) -> impl Fn(&str) -> Result<alloc::string::String, std::env::VarError> + use<> {
    match v {
        Some(raw) => var_in(&[(env, raw)]),
        None => var_in(&[]),
    }
}
