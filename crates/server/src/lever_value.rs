// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: A resolved lever value turned into the type its reader holds: an integer
//! narrowed to the reader's width, or a `Text`-declared lever read with the switch or integer
//! grammar of `metrale_settings`. A value that does not fit stops the load (PCND); nothing
//! here substitutes a default.
//!
//! Owner: server startup (lever settings).
//! Invariants: the grammars are `metrale_settings`' own (`FromLever` under `Rule::Switch` and
//! `Rule::Int`), so a `Text` lever spells its values as a declared switch or integer does.

use metrale_settings::resolve::FromLever;
use metrale_settings::spec::Rule;

/// 2026-10-17: A resolved integer lever narrowed to the width its reader holds.
///
/// # Errors
/// When `value` does not fit `T` (a negative count, say): the lever is malformed for this
/// reader, which fails the load instead of substituting a default.
pub(crate) fn narrow<T: TryFrom<i64>>(env: &str, value: i64) -> anyhow::Result<T> {
    T::try_from(value).map_err(|_| {
        anyhow::anyhow!(
            "{env}={value} does not fit a {}",
            std::any::type_name::<T>()
        )
    })
}

/// 2026-10-17: [`narrow`] for an optional lever; `None` stays `None`.
///
/// # Errors
/// As [`narrow`].
pub(crate) fn narrow_opt<T: TryFrom<i64>>(
    env: &str,
    value: Option<i64>,
) -> anyhow::Result<Option<T>> {
    value.map(|v| narrow(env, v)).transpose()
}

/// 2026-10-17: A `Text`-declared lever that is a switch: `None` when unset.
///
/// # Errors
/// A set value that is not `1|true|on|yes` or `0|false|off|no` (trimmed, any case).
pub(crate) fn text_switch(env: &str, value: Option<&str>) -> anyhow::Result<Option<bool>> {
    value
        .map(|v| {
            <bool as FromLever>::parse(Rule::Switch, v)
                .map_err(|_| anyhow::anyhow!("{env}={v:?} is not a switch (on/off, 1/0)"))
        })
        .transpose()
}

/// 2026-10-17: A `Text`-declared lever that is a count: `None` when unset.
///
/// # Errors
/// A set value that is not an integer, or does not fit `T` ([`narrow`]).
pub(crate) fn text_int<T: TryFrom<i64>>(
    env: &str,
    value: Option<&str>,
) -> anyhow::Result<Option<T>> {
    value
        .map(|v| {
            let n = <i64 as FromLever>::parse(Rule::Int, v)
                .map_err(|_| anyhow::anyhow!("{env}={v:?} is not an integer"))?;
            narrow(env, n)
        })
        .transpose()
}
