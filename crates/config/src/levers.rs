// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Every `METRALE_*` environment variable the repository reads, declared once.
//!
//! 2026-10-17: The table is derived from the `metrale-settings` groups (one `lever_group!`
//! file per group); `table_a`/`table_b` stay as the frozen copy `parity_tests` compares it to.
//!
//! `met` refuses to start with an undeclared `METRALE_*` name in its environment ([`check`],
//! called first in `serve_main`). The bench crate's `serve_env` takes the harness set from the
//! table and validates a recipe's `env:` block against it, and `met dump-serve-options`
//! publishes it. `levers_tests` checks the table against the sources: every `"METRALE_*"`
//! literal outside test files and every `getenv("METRALE_*")` in C/CUDA must be declared, each
//! lever's `reader` file must name it, and a `Runtime` lever needs a production reader. The
//! server's `manifest_tests` check the `flag` column against clap.
//!
//! Owner: config.
//! Invariants: none beyond the types.

#[cfg(test)]
macro_rules! lever {
    ($env:literal, $path:literal, $ty:ident, $default:literal, None, $class:ident, $reader:literal, $doc:literal) => {
        $crate::levers::LeverSpec {
            env: $env,
            path: $path,
            ty: $crate::levers::Ty::$ty,
            default: $default,
            flag: None,
            class: $crate::levers::Class::$class,
            reader: $reader,
            doc: $doc,
        }
    };
    ($env:literal, $path:literal, $ty:ident, $default:literal, $flag:literal, $class:ident, $reader:literal, $doc:literal) => {
        $crate::levers::LeverSpec {
            env: $env,
            path: $path,
            ty: $crate::levers::Ty::$ty,
            default: $default,
            flag: Some($flag),
            class: $crate::levers::Class::$class,
            reader: $reader,
            doc: $doc,
        }
    };
}

#[cfg(test)]
mod table_a;
#[cfg(test)]
mod table_b;

pub use metrale_settings::spec::{Class, LeverSpec, PREFIX, Ty};

/// 2026-09-26: Every lever, in name order (`levers_tests` checks the order).
pub fn all() -> impl Iterator<Item = &'static LeverSpec> {
    // 2026-10-17: Declared per group in metrale-settings; sorted once to keep the table order.
    static SORTED: std::sync::OnceLock<Vec<&'static LeverSpec>> = std::sync::OnceLock::new();
    SORTED
        .get_or_init(metrale_settings::groups::by_name)
        .iter()
        .copied()
}

#[must_use]
pub fn lookup(env: &str) -> Option<&'static LeverSpec> {
    all().find(|l| l.env == env)
}

#[must_use]
pub fn is_harness(env: &str) -> bool {
    lookup(env).is_some_and(|l| l.class == Class::Harness)
}

/// 2026-09-26: The `METRALE_*` names among `names` that no lever declares, sorted and
/// de-duplicated.
pub fn undeclared<I, S>(names: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out: Vec<String> = names
        .into_iter()
        .map(|n| n.as_ref().to_string())
        .filter(|n| n.starts_with(PREFIX) && lookup(n).is_none())
        .collect();
    out.sort();
    out.dedup();
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UndeclaredLevers(pub Vec<String>);

impl std::fmt::Display for UndeclaredLevers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "undeclared METRALE_* environment variable{}: {}. Nothing reads {}: it is a \
             typo or a lever that no longer exists, and running on as if it applied would \
             measure something other than what was asked for. Unset {}. (`met \
             dump-serve-options` lists every declared lever.)",
            if self.0.len() == 1 { "" } else { "s" },
            self.0.join(", "),
            if self.0.len() == 1 { "it" } else { "them" },
            if self.0.len() == 1 { "it" } else { "them" },
        )
    }
}

impl std::error::Error for UndeclaredLevers {}

/// 2026-09-26: Refuse an environment carrying an undeclared `METRALE_*` name.
///
/// # Errors
/// [`UndeclaredLevers`] naming every such variable.
pub fn check<I, S>(names: I) -> Result<(), UndeclaredLevers>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let bad = undeclared(names);
    if bad.is_empty() {
        Ok(())
    } else {
        Err(UndeclaredLevers(bad))
    }
}

/// 2026-09-26: `std::env::var(name).ok()`, for a declared lever.
///
/// Debug builds assert that a `METRALE_*` name is declared, so a reader of a new name fails
/// its own tests until the table carries it. Names outside the prefix are not checked.
pub fn var(name: &str) -> Option<String> {
    debug_assert!(declared_or_foreign(name), "{name} is not a declared lever");
    std::env::var(name).ok()
}

/// 2026-09-26: `std::env::var_os(name)`, for a declared lever (see [`var`]).
pub fn var_os(name: &str) -> Option<std::ffi::OsString> {
    debug_assert!(declared_or_foreign(name), "{name} is not a declared lever");
    std::env::var_os(name)
}

fn declared_or_foreign(name: &str) -> bool {
    !name.starts_with(PREFIX) || lookup(name).is_some()
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "levers_parity_tests.rs"]
mod parity_tests;
