// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The published lever row (`LeverSpec`, moved verbatim from `metrale-config`)
//! and the parse rule the resolver applies to it.
//!
//! Owner: settings.
//! Invariants: `Rule::ty` maps `Presence` to `Ty::Switch`, so adding `Rule` leaves the
//! published table unchanged.

/// 2026-09-26: Who reads a lever. `levers_tests` checks each lever's `reader` against its class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// 2026-09-26: Read by a `met` process at run time (serve or bench). A recipe may declare
    /// it in `env:`.
    Runtime,
    /// 2026-09-26: Places, logs or builds for the harness. A recipe may not declare it.
    Harness,
    /// 2026-09-26: Read when the binary is built (a build script or `option_env!`).
    Build,
    /// 2026-09-26: Read only by a test or a dev example/benchmark.
    Dev,
    /// 2026-09-26: Read only by an operator script or the site build, outside `crates/`.
    Tool,
}

impl Class {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::Harness => "harness",
            Self::Build => "build",
            Self::Dev => "dev",
            Self::Tool => "tool",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// 2026-09-26: On or off; `default` names the state when the variable is unset.
    Switch,
    Int,
    Float,
    Text,
    Path,
}

impl Ty {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Switch => "switch",
            Self::Int => "int",
            Self::Float => "float",
            Self::Text => "text",
            Self::Path => "path",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeverSpec {
    pub env: &'static str,
    /// 2026-09-26: `group.name`, ending in the variable's suffix in lower case.
    pub path: &'static str,
    pub ty: Ty,
    /// 2026-09-26: The value in force when the variable is unset: `on`/`off` for a switch (of
    /// the effect the name describes), a number, `unset`, `built-in`, or `target-declared`
    /// (the `[defaults]` table of `kernels/<hw>/HARDWARE.toml`).
    pub default: &'static str,
    /// 2026-09-26: The `met serve` flag that sets the same thing.
    pub flag: Option<&'static str>,
    pub class: Class,
    /// 2026-09-26: The repository file that reads it; `levers_tests` checks that it names it.
    pub reader: &'static str,
    pub doc: &'static str,
}

pub const PREFIX: &str = "METRALE_";

/// 2026-10-17: How a raw value parses. `Presence` is a `Ty::Switch` lever whose doc says
/// "Any value turns it on"; the table keeps `Ty::Switch` for it so the published table is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// 2026-10-17: `1|true|on|yes` or `0|false|off|no`, trimmed, ASCII case-insensitive.
    Switch,
    /// 2026-10-17: Set at all (empty and `0` included) means on.
    Presence,
    Int,
    /// 2026-10-17: Finite only; `nan` and `inf` are malformed.
    Float,
    Text,
    Path,
}

impl Rule {
    #[must_use]
    pub fn ty(self) -> Ty {
        match self {
            Self::Switch | Self::Presence => Ty::Switch,
            Self::Int => Ty::Int,
            Self::Float => Ty::Float,
            Self::Text => Ty::Text,
            Self::Path => Ty::Path,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Switch => "switch",
            Self::Presence => "presence",
            Self::Int => "int",
            Self::Float => "float",
            Self::Text => "text",
            Self::Path => "path",
        }
    }
}
