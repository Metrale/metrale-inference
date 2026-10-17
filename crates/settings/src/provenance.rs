// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Where each resolved lever value came from.
//!
//! Owner: settings.
//! Invariants: one entry per resolved lever, in resolution order.

use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Cli,
    Recipe,
    Env,
    /// 2026-10-17: The lever's declared `default`; nothing set it.
    Default,
}

impl Source {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Recipe => "recipe",
            Self::Env => "env",
            Self::Default => "default",
        }
    }
}

/// 2026-10-17: `Default` is the empty record a resolution starts from, not a config value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Provenance(Vec<(&'static str, Source)>);

impl Provenance {
    pub fn record(&mut self, env: &'static str, source: Source) {
        self.0.push((env, source));
    }

    #[must_use]
    pub fn source_of(&self, env: &str) -> Option<Source> {
        self.0.iter().find(|(e, _)| *e == env).map(|&(_, s)| s)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, Source)> + '_ {
        self.0.iter().copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
