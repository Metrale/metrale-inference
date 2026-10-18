// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The layers the resolver reads, highest precedence first: CLI, recipe, env.
//!
//! Owner: settings.
//! Invariants: a layer holds only declared `Runtime` levers, each at most once, keyed by the
//! declared (interned) env name.

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::Env;
use crate::groups;
use crate::resolve::{Layer, SettingsError};
use crate::spec::Class;

/// 2026-10-17: Values given on the `met` command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliLayer(BTreeMap<&'static str, String>);

/// 2026-10-17: Values from a recipe's `env:` block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeLayer(BTreeMap<&'static str, String>);

impl CliLayer {
    #[must_use]
    pub fn empty() -> Self {
        Self(BTreeMap::new())
    }

    /// 2026-10-17: Fails on an undeclared name (Undeclared), a non-Runtime lever (NotRuntime)
    /// or a repeat (Duplicate).
    ///
    /// # Errors
    /// The first offending pair.
    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self, SettingsError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        layer_from_pairs(Layer::Cli, pairs).map(Self)
    }

    #[must_use]
    pub fn get(&self, env: &str) -> Option<&str> {
        self.0.get(env).map(String::as_str)
    }
}

impl RecipeLayer {
    #[must_use]
    pub fn empty() -> Self {
        Self(BTreeMap::new())
    }

    /// 2026-10-17: Fails on an undeclared name (Undeclared), a non-Runtime lever (NotRuntime)
    /// or a repeat (Duplicate).
    ///
    /// # Errors
    /// The first offending pair.
    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self, SettingsError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        layer_from_pairs(Layer::Recipe, pairs).map(Self)
    }

    #[must_use]
    pub fn get(&self, env: &str) -> Option<&str> {
        self.0.get(env).map(String::as_str)
    }
}

fn layer_from_pairs<I, K, V>(
    from: Layer,
    pairs: I,
) -> Result<BTreeMap<&'static str, String>, SettingsError>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: Into<String>,
{
    let mut map = BTreeMap::new();
    for (name, value) in pairs {
        let name = name.as_ref();
        let spec = groups::lookup(name).ok_or_else(|| SettingsError::Undeclared {
            from,
            name: String::from(name),
        })?;
        if spec.class != Class::Runtime {
            return Err(SettingsError::NotRuntime {
                from,
                env: spec.env,
                class: spec.class,
            });
        }
        if map.insert(spec.env, value.into()).is_some() {
            return Err(SettingsError::Duplicate {
                from,
                env: spec.env,
            });
        }
    }
    Ok(map)
}

/// 2026-10-17: The three layers one resolution reads.
pub struct Sources<'a, E: Env> {
    pub env: &'a E,
    pub cli: &'a CliLayer,
    pub recipe: &'a RecipeLayer,
}
