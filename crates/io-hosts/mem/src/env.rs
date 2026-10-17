// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: An [`Env`] backed by a fixed map, for tests and fixtures.
//!
//! Owner: metrale-io-host-mem.
//! Invariants: the map is fixed at construction; keys are unique.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use metrale_abstract_io::{Env, EnvError};

/// 2026-10-17: A fixed set of variables. Names come back in byte order (`BTreeMap`).
#[derive(Clone, Debug)]
pub struct MapEnv {
    map: BTreeMap<String, String>,
}

impl MapEnv {
    /// Panics on a duplicate key (a test-fixture bug, fail fast).
    pub fn new<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut map = BTreeMap::new();
        for (key, value) in pairs {
            match map.entry(key.into()) {
                Entry::Vacant(slot) => {
                    slot.insert(value.into());
                }
                Entry::Occupied(slot) => panic!("MapEnv: duplicate key {}", slot.key()),
            }
        }
        Self { map }
    }
}

impl Env for MapEnv {
    fn var(&self, name: &str) -> Result<Option<String>, EnvError> {
        Ok(self.map.get(name).cloned())
    }

    fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}

#[cfg(test)]
#[path = "env_tests.rs"]
mod tests;
