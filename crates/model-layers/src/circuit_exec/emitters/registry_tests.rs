// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The registry is one slice per module, so two owners can register the same id
//! without a merge conflict. `emitter` returns the first match, which would hide the second.

use std::collections::BTreeMap;

use super::{MODULES, emitter};

#[test]
fn every_emitter_id_is_registered_once() {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for e in MODULES.iter().flat_map(|m| m.iter()) {
        *seen.entry(e.id()).or_default() += 1;
    }
    let twice: Vec<_> = seen.iter().filter(|(_, n)| **n > 1).collect();
    assert!(twice.is_empty(), "emitter ids registered more than once: {twice:?}");
}

#[test]
fn every_registered_id_resolves_to_itself() {
    for e in MODULES.iter().flat_map(|m| m.iter()) {
        let found = emitter(e.id()).expect("a registered id resolves");
        assert_eq!(found.id(), e.id());
    }
    assert!(emitter("no-such-emitter").is_err());
}
