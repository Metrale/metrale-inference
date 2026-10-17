// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The table derived from the `metrale-settings` groups against the frozen
//! hand-written table it replaced, entry for entry and in order.
//!
//! Owner: config.
//! Invariants: `table_a`/`table_b` are never edited; they are deleted when the first row
//! changes on purpose.

use super::{LeverSpec, all, table_a, table_b};

#[test]
fn derived_table_equals_frozen_table_entry_for_entry() {
    let old: Vec<&LeverSpec> = table_a::LEVERS.iter().chain(table_b::LEVERS).collect();
    let new: Vec<&LeverSpec> = all().collect();
    for (i, (o, n)) in old.iter().zip(&new).enumerate() {
        // 2026-10-17: `LeverSpec: PartialEq` compares all eight fields.
        assert_eq!(o, n, "lever {i} ({}) differs", o.env);
    }
    assert_eq!(old.len(), new.len(), "lever count");
}
