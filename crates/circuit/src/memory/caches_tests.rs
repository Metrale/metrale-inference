// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Cache sizing: one unit rule per kind, the hash-table law of the prompt-lookup
//! index, absent inputs sizing nothing, and keyed formats refused when not given.
//!
//! Owner: metrale-circuit (memory).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::ir::Section;
use crate::state::{Lifetime, StateDecl, StateFormat};

fn decl(id: &str, kind: StateKind, format: StateFormat, elements: u64) -> StateDecl {
    StateDecl {
        id: id.into(),
        local: id.into(),
        block: "b".into(),
        layer: None,
        section: Section::Main,
        kind,
        format,
        elements,
        verify: None,
        lifetime: Lifetime::of_kind(kind),
        copies: None,
    }
}

fn every_kind() -> Vec<StateDecl> {
    let f = |d| StateFormat::Fixed(d);
    vec![
        decl("snap", StateKind::PrefixSnapshot, f(StateDtype::F32), 10),
        decl("ring", StateKind::RingSnapshot, f(StateDtype::F32), 10),
        decl("carry", StateKind::CarryStash, f(StateDtype::Bf16), 10),
        decl("ctable", StateKind::CarryTable, f(StateDtype::U32), 2),
        decl("table", StateKind::VerifyTable, f(StateDtype::U64), 3),
        decl("accept", StateKind::AcceptStash, f(StateDtype::Bf16), 7),
        decl("capture", StateKind::HiddenCapture, f(StateDtype::Bf16), 7),
        decl(
            "lookup",
            StateKind::PromptLookupIndex,
            f(StateDtype::U64),
            2,
        ),
        decl("mask", StateKind::TokenTreeMask, f(StateDtype::U8), 1),
        decl("drafts", StateKind::DraftTokens, f(StateDtype::I32), 1),
        // 2026-10-02: Not caches: left to the state plan.
        decl("h", StateKind::Recurrent, f(StateDtype::F32), 10),
        decl("k", StateKind::PagedKv, f(StateDtype::Bf16), 10),
    ]
}

fn by_id(t: &[CacheTerm]) -> BTreeMap<&str, (u64, u64)> {
    t.iter()
        .map(|x| (x.state.as_str(), (x.units, x.bytes)))
        .collect()
}

#[test]
fn each_kind_takes_its_own_units_and_absent_inputs_size_nothing() {
    let inputs = CacheInputs {
        prefix_snapshot_slots: 5,
        ring: (8, 3),
        carry: (33, 64),
        verify_table_rows: 128,
        capture_rows: 100,
        lookup: Some(LookupInputs {
            sequences: 2,
            history_tokens: 1000,
        }),
        tree: Some((4, 16)),
        drafts: Some((4, 16)),
    };
    let t = cache_terms(&every_kind(), &BTreeMap::new(), &inputs).unwrap();
    let got = by_id(&t);
    // 2026-10-02: 1000 entries -> 1142 -> 2048 buckets of 17 bytes, plus a control group.
    let lookup_one = 2048 * 17 + 16;
    let want: BTreeMap<&str, (u64, u64)> = [
        ("snap", (5, 5 * 10 * 4)),
        ("ring", (24, 24 * 10 * 4)),
        ("carry", (33, 33 * 10 * 2)),
        ("ctable", (64, 64 * 2 * 4)),
        ("table", (128, 128 * 3 * 8)),
        ("accept", (128, 128 * 7 * 2)),
        ("capture", (100, 100 * 7 * 2)),
        ("lookup", (2000, 2 * lookup_one)),
        ("mask", (4 * 16 * 16, 4 * 16 * 16)),
        ("drafts", (64, 64 * 4)),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, want);
    assert!(
        t.iter()
            .all(|x| x.host == (x.kind == StateKind::PromptLookupIndex))
    );
    let none = cache_terms(&every_kind(), &BTreeMap::new(), &CacheInputs::default()).unwrap();
    assert!(
        none.iter().all(|x| x.bytes == 0 && x.units == 0),
        "{none:?}"
    );
    assert_eq!(none.len(), 10, "the recurrent and KV states are not caches");
}

#[test]
fn the_hash_table_law_follows_hashbrown_growth() {
    assert_eq!(hash_table_bytes(0, 16), Some(0));
    assert_eq!(hash_table_bytes(1, 16), Some(4 * 17 + 16));
    assert_eq!(hash_table_bytes(7, 16), Some(8 * 17 + 16));
    // 2026-10-02: 14 entries fill 16 buckets at 7/8 load; the 15th doubles them.
    assert_eq!(hash_table_bytes(14, 16), Some(16 * 17 + 16));
    assert_eq!(hash_table_bytes(15, 16), Some(32 * 17 + 16));
    assert_eq!(hash_table_bytes(u64::MAX, 16), None);
}

#[test]
fn a_keyed_cache_format_must_be_given() {
    let d = vec![decl(
        "snap",
        StateKind::PrefixSnapshot,
        StateFormat::Keyed("snap_dtype".into()),
        4,
    )];
    let inputs = CacheInputs {
        prefix_snapshot_slots: 2,
        ..CacheInputs::default()
    };
    assert_eq!(
        cache_terms(&d, &BTreeMap::new(), &inputs),
        Err(StateError::MissingFormat {
            state: "snap".into(),
            key: "snap_dtype".into()
        })
    );
    let f = BTreeMap::from([("snap_dtype".to_string(), StateDtype::F16)]);
    assert_eq!(cache_terms(&d, &f, &inputs).unwrap()[0].bytes, 2 * 4 * 2);
}

#[test]
fn an_overflowing_unit_count_is_an_error_not_a_wrap() {
    let inputs = CacheInputs {
        tree: Some((u64::MAX, 2)),
        ..CacheInputs::default()
    };
    let d = vec![decl(
        "mask",
        StateKind::TokenTreeMask,
        StateFormat::Fixed(StateDtype::U8),
        1,
    )];
    assert_eq!(
        cache_terms(&d, &BTreeMap::new(), &inputs),
        Err(StateError::Overflow("mask".into()))
    );
}
