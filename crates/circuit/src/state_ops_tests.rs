// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Tests for the state programs over the toy circuit's declared states.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;
use crate::state::{StateDecl, StateFormat, VerifySteps};
use crate::test_toy;

fn decl(id: &str, kind: StateKind, section: Section, elements: u64) -> StateDecl {
    StateDecl {
        id: id.into(),
        local: id.rsplit('.').next().unwrap().into(),
        block: "gdn".into(),
        layer: Some(0),
        section,
        kind,
        format: StateFormat::Keyed("h".into()),
        elements,
        verify: (kind == StateKind::Recurrent).then_some(VerifySteps::H),
        lifetime: crate::state::Lifetime::of_kind(kind),
        copies: None,
    }
}

#[test]
fn every_program_has_one_node_per_recurrent_target_state_sized_from_its_declaration() {
    let mut c = test_toy::circuit(1);
    c.states = vec![
        decl("l0.gdn.h", StateKind::Recurrent, Section::Main, 100),
        decl("l0.attn.k", StateKind::PagedKv, Section::Main, 8),
        decl("l1.gdn.h", StateKind::Recurrent, Section::Main, 50),
        decl("draft.gdn.h", StateKind::Recurrent, Section::Draft, 7),
    ];
    let progs = state_programs(&c);
    assert_eq!(progs.len(), StateProgramId::ALL.len());
    let f32 = BTreeMap::from([("h".to_string(), StateDtype::F32)]);
    // 2026-10-03: The toy declares no snapshot caches, so the ring and prefix programs are empty.
    for p in &progs[4..] {
        assert!(p.nodes.is_empty(), "{:?}", p.id);
    }
    for p in &progs[..4] {
        let ids: Vec<&str> = p.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                format!("{}.l0.gdn.h", p.id.name()),
                format!("{}.l1.gdn.h", p.id.name())
            ],
        );
        assert!(p.nodes.iter().all(|n| n.op == p.id.op()));
        assert_eq!(p.bytes(&c, &f32).unwrap(), (100 + 50) * 4);
    }
    let rollback = &progs[1];
    assert_eq!(
        rollback.nodes[0].op,
        StateOp::Copy {
            from: StatePlace::Checkpoint,
            to: StatePlace::Live
        }
    );
    // 2026-09-30: The keyed format is the plan input's, and a missing key is an error.
    let f16 = BTreeMap::from([("h".to_string(), StateDtype::F16)]);
    assert_eq!(progs[0].bytes(&c, &f16).unwrap(), (100 + 50) * 2);
    assert!(progs[0].bytes(&c, &BTreeMap::new()).is_err());
}

fn snapshot(id: &str, kind: StateKind, copies: &str, format: StateFormat) -> StateDecl {
    StateDecl {
        id: id.into(),
        local: id.rsplit('.').next().unwrap().into(),
        block: "gdn".into(),
        layer: Some(0),
        section: Section::Main,
        kind,
        format,
        elements: 0,
        verify: None,
        lifetime: crate::state::Lifetime::of_kind(kind),
        copies: Some(copies.into()),
    }
}

/// 2026-10-03: The ring and prefix programs pair each recurrent state with the snapshot that
/// copies it (`of`), leave out a state whose block declares none, and size each node by its
/// op: the ring keeps an f16 pool's stored bytes, the prefix cache widens them to FP32.
#[test]
fn snapshot_programs_pair_a_state_with_its_copy_and_size_by_the_op() {
    let mut c = test_toy::circuit(1);
    c.states = vec![
        decl("l0.gdn.h", StateKind::Recurrent, Section::Main, 100),
        decl("l1.gdn.h", StateKind::Recurrent, Section::Main, 50),
        snapshot(
            "l0.gdn.ring_h",
            StateKind::RingSnapshot,
            "l0.gdn.h",
            StateFormat::Fixed(StateDtype::F32),
        ),
        snapshot(
            "l0.gdn.prefix_h",
            StateKind::PrefixSnapshot,
            "l0.gdn.h",
            StateFormat::Fixed(StateDtype::F32),
        ),
        snapshot(
            "l1.gdn.prefix_h",
            StateKind::PrefixSnapshot,
            "l1.gdn.h",
            StateFormat::Fixed(StateDtype::F32),
        ),
    ];
    let progs = state_programs(&c);
    let get = |id: StateProgramId| progs.iter().find(|p| p.id == id).unwrap();
    let f16 = BTreeMap::from([("h".to_string(), StateDtype::F16)]);

    let ring = get(StateProgramId::RingSave);
    assert_eq!(ring.nodes.len(), 1, "l1 declares no ring snapshot");
    assert_eq!(ring.nodes[0].cache, Some(2));
    assert_eq!(
        ring.nodes[0].dtypes(&c, &f16).unwrap(),
        (StateDtype::F16, StateDtype::F32)
    );
    assert_eq!(ring.bytes(&c, &f16).unwrap(), 100 * 2);
    assert_eq!(get(StateProgramId::RingRestore).bytes(&c, &f16).unwrap(), 100 * 2);

    let save = get(StateProgramId::PrefixSave);
    assert_eq!(
        save.nodes.iter().map(|n| n.cache).collect::<Vec<_>>(),
        [Some(3), Some(4)]
    );
    assert_eq!(save.bytes(&c, &f16).unwrap(), (100 + 50) * 4);
    let restore = get(StateProgramId::PrefixRestore);
    assert_eq!(
        restore.nodes[1].dtypes(&c, &f16).unwrap(),
        (StateDtype::F32, StateDtype::F16)
    );
    assert_eq!(restore.bytes(&c, &f16).unwrap(), (100 + 50) * 2);
}

/// 2026-10-03: A node that touches a snapshot without naming its cache is refused, never sized
/// as the live state.
#[test]
fn a_snapshot_node_without_its_cache_is_refused() {
    let mut c = test_toy::circuit(1);
    c.states = vec![decl("l0.gdn.h", StateKind::Recurrent, Section::Main, 100)];
    let node = StateOpNode {
        id: "ring_save.l0.gdn.h".into(),
        state: 0,
        cache: None,
        op: StateProgramId::RingSave.op(),
    };
    let f32 = BTreeMap::from([("h".to_string(), StateDtype::F32)]);
    assert!(node.bytes(&c, &f32).is_err());
}

/// 2026-10-03: The hidden-row prefix snapshot is the target's prefix snapshot that copies no
/// state; a state's own prefix copy is never taken for it.
#[test]
fn the_prefix_hidden_row_is_the_snapshot_that_copies_no_state() {
    let mut c = test_toy::circuit(1);
    c.states = vec![
        decl("l0.gdn.h", StateKind::Recurrent, Section::Main, 100),
        snapshot(
            "l0.gdn.prefix_h",
            StateKind::PrefixSnapshot,
            "l0.gdn.h",
            StateFormat::Fixed(StateDtype::F32),
        ),
    ];
    assert_eq!(prefix_hidden(&c), None);
    let mut hidden = snapshot(
        "head.prefix_hidden",
        StateKind::PrefixSnapshot,
        "unused",
        StateFormat::Fixed(StateDtype::Bf16),
    );
    hidden.copies = None;
    c.states.push(hidden);
    assert_eq!(prefix_hidden(&c), Some(2));
}
