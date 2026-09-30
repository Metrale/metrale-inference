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
    assert_eq!(progs.len(), 4);
    let f32 = BTreeMap::from([("h".to_string(), StateDtype::F32)]);
    for p in &progs {
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
