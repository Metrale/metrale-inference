// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Tests for the state plan's arithmetic and its refusals, over declarations on the
//! toy circuit.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::ir::Circuit;
use crate::test_toy;

fn decl(id: &str, kind: StateKind, format: StateFormat, elements: u64) -> StateDecl {
    StateDecl {
        id: id.into(),
        local: id.rsplit('.').next().unwrap().into(),
        block: "gdn".into(),
        layer: Some(0),
        section: Section::Main,
        kind,
        format,
        elements,
        verify: match (kind, id.ends_with(".h")) {
            (StateKind::PagedKv, _) => None,
            (_, true) => Some(VerifySteps::H),
            (_, false) => Some(VerifySteps::Conv),
        },
        lifetime: Lifetime::of_kind(kind),
        copies: None,
    }
}

fn toy() -> Circuit {
    let mut c = test_toy::circuit(1);
    c.states = vec![
        decl(
            "l0.gdn.h",
            StateKind::Recurrent,
            StateFormat::Keyed("h".into()),
            1000,
        ),
        decl(
            "l0.gdn.conv",
            StateKind::Recurrent,
            StateFormat::Fixed(StateDtype::F32),
            30,
        ),
        decl(
            "l0.attn.k",
            StateKind::PagedKv,
            StateFormat::Keyed("kv".into()),
            8,
        ),
    ];
    c
}

fn inputs(h: StateDtype) -> StateInputs {
    StateInputs {
        formats: BTreeMap::from([("h".into(), h), ("kv".into(), StateDtype::Fp8)]),
        slots: 5,
        verify: Some(VerifyInputs {
            h_steps: vec![3, 1, 3],
            conv_steps: 4,
        }),
        kv: Some(KvInputs {
            blocks: 10,
            block_size: 16,
        }),
        draft_kv: None,
    }
}

#[test]
fn every_term_is_units_times_elements_times_element_size() {
    let p = StatePlan::new(&toy().states, &inputs(StateDtype::F32)).unwrap();
    let term = |s: &str, h: Holding| {
        p.terms
            .iter()
            .find(|t| t.state == s && t.holding == h)
            .map(|t| (t.units, t.bytes))
    };
    assert_eq!(term("l0.gdn.h", Holding::Live), Some((5, 5 * 4000)));
    assert_eq!(term("l0.gdn.h", Holding::Steps), Some((7, 7 * 4000)));
    assert_eq!(term("l0.gdn.h", Holding::Checkpoint), Some((3, 3 * 4000)));
    assert_eq!(term("l0.gdn.conv", Holding::Steps), Some((12, 12 * 120)));
    assert_eq!(term("l0.attn.k", Holding::Blocks), Some((160, 160 * 8)));
    assert_eq!(p.terms.len(), 7);
    assert_eq!(p.bytes(), p.terms.iter().map(|t| t.bytes).sum::<u64>());
    // 2026-09-30: The keyed format is the input's: an f16 h halves only the h terms.
    let half = StatePlan::new(&toy().states, &inputs(StateDtype::F16)).unwrap();
    let h = |p: &StatePlan| p.bytes_where(|t| t.state == "l0.gdn.h");
    assert_eq!(h(&half) * 2, h(&p));
    assert_eq!(p.bytes() - half.bytes(), h(&half));
}

#[test]
fn without_speculation_or_a_cache_only_the_live_slots_are_sized() {
    let mut i = inputs(StateDtype::F32);
    i.verify = None;
    i.kv = None;
    let p = StatePlan::new(&toy().states, &i).unwrap();
    assert!(p.terms.iter().all(|t| t.holding == Holding::Live));
    assert_eq!(p.bytes(), 5 * 4000 + 5 * 120);
}

#[test]
fn a_missing_format_key_or_verify_rule_is_refused() {
    let mut i = inputs(StateDtype::F32);
    i.formats.remove("kv");
    assert!(matches!(
        StatePlan::new(&toy().states, &i),
        Err(StateError::MissingFormat { key, .. }) if key == "kv"
    ));
    let mut c = toy();
    c.states[1].verify = None;
    assert!(matches!(
        StatePlan::new(&c.states, &inputs(StateDtype::F32)),
        Err(StateError::Inputs(m)) if m.contains("l0.gdn.conv")
    ));
}

#[test]
fn format_and_verify_spellings_parse_or_are_refused() {
    assert_eq!(
        StateFormat::parse("{kv_cache_dtype}"),
        Some(StateFormat::Keyed("kv_cache_dtype".into()))
    );
    assert_eq!(
        StateFormat::parse("f32"),
        Some(StateFormat::Fixed(StateDtype::F32))
    );
    assert_eq!(StateFormat::parse("{}"), None);
    assert_eq!(StateFormat::parse("fp4"), None);
    assert_eq!(VerifySteps::parse("h_steps"), Some(VerifySteps::H));
    assert_eq!(VerifySteps::parse("steps"), None);
}

/// 2026-09-30: Every term carries its lifetime: the live slots and KV blocks as their state is
/// declared, the verify intermediates and checkpoints only across one verify.
#[test]
fn every_term_carries_the_lifetime_of_its_holding() {
    let p = StatePlan::new(&toy().states, &inputs(StateDtype::F32)).unwrap();
    for t in &p.terms {
        let want = match t.holding {
            Holding::Live => Lifetime::Sequence,
            Holding::Blocks => Lifetime::Model,
            Holding::Steps | Holding::Checkpoint => Lifetime::Verify,
        };
        assert_eq!(t.lifetime, want, "{} {:?}", t.state, t.holding);
    }
    assert_eq!(Lifetime::parse("verify"), Some(Lifetime::Verify));
    assert_eq!(Lifetime::parse("step"), None);
}

/// 2026-09-30: An engine-configured `model_type` finds its circuit's recurrent layer block,
/// renamed or not; an attention-only circuit has none; an unserved type has no circuit.
#[test]
fn recurrent_states_follow_the_engine_model_type() {
    let dims: BTreeMap<String, u64> = [
        ("lin_k_heads", 16),
        ("lin_k_dim", 128),
        ("lin_v_heads", 32),
        ("lin_v_dim", 128),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    for t in ["qwen3_5", "qwen3_5_moe", "qwen3_6_moe", "holo3_1_moe"] {
        let s = crate::recurrent_states(t, &dims).unwrap().unwrap();
        let ids: Vec<&str> = s.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["gdn.h", "gdn.conv"], "{t}");
        assert_eq!(s[0].elements, 32 * 128 * 128, "{t}");
        assert_eq!(s[1].elements, (16 * 128 * 2 + 32 * 128) * 4, "{t}");
    }
    assert_eq!(
        crate::recurrent_states("llama", &dims).unwrap(),
        Some(Vec::new())
    );
    assert_eq!(crate::recurrent_states("gemma4", &dims).unwrap(), None);
    // 2026-09-30: A dim the declaration reads and the caller does not give is an error.
    assert!(crate::recurrent_states("nemotron_h", &dims).is_err());
}
