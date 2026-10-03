// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The swap record over hand-declared states: legacy's order, the sizes from the
//! declarations and the keyed formats, what it leaves out, and the refusals.
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants: none beyond the types.

use super::*;
use crate::state::{Lifetime, StateDecl, VerifySteps};
use crate::test_toy;

fn decl(
    id: &str,
    layer: Option<usize>,
    section: Section,
    kind: StateKind,
    fmt: &str,
    elements: u64,
) -> StateDecl {
    StateDecl {
        id: id.into(),
        local: id.rsplit('.').next().unwrap().into(),
        block: "b".into(),
        layer,
        section,
        kind,
        format: StateFormat::parse(fmt).unwrap(),
        elements,
        verify: (kind == StateKind::Recurrent).then_some(VerifySteps::H),
        lifetime: Lifetime::of_kind(kind),
        copies: None,
    }
}

/// 2026-10-03: Layers 0 and 2 GatedDeltaNet (h, conv), 1 and 3 attention (k, v); a draft KV, a
/// prefix-snapshot cache and an embedding-block cache that must stay out.
fn circuit() -> Circuit {
    let mut c = test_toy::circuit(4);
    let main = Section::Main;
    c.states = vec![
        decl(
            "embed.lookup",
            None,
            main,
            StateKind::PromptLookupIndex,
            "u64",
            2,
        ),
        decl(
            "l0.gdn.h",
            Some(0),
            main,
            StateKind::Recurrent,
            "{ssm_h_storage}",
            100,
        ),
        decl("l0.gdn.conv", Some(0), main, StateKind::Recurrent, "f32", 7),
        decl(
            "l0.gdn.prefix_h",
            Some(0),
            main,
            StateKind::PrefixSnapshot,
            "f32",
            100,
        ),
        decl(
            "l1.attn.k",
            Some(1),
            main,
            StateKind::PagedKv,
            "{kv_cache_dtype}",
            8,
        ),
        decl(
            "l1.attn.v",
            Some(1),
            main,
            StateKind::PagedKv,
            "{kv_cache_dtype}",
            8,
        ),
        decl(
            "l2.gdn.h",
            Some(2),
            main,
            StateKind::Recurrent,
            "{ssm_h_storage}",
            100,
        ),
        decl("l2.gdn.conv", Some(2), main, StateKind::Recurrent, "f32", 7),
        decl(
            "l3.attn.k",
            Some(3),
            main,
            StateKind::PagedKv,
            "{kv_cache_dtype}",
            4,
        ),
        decl(
            "l3.attn.v",
            Some(3),
            main,
            StateKind::PagedKv,
            "{kv_cache_dtype}",
            4,
        ),
        decl(
            "draft.attn.k",
            None,
            Section::Draft,
            StateKind::PagedKv,
            "bf16",
            8,
        ),
    ];
    c
}

fn formats(h: StateDtype) -> BTreeMap<String, StateDtype> {
    BTreeMap::from([
        ("kv_cache_dtype".to_string(), StateDtype::Bf16),
        ("ssm_h_storage".to_string(), h),
    ])
}

#[test]
fn the_record_is_every_block_layer_by_layer_then_the_recurrent_units_in_layer_order() {
    let p = SwapPlan::new(&circuit(), &formats(StateDtype::F32), 16).unwrap();
    assert_eq!(
        p.kv,
        vec![
            KvLayer {
                layer: 1,
                k_block_bytes: 8 * 2 * 16,
                v_block_bytes: 8 * 2 * 16
            },
            KvLayer {
                layer: 3,
                k_block_bytes: 4 * 2 * 16,
                v_block_bytes: 4 * 2 * 16
            },
        ]
    );
    let rec: Vec<(&str, u64)> = p
        .recurrent
        .iter()
        .map(|r| (r.state.as_str(), r.bytes))
        .collect();
    assert_eq!(
        rec,
        [
            ("l0.gdn.h", 400),
            ("l0.gdn.conv", 28),
            ("l2.gdn.h", 400),
            ("l2.gdn.conv", 28)
        ]
    );
    let segs = p.segments(2);
    let pieces: Vec<Piece> = segs.iter().map(|s| s.piece).collect();
    assert_eq!(
        pieces,
        [
            Piece::K { attn: 0, block: 0 },
            Piece::V { attn: 0, block: 0 },
            Piece::K { attn: 1, block: 0 },
            Piece::V { attn: 1, block: 0 },
            Piece::K { attn: 0, block: 1 },
            Piece::V { attn: 0, block: 1 },
            Piece::K { attn: 1, block: 1 },
            Piece::V { attn: 1, block: 1 },
            Piece::Recurrent { index: 0 },
            Piece::Recurrent { index: 1 },
            Piece::Recurrent { index: 2 },
            Piece::Recurrent { index: 3 },
        ]
    );
    let mut at = 0;
    for s in &segs {
        assert_eq!(s.offset, at, "{:?} starts where the previous ends", s.piece);
        at += s.bytes;
    }
    assert_eq!(at, p.record_bytes(2));
    assert_eq!(p.largest_segment(), 400);
    assert_eq!(p.staging_chunk(), 800);
}

#[test]
fn the_h_unit_follows_its_storage_format() {
    let p = SwapPlan::new(&circuit(), &formats(StateDtype::F16), 16).unwrap();
    assert_eq!(p.recurrent[0].bytes, 200);
    assert_eq!(p.recurrent[1].bytes, 28, "the conv window stays f32");
}

#[test]
fn the_refusals_name_their_cause() {
    let c = circuit();
    let only_kv = BTreeMap::from([("kv_cache_dtype".to_string(), StateDtype::Bf16)]);
    assert_eq!(
        SwapPlan::new(&c, &only_kv, 16),
        Err(SwapError::MissingFormat {
            state: "l0.gdn.h".into(),
            key: "ssm_h_storage".into()
        })
    );
    let mut lopsided = circuit();
    lopsided.states.retain(|s| s.id != "l3.attn.v");
    assert_eq!(
        SwapPlan::new(&lopsided, &formats(StateDtype::F32), 16),
        Err(SwapError::Sides(3))
    );
}
