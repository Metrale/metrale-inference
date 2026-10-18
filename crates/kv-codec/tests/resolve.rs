// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Plan resolution: the boundary-layer rule and the auto boundary count, ported from
//! the server's `build_layer_kv_dtypes` and `auto_high_precision_layers` tests, plus a golden
//! table of the auto count for every accepted spelling against the old formulas.

use metrale_kv_codec::{
    AutoBoundary, BlockGeometry, KvCachePlan, KvCodecId, KvPair, KvPlanInputs, PAIR_ALIASES,
    layer_entry,
};

const G: BlockGeometry = BlockGeometry {
    block_size: 16,
    num_kv_heads: 4,
    head_dim: 128,
};

fn pair(s: &str) -> KvPair {
    s.parse().unwrap()
}

fn plan(requested: &str, n: usize, boundary_layers: u32) -> KvCachePlan {
    KvCachePlan::resolve(&KvPlanInputs {
        requested: pair(requested),
        boundary_layers,
        boundary: KvPair::symmetric(KvCodecId::Bf16),
        geometry: &vec![G; n],
    })
}

fn names(p: &KvCachePlan) -> Vec<&'static str> {
    p.layers.iter().map(|l| l.pair().name()).collect()
}

#[test]
fn zero_boundary_layers_is_uniform() {
    let p = plan("nvfp4", 12, 0);
    assert_eq!(p.layers.len(), 12);
    assert_eq!(p.uniform(), Some(pair("nvfp4")));
}

#[test]
fn a_requested_pair_equal_to_the_boundary_needs_no_overlay() {
    assert_eq!(plan("bf16", 12, 2).uniform(), Some(pair("bf16")));
}

#[test]
fn the_first_and_last_boundary_layers_hold_the_boundary_pair() {
    let p = plan("nvfp4", 12, 2);
    let mut want = vec!["nvfp4"; 12];
    for i in [0, 1, 10, 11] {
        want[i] = "bf16";
    }
    assert_eq!(names(&p), want);
    assert_eq!(p.uniform(), None);
}

#[test]
fn overlapping_boundaries_cover_every_layer() {
    // 2026-10-18: With 4 layers the first 3 and the last 3 overlap and cover them all.
    assert_eq!(names(&plan("fp8", 4, 3)), vec!["bf16"; 4]);
    assert_eq!(names(&plan("nvfp4", 1, 1)), vec!["bf16"]);
    assert_eq!(names(&plan("turbo3", 3, u32::MAX)), vec!["bf16"; 3]);
}

#[test]
fn an_asymmetric_pair_keeps_both_sides_outside_the_boundary() {
    let p = plan("turbo4k3v", 5, 1);
    assert_eq!(
        names(&p),
        [
            "bf16",
            "turbo4k_turbo3v",
            "turbo4k_turbo3v",
            "turbo4k_turbo3v",
            "bf16"
        ]
    );
    assert_eq!(
        p.pair(2).map(|q| (q.k(), q.v())),
        Some((KvCodecId::Turbo4, KvCodecId::Turbo3))
    );
    assert_eq!(p.pair(5), None);
}

#[test]
fn a_plan_has_one_entry_per_layer_and_none_for_no_layers() {
    let p = plan("turbo4", 0, 3);
    assert!(p.layers.is_empty());
    assert_eq!(p.uniform(), None);
}

/// 2026-10-18: The server's auto boundary policy per spelling before the registry moved it into
/// the table: `None` for bf16/fp8/nvfp4, `FourFifths` for turbo2 and bf16k_turbo3v, `OneThird`
/// for every other turbo spelling.
const GOLDEN_AUTO: [(&str, AutoBoundary); 16] = [
    ("bf16", AutoBoundary::None),
    ("fp8", AutoBoundary::None),
    ("nvfp4", AutoBoundary::None),
    ("turbo4", AutoBoundary::OneThird),
    ("turbo3", AutoBoundary::OneThird),
    ("turbo2", AutoBoundary::FourFifths),
    ("turbo8", AutoBoundary::OneThird),
    ("turbo4k_turbo3v", AutoBoundary::OneThird),
    ("turbo4k_turbo8v", AutoBoundary::OneThird),
    ("turbo3k_turbo8v", AutoBoundary::OneThird),
    ("bf16k_turbo4v", AutoBoundary::OneThird),
    ("bf16k_turbo3v", AutoBoundary::FourFifths),
    ("fp8k_turbo4v", AutoBoundary::OneThird),
    ("fp8k_turbo3v", AutoBoundary::OneThird),
    ("bf16k_turbo2v", AutoBoundary::OneThird),
    ("fp8k_turbo2v", AutoBoundary::OneThird),
];

#[test]
fn every_accepted_spelling_has_its_golden_auto_policy() {
    assert_eq!(
        GOLDEN_AUTO.len(),
        KvCodecId::ALL.len() + PAIR_ALIASES.len(),
        "a new codec or pair needs its golden auto policy"
    );
    for (s, want) in GOLDEN_AUTO {
        assert_eq!(pair(s).auto_boundary(), want, "{s}");
    }
    for a in PAIR_ALIASES {
        assert_eq!(pair(a.short).auto_boundary(), pair(a.long).auto_boundary());
    }
}

#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn the_auto_count_matches_the_old_formulas() {
    // 2026-10-18: The old code computed ceil(n / 3) through f32 and ceil(4n / 5) in usize.
    for n in 0u32..=100_000 {
        let third = ((n as f32 / 3.0).ceil() as u32).max(2);
        let four_fifths = (n * 4).div_ceil(5).max(4);
        assert_eq!(AutoBoundary::OneThird.layers(n), Some(third), "n={n}");
        assert_eq!(
            AutoBoundary::FourFifths.layers(n),
            Some(four_fifths),
            "n={n}"
        );
        assert_eq!(AutoBoundary::None.layers(n), None);
    }
    assert_eq!(AutoBoundary::OneThird.layers(10), Some(4));
    assert_eq!(AutoBoundary::OneThird.layers(3), Some(2));
    assert_eq!(AutoBoundary::FourFifths.layers(10), Some(8));
    assert_eq!(AutoBoundary::FourFifths.layers(2), Some(4));
}

#[test]
fn a_layer_past_the_list_is_a_coded_error_not_a_default() {
    let list = [KvCodecId::Bf16, KvCodecId::Turbo4];
    assert_eq!(layer_entry(&list, 1).unwrap(), KvCodecId::Turbo4);
    let e = layer_entry(&list, 2).unwrap_err();
    assert_eq!(e.code(), metrale_error::ErrorCode::KvPlanMissing);
    assert!(layer_entry::<KvCodecId>(&[], 0).is_err());
}
