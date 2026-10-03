// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The attention prefill rules of FUSIONS.toml against the emitter's route: every
//! `prefill_attention` rule lists exactly the kernels the emitter launches for its arm, its rows
//! sit inside one projection arm and one attention kernel at both ends (the legacy dispatch
//! switches nowhere inside a bucket), and each prefill mode's rules tile every row count once.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::{Mode, Rule};

use super::attn_route::{Attn, Proj, expected};

fn rules() -> Vec<Rule> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/FUSIONS.toml"
    );
    let text = std::fs::read_to_string(path).unwrap();
    metrale_circuit::parse_rule_set(&text)
        .unwrap()
        .rules
        .into_iter()
        .filter(|r| r.emitter == "prefill_attention")
        .collect()
}

/// 2026-10-03: The route a rule's kernels name: its arm, its RoPE kernel and its attention.
fn route(r: &Rule) -> (Proj, (String, String), Attn) {
    let proj = Proj::of(&r.kernels[0]).unwrap();
    let per = proj.kernels().len();
    let at = |i: usize| (r.kernels[i].module.clone(), r.kernels[i].func.clone());
    let attn = match at(3 * per + 4) {
        (m, f) if (m.as_str(), f.as_str()) == ("attn_prefill_fa128", "attn_prefill_fa128") => {
            Attn::Contiguous
        }
        (m, f) if (m.as_str(), f.as_str()) == ("prefill_paged", "attn_prefill_paged") => {
            Attn::PagedSmall
        }
        (m, f)
            if (m.as_str(), f.as_str()) == ("attn_prefill_fa128", "attn_prefill_fa128_paged") =>
        {
            Attn::PagedFa128
        }
        other => panic!("rule {}: {other:?} is no prefill attention kernel", r.id),
    };
    (proj, at(3 * per + 2), attn)
}

#[test]
fn every_prefill_attention_rule_lists_the_emitters_launches_for_its_arm() {
    let rules = rules();
    assert_eq!(rules.len(), 7, "the contiguous and paged ladders");
    for r in &rules {
        let (proj, rope, attn) = route(r);
        let want = expected(proj, (rope.0.as_str(), rope.1.as_str()), attn);
        let got: Vec<(String, String)> = r
            .kernels
            .iter()
            .map(|k| (k.module.clone(), k.func.clone()))
            .collect();
        assert_eq!(got, want, "rule {}", r.id);
        let paged = r.modes.contains(&Mode::PrefillChunk);
        assert_eq!(
            paged,
            attn != Attn::Contiguous,
            "rule {}: the mode and the attention route disagree",
            r.id
        );
        assert_eq!(
            r.copies.is_some(),
            paged,
            "rule {}: only the paged route zeroes V (one copy)",
            r.id
        );
    }
}

/// 2026-10-03: A rule fused at its bucket's top must be exact at its bottom: the legacy
/// projection arm and attention kernel are the same at both ends.
#[test]
fn every_rule_sits_inside_one_legacy_arm() {
    for r in rules() {
        let (proj, _, attn) = route(&r);
        for m in [r.rows.0, r.rows.1.min(1 << 20)] {
            let m = u32::try_from(m).unwrap();
            assert!(
                proj.serves(m),
                "rule {}: {proj:?} does not serve {m} rows",
                r.id
            );
            assert!(
                attn.serves(m),
                "rule {}: {attn:?} does not serve {m} rows",
                r.id
            );
        }
    }
}

#[test]
fn each_prefill_mode_is_tiled_once() {
    let rules = rules();
    for mode in Mode::PREFILL {
        let mut ranges: Vec<(u64, u64)> = rules
            .iter()
            .filter(|r| r.modes.contains(&mode))
            .map(|r| r.rows)
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().map(|r| r.0), Some(1), "{mode:?}");
        for w in ranges.windows(2) {
            assert_eq!(w[0].1 + 1, w[1].0, "{mode:?}: a gap or an overlap at {w:?}");
        }
        assert_eq!(ranges.last().map(|r| r.1), Some(1 << 20), "{mode:?}");
    }
}
