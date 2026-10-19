// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: `torch_cpu_topk_set` against fixed expected index sets,
//! including the tie cases the indexer produces (many -inf entries, equal
//! finite scores), and the e2m1 tie rounding.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn topk_tie_order_matches_torch_cpu() {
    let cases: Vec<(Vec<f32>, usize, Vec<usize>)> = vec![
        (
            vec![
                1.0f32,
                1.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            6,
            vec![0, 1, 7, 8, 9, 11],
        ),
        (
            vec![
                0.0f32,
                0.0f32,
                0.0f32,
                f32::NEG_INFINITY,
                0.25f32,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.25f32,
            ],
            6,
            vec![0, 2, 4, 6, 8, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                0.25f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                2.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            6,
            vec![1, 3, 8, 9, 10, 12],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
            ],
            6,
            vec![3, 8, 9, 10, 11, 12],
        ),
        (
            vec![
                1.0f32,
                0.0f32,
                1.0f32,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                2.0f32,
                0.5f32,
                f32::NEG_INFINITY,
                1.0f32,
                1.0f32,
            ],
            6,
            vec![0, 2, 7, 8, 10, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                0.0f32,
                f32::NEG_INFINITY,
                0.5f32,
            ],
            6,
            vec![6, 7, 8, 9, 10, 12],
        ),
        (
            vec![
                0.25f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
            ],
            6,
            vec![0, 7, 8, 9, 12, 13],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                1.0f32,
                0.5f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                2.0f32,
                0.25f32,
                0.0f32,
                0.0f32,
                1.0f32,
            ],
            6,
            vec![1, 2, 5, 7, 8, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                0.25f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.5f32,
                f32::NEG_INFINITY,
                0.0f32,
                2.0f32,
                f32::NEG_INFINITY,
                0.0f32,
                0.25f32,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            6,
            vec![1, 4, 6, 7, 10, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.5f32,
                f32::INFINITY,
            ],
            3,
            vec![3, 4, 5],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                0.0f32,
                f32::INFINITY,
                0.5f32,
            ],
            3,
            vec![3, 5, 6],
        ),
        (
            vec![
                1.0f32,
                f32::NEG_INFINITY,
                0.5f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::INFINITY,
            ],
            3,
            vec![0, 2, 5],
        ),
    ];
    for (i, (vals, k, want)) in cases.iter().enumerate() {
        let mut got = torch_cpu_topk_set(vals, *k);
        got.sort_unstable();
        assert_eq!(&got, want, "case {i}: {vals:?} k={k}");
    }
}

#[test]
fn e2m1_rounding_ties_to_even_code() {
    // 2026-09-25: Each tie goes to the even code (the index into `E2M1`):
    // 0.25 (codes 0, 1) -> 0.0, 0.75 (1, 2) -> 1.0, 2.5 (4, 5) -> 2.0,
    // 5.0 (6, 7) -> 4.0.
    assert_eq!(to_e2m1_rne(0.25), 0.0);
    assert_eq!(to_e2m1_rne(0.75), 1.0);
    assert_eq!(to_e2m1_rne(2.5), 2.0);
    assert_eq!(to_e2m1_rne(5.0), 4.0);
    assert_eq!(to_e2m1_rne(-1.2), -1.0);
    assert_eq!(to_e2m1_rne(7.0), 6.0);
}

/// 2026-10-19: A kv source's decode step that completes no group has no latent. The golden's
/// revision leaves the shared index keys as the last publisher left them; the 1017 revision
/// republishes this layer's own cache (its `owns_k` branch), so a later layer that reads the
/// shared slot sees this layer's keys.
#[test]
fn index_k_every_step_republishes_the_cache_without_a_latent() {
    let (ihd, q_rank, dim, ratio) = (32, 32, 32, 4);
    let wq_b: Vec<f32> = (0..ihd * q_rank)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.125)
        .collect();
    let weights_proj = vec![0.25f32; dim];
    let w = IndexerWeights {
        wq_b: wq_b.as_slice(),
        weights_proj: weights_proj.as_slice(),
        wk: None,
        k_norm: None,
    };
    let fc = vec![(1.0f32, 0.0f32); 8];
    let x = vec![0.5f32; dim];
    let qr = vec![0.25f32; q_rank];
    let run = |every_step: bool| {
        let c = IndexerCfg {
            n_heads: 1,
            index_hd: ihd,
            rope_dim: 2,
            q_rank,
            dim,
            hd: 32,
            index_topk: 1,
            cand_topk_blocks: 1,
            cand_block: 1,
            eps: 1e-6,
            index_k_every_step: every_step,
        };
        let mut own = vec![3.0f32; 2 * ihd];
        let mut shared = SharedRuntime {
            index_k: vec![7.0f32; 2 * ihd],
            ..SharedRuntime::default()
        };
        indexer(
            &x,
            &qr,
            None,
            1,
            5,
            0,
            ratio,
            &c,
            &w,
            &fc,
            Some(&mut own),
            &mut shared,
            false,
            false,
        );
        (shared.index_k, own)
    };
    let (published, own) = run(true);
    assert_eq!(published, own);
    let (stale, _) = run(false);
    assert_eq!(
        stale,
        vec![7.0f32; 2 * ihd],
        "control: the old revision does not publish"
    );
}
