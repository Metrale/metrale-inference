// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Tests of `tp_split` and the TP head division, without a GPU.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use super::*;
use crate::parse_config;

/// 2026-10-08: The GLM-5.3-Flash checkpoint config fixture, the geometry uneven TP exists for.
const GLM_CONFIG: &str =
    include_str!("../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn slices(total: usize, tp: usize, align: usize) -> Result<Vec<TpSlice>> {
    (0..tp).map(|r| tp_split(total, tp, r, align)).collect()
}

/// 2026-10-08: The partition invariant: contiguous from 0, in rank order, each slice non-empty
/// and aligned, ending at `total`, widest minus narrowest at most one unit.
fn assert_partition(total: usize, tp: usize, align: usize, s: &[TpSlice]) {
    let ctx = format!("total={total} tp={tp} align={align}: {s:?}");
    assert_eq!(s.len(), tp, "{ctx}");
    let mut next = 0;
    for x in s {
        assert_eq!(x.start, next, "contiguous, in rank order: {ctx}");
        assert!(x.len > 0, "non-empty: {ctx}");
        if tp > 1 {
            assert_eq!(x.start % align, 0, "aligned start: {ctx}");
            assert_eq!(x.len % align, 0, "aligned length: {ctx}");
        }
        next = x.end();
    }
    assert_eq!(next, total, "covers the dimension: {ctx}");
    let (lo, hi) = (
        s.iter().map(|x| x.len).min().unwrap(),
        s.iter().map(|x| x.len).max().unwrap(),
    );
    assert!(hi - lo <= align, "balanced to one unit: {ctx}");
    assert!(
        s.windows(2).all(|w| w[0].len >= w[1].len),
        "the wider ranks come first: {ctx}"
    );
}

/// 2026-10-08: Wherever `total` divides into `tp * align`, every rank gets exactly the range the
/// even split gave: `[r * total / tp, (r + 1) * total / tp)`.
#[test]
fn an_even_geometry_reproduces_the_even_split() {
    for tp in [1usize, 2, 4, 8] {
        for align in [1usize, 8, 16] {
            for total in [64usize, 2048, 12288, 8192, 154_880, 32_768] {
                if !total.is_multiple_of(tp * align) {
                    continue;
                }
                for r in 0..tp {
                    let s = tp_split(total, tp, r, align).unwrap();
                    assert_eq!(
                        (s.start, s.end()),
                        (r * total / tp, (r + 1) * total / tp),
                        "total={total} tp={tp} r={r} align={align}"
                    );
                }
            }
        }
    }
}

/// 2026-10-08: GLM-5.3's 64 heads over three ranks: 22, 21, 21, starting at 0, 22, 43.
#[test]
fn sixty_four_heads_over_three_ranks() {
    let s = slices(64, 3, 1).unwrap();
    assert_eq!(
        s,
        vec![
            TpSlice { start: 0, len: 22 },
            TpSlice { start: 22, len: 21 },
            TpSlice { start: 43, len: 21 },
        ]
    );
}

/// 2026-10-08: The 2048-wide shared expert in 8-element units (256 units): 86, 85, 85 units,
/// so 688, 680, 680 columns. A one-element split (683, 683, 682) would break the BF16 GEMV's
/// `K % 8 == 0` rule on every rank.
#[test]
fn the_shared_expert_width_over_three_ranks_in_eight_element_units() {
    let s = slices(2048, 3, 8).unwrap();
    assert_eq!(
        s.iter().map(|x| (x.start, x.len)).collect::<Vec<_>>(),
        vec![(0, 688), (688, 680), (1368, 680)]
    );
    assert!(s.iter().all(|x| x.len % 8 == 0 && x.start % 8 == 0));
}

/// 2026-10-08: Every (total, tp, align) in a small cube either partitions or is refused, and a
/// refusal happens exactly when the alignment does not divide or a rank would get nothing.
#[test]
fn every_small_geometry_partitions_or_is_refused_for_a_stated_reason() {
    for tp in 1..=9usize {
        for align in 1..=9usize {
            for total in 0..=200usize {
                let got = slices(total, tp, align);
                let refusable = tp > 1 && (!total.is_multiple_of(align) || total / align < tp);
                match got {
                    Ok(s) if !refusable => {
                        if tp == 1 {
                            assert_eq!(
                                s,
                                vec![TpSlice {
                                    start: 0,
                                    len: total
                                }]
                            );
                        } else {
                            assert_partition(total, tp, align, &s);
                        }
                    }
                    Ok(s) => panic!("total={total} tp={tp} align={align} accepted: {s:?}"),
                    Err(e) => assert!(refusable, "total={total} tp={tp} align={align}: {e}"),
                }
            }
        }
    }
}

/// 2026-10-08: The argument refusals, each with its reason.
#[test]
fn degenerate_arguments_are_refused() {
    let msg = |r: Result<TpSlice>| r.unwrap_err().to_string();
    assert!(msg(tp_split(64, 0, 0, 1)).contains("tp_size is 0"));
    assert!(msg(tp_split(64, 3, 3, 1)).contains(">= tp_size"));
    assert!(msg(tp_split(64, 3, 0, 0)).contains("align is 0"));
    assert!(msg(tp_split(2047, 3, 0, 8)).contains("not a multiple"));
    assert!(msg(tp_split(16, 3, 0, 8)).contains("would own nothing"));
}

/// 2026-10-08: One rank slices nothing, so an unaligned width is still returned whole.
#[test]
fn one_rank_takes_the_whole_dimension_at_any_alignment() {
    assert_eq!(
        tp_split(2047, 1, 0, 8).unwrap(),
        TpSlice {
            start: 0,
            len: 2047
        }
    );
}

fn glm_at(tp: usize, rank: usize) -> ModelConfig {
    let mut c = parse_config(GLM_CONFIG).expect("the real checkpoint config parses");
    c.tp_world_size = tp;
    c.tp_rank = rank;
    c
}

/// 2026-10-08: GLM-5.3 at TP=3, uneven: every head family is 22/21/21 and the pre-shard counts
/// are recorded on every rank.
#[test]
fn glm_heads_split_unevenly_at_tp3() {
    let want = [22usize, 21, 21];
    for (rank, w) in want.iter().enumerate() {
        let mut c = glm_at(3, rank);
        c.shard_heads_for_tp(TpSupport::Uneven {
            linear_channel_unit: 1,
        })
        .unwrap();
        assert_eq!(
            (
                c.num_attention_heads,
                c.num_key_value_heads,
                c.linear_num_key_heads,
                c.linear_num_value_heads
            ),
            (*w, *w, *w, *w),
            "rank {rank}"
        );
        assert_eq!(
            c.pre_shard_heads().unwrap(),
            TpPreShardHeads {
                num_attention_heads: 64,
                num_key_value_heads: 64,
                linear_num_key_heads: 64,
                linear_num_value_heads: 64,
            }
        );
    }
}

/// 2026-10-08: `Even` refuses TP=3 with the message serve's topology always printed.
#[test]
fn even_support_refuses_tp3_as_before() {
    let mut c = glm_at(3, 0);
    let err = c.shard_heads_for_tp(TpSupport::Even).unwrap_err();
    assert_eq!(
        err.to_string(),
        "TP requires num_attention_heads (64) divisible by tp_size (3)"
    );
    assert_eq!(
        c.num_attention_heads, 64,
        "a refusal leaves the counts whole"
    );
    assert!(c.tp_pre_shard_heads.is_none());
}

/// 2026-10-08: At TP=2 the two policies give identical counts, so a loader that opts in to
/// uneven splits shards an even geometry exactly as before.
#[test]
fn even_and_uneven_agree_at_tp2() {
    for rank in 0..2 {
        let (mut e, mut u) = (glm_at(2, rank), glm_at(2, rank));
        e.shard_heads_for_tp(TpSupport::Even).unwrap();
        u.shard_heads_for_tp(TpSupport::Uneven {
            linear_channel_unit: 1,
        })
        .unwrap();
        assert_eq!(e.num_attention_heads, 32);
        assert_eq!(
            (
                e.num_attention_heads,
                e.num_key_value_heads,
                e.linear_num_key_heads,
                e.linear_num_value_heads
            ),
            (
                u.num_attention_heads,
                u.num_key_value_heads,
                u.linear_num_key_heads,
                u.linear_num_value_heads
            )
        );
    }
}

/// 2026-10-08: A second division would shard the shards; it is refused. One rank is a no-op,
/// and an unsupported loader is refused above one rank.
#[test]
fn the_division_runs_once_and_only_where_supported() {
    let mut c = glm_at(3, 1);
    c.shard_heads_for_tp(TpSupport::Uneven {
        linear_channel_unit: 1,
    })
    .unwrap();
    let err = c
        .shard_heads_for_tp(TpSupport::Uneven {
            linear_channel_unit: 1,
        })
        .unwrap_err();
    assert!(err.to_string().contains("already divided"), "{err}");
    assert_eq!(c.num_attention_heads, 21);

    let mut one = glm_at(1, 0);
    one.shard_heads_for_tp(TpSupport::Unsupported).unwrap();
    assert_eq!(one.num_attention_heads, 64);
    assert_eq!(one.pre_shard_heads().unwrap().num_attention_heads, 64);

    let err = glm_at(2, 0)
        .shard_heads_for_tp(TpSupport::Unsupported)
        .unwrap_err();
    assert!(err.to_string().contains("does not support TP"), "{err}");
}

/// 2026-10-08: Above one rank the pre-shard counts exist only once the division ran; the
/// per-rank counts alone cannot say whether 21 came from 63 or 64.
#[test]
fn pre_shard_heads_refuse_an_undivided_multi_rank_config() {
    let err = glm_at(3, 0).pre_shard_heads().unwrap_err();
    assert!(err.to_string().contains("never divided"), "{err}");
}

/// 2026-10-08: Grouped heads split in whole groups: 32 query heads over 8 KV heads at TP=3 give
/// KV 3/3/2 and query 12/12/8, four per KV head on every rank. Query heads that are not whole
/// groups are refused.
#[test]
fn grouped_heads_split_in_whole_groups() {
    let mut got = Vec::new();
    for rank in 0..3 {
        let mut c = glm_at(3, rank);
        c.num_attention_heads = 32;
        c.num_key_value_heads = 8;
        c.shard_heads_for_tp(TpSupport::Uneven {
            linear_channel_unit: 1,
        })
        .unwrap();
        assert_eq!(c.num_attention_heads, 4 * c.num_key_value_heads);
        got.push((c.num_key_value_heads, c.num_attention_heads));
    }
    assert_eq!(got, vec![(3, 12), (3, 12), (2, 8)]);

    let mut c = glm_at(3, 0);
    c.num_attention_heads = 30;
    c.num_key_value_heads = 8;
    let err = c
        .shard_heads_for_tp(TpSupport::Uneven {
            linear_channel_unit: 1,
        })
        .unwrap_err();
    assert!(err.to_string().contains("whole groups"), "{err}");
}

/// 2026-10-09: With a 256-channel linear unit GLM-5.3's 128-wide KDA heads split in pairs at TP=3
/// (22/22/20, every rank a multiple of 256 channels) while its attention heads stay 22/21/21;
/// a unit of 1 is the head-by-head split. The widest rank keeps 22 heads either way.
#[test]
fn a_linear_channel_unit_splits_kda_heads_in_whole_units() {
    let split = |unit: usize| -> Vec<(usize, usize, usize)> {
        (0..3)
            .map(|rank| {
                let mut c = glm_at(3, rank);
                c.shard_heads_for_tp(TpSupport::Uneven {
                    linear_channel_unit: unit,
                })
                .unwrap();
                (
                    c.num_attention_heads,
                    c.linear_num_key_heads,
                    c.linear_num_value_heads,
                )
            })
            .collect()
    };
    assert_eq!(split(256), vec![(22, 22, 22), (21, 22, 22), (21, 20, 20)]);
    assert_eq!(split(1), vec![(22, 22, 22), (21, 21, 21), (21, 21, 21)]);
    assert!(
        split(256)
            .iter()
            .all(|&(_, lk, _)| (lk * 128).is_multiple_of(256))
    );

    assert_eq!(linear_head_unit(128, 256).unwrap(), 2);
    assert_eq!(linear_head_unit(128, 1).unwrap(), 1);
    assert_eq!(linear_head_unit(128, 128).unwrap(), 1);
    assert_eq!(linear_head_unit(96, 256).unwrap(), 8);
    assert_eq!(linear_head_unit(0, 1).unwrap(), 1);
    assert!(linear_head_unit(128, 0).is_err());
}
