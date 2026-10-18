// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The FP8 indexer layout and its ring: sizes, offsets, and which rewinds keep
//! every row a reopened pool needs.

use std::collections::HashMap;

use super::*;

/// 2026-10-16: The ring needs `S` sets for a rewind of `R` rows, and no fewer: `tail_sets_for`
/// equals the most pools any rewind of at most `R` rows reopens with rows to keep, found by
/// enumeration.
#[test]
fn tail_sets_is_the_fewest_that_cover_every_rewind() {
    for kp in [2usize, 4, 8] {
        for r in 0usize..=24 {
            let mut need = 1usize;
            for len in 1..=(4 * kp + 2 * r + 8) {
                for n in len.saturating_sub(r)..len {
                    if !n.is_multiple_of(kp) {
                        need = need.max((len - 1) / kp - n / kp + 1);
                    }
                }
            }
            assert_eq!(tail_sets_for(r, kp).unwrap(), need, "kp {kp} R {r}");
        }
    }
    assert_eq!(tail_sets_for(1, 4).unwrap(), 1);
    assert_eq!(tail_sets_for(2, 4).unwrap(), 2);
    assert_eq!(tail_sets_for(8, 4).unwrap(), 3);
    assert!(tail_sets_for(1, 1).is_err());
}

/// 2026-10-16: A model of the ring: tokens 0..len stored one call at a time, each open-pool
/// token into its slot of its block. A row is intact when every row of the pool `n` reopens,
/// below `n`, is still in its slot. `rewind_keeps_tail` must never accept a clobbered tail
/// (every block size); with one block (bs 64 covers every len) it is exact. With bs 16 it may
/// refuse a rewind whose newer pool landed in the next block: conservative, never unsafe. The
/// refused cases are the control: they show the model detects a clobbered row.
#[test]
fn rewind_keeps_tail_matches_a_ring_model() {
    let kp = 4usize;
    let mut refused = 0usize;
    for bs in [16usize, 64] {
        for sets in 1usize..=3 {
            let lay = Fp8IndexerLayout::new(
                bs,
                Fp8IndexerLayout::block_bytes(bs, 128, kp, sets).unwrap(),
                128,
                kp,
                sets,
            )
            .unwrap();
            for len in 1usize..=64 {
                let mut ring: HashMap<(usize, usize), usize> = HashMap::new();
                for x in 0..len {
                    if x % kp != kp - 1 {
                        ring.insert((x / bs, lay.ring_key_off(x)), x);
                    }
                }
                for n in 0..len {
                    let intact = (n - n % kp..n)
                        .all(|x| ring.get(&(x / bs, lay.ring_key_off(x))) == Some(&x));
                    let keeps = rewind_keeps_tail(len, n, kp, sets);
                    assert!(!keeps || intact, "unsafe: bs {bs} S {sets} {len}->{n}");
                    if bs >= len {
                        assert_eq!(keeps, intact, "bs {bs} S {sets} {len}->{n}");
                        refused += usize::from(!intact);
                    }
                }
            }
        }
    }
    assert!(
        refused > 0,
        "the model never clobbered a row: the check is vacuous"
    );
}

/// 2026-10-16: The bytes per block: 2064 at the serve's block of 16 with one ring set.
#[test]
fn block_bytes_formula() {
    assert_eq!(Fp8IndexerLayout::block_bytes(16, 128, 4, 1).unwrap(), 2064);
    assert_eq!(
        Fp8IndexerLayout::block_bytes(16, 128, 4, 2).unwrap(),
        2064 + 1536
    );
    assert_eq!(
        Fp8IndexerLayout::block_bytes(64, 128, 4, 1).unwrap(),
        16 * 132 + 1536
    );
    let fmt = IndexerRowFormat::Fp8 { tail_sets: 1 };
    assert_eq!(fmt.v_block_bytes(16, 128, 4).unwrap(), Some(2064));
    assert_eq!(
        IndexerRowFormat::Bf16.v_block_bytes(16, 128, 4).unwrap(),
        None
    );
}

/// 2026-10-16: The kernels' geometry limits are refused, each naming its cause.
#[test]
fn unsupported_geometry_is_refused() {
    let e = Fp8IndexerLayout::block_bytes(16, 64, 4, 1).unwrap_err();
    assert!(e.to_string().contains("compiled for 128"), "{e}");
    let e = Fp8IndexerLayout::block_bytes(18, 128, 4, 1).unwrap_err();
    assert!(e.to_string().contains("multiple of index_kpool"), "{e}");
    let e = Fp8IndexerLayout::new(16, 2060, 128, 4, 1).unwrap_err();
    assert!(e.to_string().contains("need 2064 B"), "{e}");
    let e = Fp8IndexerLayout::new(16, 2066, 128, 4, 1).unwrap_err();
    assert!(e.to_string().contains("multiple of 4"), "{e}");
    assert!(Fp8IndexerLayout::new(16, 2068, 128, 4, 1).is_ok());
}

/// 2026-10-16: Every token's entry, scale and ring rows fall inside the block, the regions of
/// distinct pools and slots never overlap, scales are 4-aligned, and together they tile the
/// block exactly.
#[test]
fn offsets_tile_the_block() {
    for (bs, kp, sets) in [
        (16usize, 4usize, 1usize),
        (16, 4, 2),
        (64, 4, 3),
        (16, 2, 1),
        (32, 8, 2),
    ] {
        let bytes = Fp8IndexerLayout::block_bytes(bs, 128, kp, sets).unwrap();
        let lay = Fp8IndexerLayout::new(bs, bytes, 128, kp, sets).unwrap();
        let mut owner: Vec<Option<(char, usize)>> = vec![None; bytes];
        let mut claim = |off: usize, len: usize, who: (char, usize)| {
            assert!(off + len <= bytes, "{who:?} at {off}+{len} past {bytes}");
            for b in &mut owner[off..off + len] {
                assert!(b.is_none() || *b == Some(who), "{who:?} overlaps {b:?}");
                *b = Some(who);
            }
        };
        // 2026-10-16: Two blocks of tokens and more than S pools, so set reuse is exercised.
        for x in bs..3 * bs {
            let pool = (x % bs) / kp;
            assert_eq!(lay.scale_off(x) % 4, 0);
            claim(lay.entry_off(x), 128, ('e', pool));
            claim(lay.scale_off(x), 4, ('s', pool));
            if x % kp != kp - 1 {
                let slot = ((x / kp) % sets) * (kp - 1) + x % kp;
                claim(lay.ring_key_off(x), 256, ('k', slot));
                claim(lay.ring_gate_off(x), 256, ('g', slot));
            }
        }
        assert!(
            owner.iter().all(Option::is_some),
            "bs {bs} S {sets}: unused bytes"
        );
    }
}

/// 2026-10-16: The V block override the flag gives a GLM-shaped text stack (`index_head_dim`
/// 128, `index_kpool` 4) for a rewind of `rewind` rows, through the loader's path.
fn flag_v_bytes(dtype: IndexerCacheDtype, rewind: usize, block_size: usize) -> Option<usize> {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.index_head_dim = 128;
    config.index_kpool = 4;
    config.indexer_cache_dtype = dtype;
    config.indexer_rewind_rows = rewind;
    IndexerRowFormat::for_text_stack(&config, IndexerCache::Paged)
        .unwrap()
        .v_block_bytes(block_size, 128, 4)
        .unwrap()
}

/// 2026-10-16: The planner's gain at a fixed byte budget, for 11 DSA layers with a 512-wide
/// FP8 latent on the K side and the indexer rows on the V side. BF16 is the control: no
/// override, so the plan is exactly the one before the flag (8192 B of V per block of 16, the
/// key and gate rows). FP8 with one ring set holds 1.60x the tokens, two sets (a rewind of 2
/// to 5 rows) 1.39x, and blocks of 64 1.80x.
#[test]
fn fp8_indexer_rows_buy_tokens_at_a_fixed_budget() {
    use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
    let kv = |block_size: usize, v_block_bytes: Option<usize>| KvCacheConfig {
        block_size,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 11,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_block_bytes,
    };
    let budget = 64usize << 30;
    let tokens =
        |c: &KvCacheConfig| PagedKvCache::compute_num_blocks(c, budget).unwrap() * c.block_size;
    let bytes_per_token = |c: &KvCacheConfig| c.block_bytes_kv_all_layers() / c.block_size;

    let bf16 = kv(16, flag_v_bytes(IndexerCacheDtype::Bf16, 3, 16));
    assert_eq!(bf16.v_block_bytes, None);
    assert_eq!(bytes_per_token(&bf16), 11 * (512 + 512));
    assert_eq!(tokens(&bf16), tokens(&kv(16, None)));

    let mut gains = Vec::new();
    for (rewind, bs, per_token) in [
        (1usize, 16usize, 7_051usize),
        (2, 16, 8_107),
        (5, 16, 8_107),
        (1, 64, 6_259),
    ] {
        let fp8 = kv(bs, flag_v_bytes(IndexerCacheDtype::Fp8, rewind, bs));
        assert_eq!(bytes_per_token(&fp8), per_token, "R {rewind} bs {bs}");
        let base = tokens(&kv(bs, None));
        let got = tokens(&fp8);
        assert_eq!(got, budget / (per_token * bs) * bs);
        gains.push(got * 1000 / base);
    }
    assert_eq!(gains, vec![1597, 1389, 1389, 1799]);
}

/// 2026-10-16: FP8 rows live in the KV blocks, so the flat cache refuses the flag.
#[test]
fn fp8_needs_the_paged_cache() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.index_kpool = 4;
    config.indexer_cache_dtype = IndexerCacheDtype::Fp8;
    let e = IndexerRowFormat::for_text_stack(&config, IndexerCache::Flat).unwrap_err();
    assert!(
        e.to_string().contains("needs the paged indexer cache"),
        "{e}"
    );
    config.indexer_cache_dtype = IndexerCacheDtype::Bf16;
    assert_eq!(
        IndexerRowFormat::for_text_stack(&config, IndexerCache::Flat).unwrap(),
        IndexerRowFormat::Bf16
    );
}
