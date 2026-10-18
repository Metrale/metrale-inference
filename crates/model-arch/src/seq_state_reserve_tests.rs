// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Exact-integer pins for the per-sequence reserve, from the GLM-5.3
//! dimensions. No GPU and no checkpoint.
//!
//! Owner: model-arch (serve memory reserve).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::state::{dsa_capacity, indexer_state_bytes};

/// 2026-09-25: GLM-5.3, from the checked-in config fixture
/// (`crates/model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json`):
/// `index_head_dim = 128`, `index_kpool = 4`, `hidden_size = 4096`,
/// `vocab_size = 154880`, 11 `deepseek_sparse_attention` layers.
const HEAD_DIM: usize = 128;
const KPOOL: usize = 4;
const HIDDEN: usize = 4096;
const VOCAB: usize = 154_880;
const DSA_LAYERS: usize = 11;
const MSL: usize = 131_072;

/// 2026-09-25: One KV block over the 11 DSA layers for a 1-byte (FP8) cache of
/// `kv_lora_rank` 512 wide rows: `2 (k + v) * block_size 16 * num_kv_heads 1 * 512`
/// = 16,384 B per layer per block.
const KV_BLOCK_BYTES: usize = (2 * 16) * 512 * DSA_LAYERS;

#[test]
fn the_kv_block_is_180_224_bytes_exactly() {
    assert_eq!(KV_BLOCK_BYTES, 180_224);
}

#[test]
fn indexer_bytes_are_513_per_token_per_layer() {
    assert_eq!(indexer_state_bytes(1, HEAD_DIM), 4 * HEAD_DIM + 1);
    assert_eq!(indexer_state_bytes(1, HEAD_DIM), 513);
    assert_eq!(indexer_state_bytes(MSL, HEAD_DIM), 67_239_936);
}

#[test]
fn capacity_rounds_down_to_whole_pools_and_is_the_only_spelling() {
    // 2026-09-25: Exact multiples are unchanged.
    assert_eq!(dsa_capacity(MSL, KPOOL), MSL);
    assert_eq!(dsa_capacity(65_536, KPOOL), 65_536);
    // 2026-09-25: A context that does not divide evenly rounds down to whole pools.
    assert_eq!(dsa_capacity(4_099, KPOOL), 4_096);
    assert_eq!(dsa_capacity(3, KPOOL), 0);
    // 2026-09-25: A kpool of 0 is treated as 1.
    assert_eq!(dsa_capacity(MSL, 0), MSL);
}

/// 2026-09-25: The reserve must equal what `Glm5NextDsaState::alloc` takes: three
/// allocations, `capacity*d*2`, `capacity*d*2`, `capacity`.
#[test]
fn the_reserve_equals_what_dsa_alloc_actually_allocates() {
    let cap = dsa_capacity(MSL, KPOOL);
    let what_alloc_takes = cap * HEAD_DIM * 2 + cap * HEAD_DIM * 2 + cap;
    assert_eq!(indexer_state_bytes(cap, HEAD_DIM), what_alloc_takes);
}

#[test]
fn target_layer_charge_at_131072_is_exact() {
    let per_layer = indexer_state_bytes(dsa_capacity(MSL, KPOOL), HEAD_DIM);
    assert_eq!(DSA_LAYERS * per_layer, 739_639_296);
}

#[test]
fn proposer_charge_at_131072_is_exact_and_is_a_separate_owner() {
    let per_layer = indexer_state_bytes(dsa_capacity(MSL, KPOOL), HEAD_DIM);
    let proposer = per_layer + 2 * HIDDEN * 2 + HIDDEN * 2 + VOCAB * 2 + 8 + 48;
    assert_eq!(proposer, 67_574_328);
    // 2026-09-25: The proposer owns one indexer block plus five small buffers (334,392 B; 2026-10-11: the
    // exchange buffer is 48 B, four ranks of the log-probability layout, and `arg` is 8 B).
    assert_eq!(proposer - per_layer, 334_392);
    assert_eq!(12 * per_layer, 806_879_232);
    assert_ne!(
        proposer,
        12 * per_layer - DSA_LAYERS * per_layer + 334_392 + 1
    );
}

#[test]
fn total_per_sequence_and_the_batch_3_charge_are_exact() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    assert_eq!(s.total(), 807_213_588);
    assert_eq!(s.for_batch(3), 2_421_640_764);
    assert_eq!(s.for_batch(1), s.total());
    assert_eq!(
        s.for_batch(0),
        s.total(),
        "batch 0 is clamped to 1, never zero-charged"
    );
}

/// 2026-09-25: The batch-3 charge in `KV_BLOCK_BYTES` blocks.
#[test]
fn the_batch_3_charge_converts_to_exactly_13_436_kv_blocks() {
    let charge = 2_421_640_764usize;
    assert_eq!(charge / KV_BLOCK_BYTES, 13_436);
    let reachable = 3 * MSL.div_ceil(16) + 3 + 1;
    assert_eq!(reachable, 24_580);
    assert_eq!(reachable + charge / KV_BLOCK_BYTES, 38_016);
}

#[test]
fn the_source_derived_block_size_reproduces_both_measured_clamp_lines() {
    let gib = |b: usize| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    let freed_131k = (46_887 - 8_194) * KV_BLOCK_BYTES;
    let freed_65k = (47_537 - 4_098) * KV_BLOCK_BYTES;
    assert_eq!(format!("{:.2}", gib(freed_131k)), "6.49");
    assert_eq!(format!("{:.2}", gib(freed_65k)), "7.29");
}

#[test]
fn a_non_glm_config_is_charged_nothing() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_ne!(cfg.model_type, "glm5_next");
    let s = per_sequence_state_bytes(
        &cfg,
        MSL,
        true,
        &metrale_model_layers::lever_equivalence::defaults(),
    )
    .expect("non-GLM is inert, not an error");
    assert_eq!(s, PerSequenceState::default());
    assert_eq!(s.total(), 0);
}

const GLM_CONFIG: &str =
    include_str!("../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

/// 2026-10-11: One sequence's drafter KV region: the fixture leaves `serve_max_seq_len` unset,
/// so the DSA context is 16,384 rows; 16,384 / 16 + 2 = 1,026 blocks of
/// `2 (k + v) * 16 * 512` B (FP8 latent, one layer) = 16,384 B each.
#[test]
fn a_drafter_kv_region_is_1026_blocks_of_16_kib() {
    let cfg = metrale_config::parse_config(GLM_CONFIG).expect("real config parses");
    assert_eq!(
        crate::glm5next_mtp_head::drafter_kv_bytes_per_seq(&cfg, MSL, 0).unwrap(),
        1026 * 16_384
    );
    assert_eq!(1026 * 16_384, 16_809_984);
}

/// 2026-10-09: The flat indexer is charged per sequence (11 layers at 513 B per token); the
/// paged one lives in the KV pool and is charged nothing per sequence. The MTP proposer's own
/// flat indexer is charged either way.
#[test]
fn a_paged_text_indexer_is_not_charged_per_sequence() {
    let cfg = metrale_config::parse_config(GLM_CONFIG).expect("real config parses");
    let flat = per_sequence_state_bytes_for(&cfg, MSL, true, IndexerCache::Flat, 0, false).unwrap();
    let paged =
        per_sequence_state_bytes_for(&cfg, MSL, true, IndexerCache::Paged, 0, false).unwrap();
    assert_eq!(flat.target_layers, 739_639_296);
    assert_eq!(paged.target_layers, 0);
    assert_eq!(paged.proposer, flat.proposer);
    // 2026-10-11: The indexer block and buffers (67,574,328 B) plus the drafter KV region
    // (16,809,984 B).
    assert_eq!(paged.proposer, 67_574_328 + 16_809_984);
}

/// 2026-10-13: The proposer's charge at `--max-seq-len` `msl` (set as the serve sets it) under
/// drafter cap `cap`, with the paged text indexer of the lever recipe.
fn proposer_at(msl: usize, cap: usize) -> usize {
    let mut cfg = metrale_config::parse_config(GLM_CONFIG).expect("real config parses");
    cfg.serve_max_seq_len = msl;
    per_sequence_state_bytes_for(&cfg, msl, true, IndexerCache::Paged, cap, false)
        .unwrap()
        .proposer
}

/// 2026-10-13: Under the drafter cap the proposer's charge is the same at every
/// `--max-seq-len` at or above it: 65,536 rows of indexer (33,619,968 B) and 4,098 KV blocks
/// (67,141,632 B) plus the 334,392 B of buffers. Uncapped (the negative control) it grows with
/// `--max-seq-len`, ~1,537 B per row, which is what failed the 512k boot.
#[test]
fn the_capped_proposer_charge_does_not_grow_with_max_seq_len() {
    let at_cap = proposer_at(65_536, 65_536);
    assert_eq!(at_cap, 33_619_968 + 4_098 * 16_384 + 334_392);
    for msl in [131_072, 262_144, 524_288, 1_048_576] {
        assert_eq!(proposer_at(msl, 65_536), at_cap, "msl {msl}");
        assert!(proposer_at(msl, 0) > at_cap, "uncapped msl {msl} must grow");
    }
    assert_eq!(
        proposer_at(524_288, 0) - proposer_at(262_144, 0),
        262_144 * 513 + (262_144 / 16) * 16_384
    );
}

/// 2026-10-13: At or below the cap the charge is the uncapped one, byte for byte (the lever
/// recipe's 40,960 included).
#[test]
fn below_the_cap_the_charge_is_the_uncapped_one() {
    for msl in [16_384, 40_960, 65_536] {
        assert_eq!(proposer_at(msl, 65_536), proposer_at(msl, 0), "msl {msl}");
    }
}
