// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `StatePool`: a reused drafter state starts empty and keeps its
//! buffers and region; the freed handle cannot release them again.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::Glm5NextDsaConfig;

/// 2026-10-14: GLM-5.3's DSA shape (the same values as `glm5next_mtp_head_tests::cfg`).
fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// 2026-10-14: A state mid-sequence: 300 rows written, 2 drafts outstanding, region 3. The
/// DSA counter is moved with `adopt_kv_rows`, which needs a paged state and no device memory.
fn live(cfg: &Glm5NextDsaConfig) -> Glm5NextMtpProposerState {
    let mut dsa = Glm5NextDsaState::paged(cfg).unwrap();
    dsa.adopt_kv_rows(300).unwrap();
    Glm5NextMtpProposerState {
        dsa,
        seq_len: 300,
        block_table: vec![12, 13, 14],
        last_drafted: 2,
        concat: DevicePtr(0x1000),
        x: DevicePtr(0x2000),
        logits: DevicePtr(0x3000),
        arg: DevicePtr(0x4000),
        head_xchg: DevicePtr(0x5000),
        released: false,
        kv_region: Some(3),
        graph: None,
    }
}

/// 2026-10-14: The taken state is the kept one (same buffers, region, table) at zero rows on
/// every counter; skipping any reset in `take` fails here.
#[test]
fn a_taken_state_is_the_kept_one_at_zero_rows() {
    let c = cfg();
    let pool = StatePool::new();
    let mut st = live(&c);
    pool.keep(&c, &mut st).unwrap();
    let got = pool.take().unwrap().expect("one state was kept");
    assert_eq!((got.seq_len, got.last_drafted, got.dsa.len()), (0, 0, 0));
    assert!(!got.released, "a taken state must be live for free_state");
    assert_eq!(got.kv_region, Some(3));
    assert_eq!(got.block_table, vec![12, 13, 14]);
    assert_eq!(
        [got.concat, got.x, got.logits, got.arg, got.head_xchg].map(|p| p.0),
        [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
    );
    assert!(
        pool.take().unwrap().is_none(),
        "a state was handed out twice"
    );
}

/// 2026-10-14: The freed handle keeps nothing that a second `free_state` could release: no
/// region, no graph, no table, a DSA state that owns no memory, and the released flag set.
#[test]
fn the_freed_handle_owns_nothing_after_keep() {
    let c = cfg();
    let pool = StatePool::new();
    let mut st = live(&c);
    pool.keep(&c, &mut st).unwrap();
    assert!(st.released);
    assert_eq!(st.kv_region, None);
    assert!(st.graph.is_none() && st.block_table.is_empty());
    assert_eq!(
        st.dsa.cache(),
        crate::glm5next_dsa::paged::IndexerCache::Paged
    );
    assert_eq!(st.seq_len, 0);
}

/// 2026-10-14: An empty pool hands out nothing; the most recently kept state comes back first.
#[test]
fn an_empty_pool_hands_out_nothing_and_reuse_is_last_in_first_out() {
    let c = cfg();
    let pool = StatePool::new();
    assert!(pool.take().unwrap().is_none());
    let (mut a, mut b) = (live(&c), live(&c));
    b.kv_region = Some(5);
    pool.keep(&c, &mut a).unwrap();
    pool.keep(&c, &mut b).unwrap();
    assert_eq!(pool.take().unwrap().unwrap().kv_region, Some(5));
    assert_eq!(pool.take().unwrap().unwrap().kv_region, Some(3));
}
