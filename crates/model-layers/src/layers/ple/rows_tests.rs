// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests for the PLE host half shared by every entry point
//! (`rows.rs` `next_ids`): the ids are the reference's tail slice of the hashed
//! window, the history advances by exactly the new tokens, and a batch of
//! single-token rows hashes as the rows would one by one.
//!
//! Owner: model-layers (PLE).
//! Invariants: none beyond the types.

use super::super::ids::{PleIdDims, ple_ngram_ids};
use super::rows::next_ids;

/// 2026-10-13: A small valid geometry: 3-grams, two heads per order (four heads),
/// odd multipliers, prime moduli, disjoint offsets. `validate` accepts it.
fn dims() -> PleIdDims {
    let d = PleIdDims {
        ngram_size: 3,
        heads_per_ngram: 2,
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15,
            0xBF58_476D_1CE4_E5B9,
            0x94D0_49BB_1331_11EB,
        ],
        head_vocab_sizes: vec![97, 101, 103, 107],
        head_offsets: vec![0, 97, 198, 301],
        eos_token_id: 2,
    };
    d.validate().expect("valid geometry");
    d
}

#[test]
fn ids_are_the_tail_of_the_hashed_window_and_the_history_advances() {
    let d = dims();
    let mut history = vec![11u32, 12];
    let tokens = [13u32, 14, 15];
    let flat = next_ids(&d, &mut history, &tokens);

    let window = [11u32, 12, 13, 14, 15];
    let all = ple_ngram_ids(&d, &window);
    let want: Vec<u64> = all[all.len() - 3..].iter().flatten().copied().collect();
    assert_eq!(flat, want);
    assert_eq!(flat.len(), 3 * d.ngram_heads());
    assert_eq!(history, vec![14, 15], "the last context_len tokens");
}

/// 2026-10-13: What `forward_rows` relies on: hashing each row's one token
/// against its own history gives, per row, exactly what that sequence's
/// single-token step gives, and a multi-token step equals the chain of
/// single-token steps on the same history.
#[test]
fn single_token_steps_compose_like_one_multi_token_step() {
    let d = dims();
    let tokens = [21u32, 22, 23, 24];

    let mut h_multi = vec![5u32, 6];
    let multi = next_ids(&d, &mut h_multi, &tokens);

    let mut h_steps = vec![5u32, 6];
    let mut steps = Vec::new();
    for &t in &tokens {
        steps.extend(next_ids(&d, &mut h_steps, &[t]));
    }
    assert_eq!(steps, multi);
    assert_eq!(h_steps, h_multi);
}

#[test]
fn different_histories_give_different_ids_for_the_same_token() {
    let d = dims();
    let mut a = vec![1u32, 2];
    let mut b = vec![3u32, 4];
    let ia = next_ids(&d, &mut a, &[9]);
    let ib = next_ids(&d, &mut b, &[9]);
    assert_ne!(ia, ib, "the n-gram ids depend on the carried history");
    assert_eq!(a, vec![2, 9]);
    assert_eq!(b, vec![4, 9]);
}
