// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `forward_block`: the rt2 vocab GEMV guard.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::rt2_16_covers_the_batch;

/// 2026-09-25: The guard takes the total row count: eligible exactly for 1 to 16
/// rows, the range `ops::fp8_gemv_rowscale_batch16_rt2` accepts.
#[test]
fn the_rt2_vocab_kernel_is_only_eligible_for_batches_it_covers() {
    for gamma in 1..=16u32 {
        assert!(rt2_16_covers_the_batch(gamma), "n_seq=1, gamma={gamma}");
    }
    assert!(!rt2_16_covers_the_batch(10 * 2));
    assert!(rt2_16_covers_the_batch(16));
    assert!(!rt2_16_covers_the_batch(17));
    assert!(!rt2_16_covers_the_batch(0));
    assert!(!rt2_16_covers_the_batch(8 * 8));
}

/// 2026-10-09: The window argument masks exactly the keys `window` positions behind the query:
/// with the ctx starting at position 0 (slot index = position) it is the window itself, and
/// after the ctx slid by `p0` positions it grows by `p0`, the gap between a key's slot index
/// and its position. No window gives 0, which the kernel reads as none.
#[test]
fn the_attention_window_argument_tracks_the_slot_to_position_offset() {
    use crate::dflash_head::forward_block_layer_paged::attn_window_arg;
    assert_eq!(attn_window_arg(Some(2048), 100, 100), 2048);
    assert_eq!(attn_window_arg(Some(2048), 5000, 3000), 4048);
    assert_eq!(attn_window_arg(None, 5000, 3000), 0);
    assert_eq!(attn_window_arg(Some(0), 5000, 3000), 0);
    // A key at slot k holds position k + 2000; the query's first row is position 5000. The
    // kernel masks when 5000 - k >= arg, i.e. when the key's position <= 5000 - 2048.
    let arg = attn_window_arg(Some(2048), 5000, 3000) as usize;
    let k_masked = 5000 - arg;
    assert_eq!(k_masked + 2000, 5000 - 2048);
}
