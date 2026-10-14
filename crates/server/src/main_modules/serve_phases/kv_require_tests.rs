// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `require_kv_tokens` at boot fn-b19a's pool (1,678 blocks of 16 tokens, 384 KiB a
//! block), which a 36,864-token run needs 2,304 blocks of.

use super::require_kv_tokens;

const BLOCK_BYTES: usize = 12 * 32_768;

#[test]
fn without_the_flag_any_pool_passes() {
    require_kv_tokens(None, 0, None, 0).unwrap();
    require_kv_tokens(None, 1_678, Some(16), BLOCK_BYTES).unwrap();
}

#[test]
fn a_short_pool_fails_with_its_numbers() {
    let err = require_kv_tokens(Some(36_864), 1_678, Some(16), BLOCK_BYTES)
        .unwrap_err()
        .to_string();
    assert!(err.contains("holds 26848 tokens"), "{err}");
    assert!(err.contains("10016 tokens short (626 blocks"), "{err}");
}

#[test]
fn the_boundary_is_inclusive() {
    require_kv_tokens(Some(36_864), 2_304, Some(16), BLOCK_BYTES).unwrap();
    assert!(require_kv_tokens(Some(36_865), 2_304, Some(16), BLOCK_BYTES).is_err());
}

#[test]
fn a_model_without_a_paged_pool_fails_when_asked() {
    let err = require_kv_tokens(Some(1), 0, None, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no paged KV pool"), "{err}");
}
