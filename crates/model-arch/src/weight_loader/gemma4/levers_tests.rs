// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The Gemma-4 loader levers read from the settings text against the environment
//! reads they replaced, frozen here. Every row is text, so no probe reads differently.
//!
//! Owner: model-arch weight loader (Gemma-4).
//! Invariants: none beyond the assertions.

use super::loader_a::text_switch;
use metrale_model_layers::lever_equivalence::{Old, check};

/// 2026-10-17: `METRALE_GEMMA4_BF16_ATTN` / `_MLP` as `load_layers_impl` matched them.
fn old_switch(v: Option<&str>, default: bool) -> bool {
    match v {
        Some("0") => false,
        Some("1") => true,
        _ => default,
    }
}

#[test]
fn the_bf16_levers_read_as_before_at_both_defaults() {
    for default in [false, true] {
        assert_eq!(
            check(
                "METRALE_GEMMA4_BF16_ATTN",
                Old::Verbatim,
                |p| old_switch(p, default),
                |s| text_switch(s.kernels.gemma4_bf16_attn.as_deref(), default)
            ),
            []
        );
        assert_eq!(
            check(
                "METRALE_GEMMA4_BF16_MLP",
                Old::Verbatim,
                |p| old_switch(p, default),
                |s| text_switch(s.kernels.gemma4_bf16_mlp.as_deref(), default)
            ),
            []
        );
    }
}

#[test]
fn the_ffn_transpose_opt_out_reads_as_before() {
    assert_eq!(
        check(
            "METRALE_GEMMA4_FFN_TRANSPOSE",
            Old::Verbatim,
            |p| p == Some("0"),
            |s| s.kernels.gemma4_ffn_transpose.as_deref() == Some("0")
        ),
        []
    );
}
