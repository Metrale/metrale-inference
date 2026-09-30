// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Tests for `apply_dflash_gamma`: the serve's γ is the pinned flag, else the
//! drafter's block size + 2 through the one resolver, over a drafter directory on disk.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use clap::Parser;

use super::*;

/// 2026-09-30: A drafter checkpoint whose `dflash_config.block_size` is 8 (DFlash2's).
fn drafter(dir: &std::path::Path) {
    std::fs::write(
        dir.join("config.json"),
        r#"{"hidden_size": 5120, "num_hidden_layers": 5, "num_attention_heads": 32,
            "num_key_value_heads": 8, "intermediate_size": 17408, "vocab_size": 248320,
            "head_dim": 128,
            "dflash_config": {"block_size": 8, "mask_token_id": 248070,
                              "target_layer_ids": [1, 10, 19, 28, 37]}}"#,
    )
    .unwrap();
}

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut argv = vec!["serve", "some/model"];
    argv.extend_from_slice(extra);
    cli::ServeArgs::try_parse_from(argv).unwrap()
}

#[test]
fn a_pinned_gamma_wins_and_an_unpinned_one_is_the_drafters_block_plus_two() {
    let dir = tempfile::tempdir().unwrap();
    drafter(dir.path());
    let d = dir.path().to_str().unwrap();
    let mut pinned = args(&["--dflash", "--draft-model", d, "--dflash-gamma", "8"]);
    apply_dflash_gamma(&mut pinned, None).unwrap();
    assert_eq!(
        pinned.serve_dflash_gamma(),
        8,
        "--dflash-gamma 8 is what the build sizes"
    );
    assert_eq!(pinned.dflash_gamma, Some(8), "the flag itself is untouched");
    let mut free = args(&["--dflash", "--draft-model", d]);
    apply_dflash_gamma(&mut free, None).unwrap();
    assert_eq!(free.serve_dflash_gamma(), 10, "block 8 + 2");
    assert_eq!(
        free.dflash_gamma, None,
        "an unpinned γ stays unpinned for the rung"
    );
    // 2026-09-30: The build reaches the same γ from the same drafter config.
    let cfg = read_dflash_config(dir.path()).unwrap();
    for (flag, want) in [(Some(8), 8), (None, 10)] {
        assert_eq!(
            metrale_model_layers::layers::qwen3_ssm::resolve_dflash_gamma(
                flag,
                Some(cfg.effective_block_size())
            ),
            want
        );
    }
}

#[test]
fn without_dflash_nothing_is_resolved() {
    let mut a = args(&[]);
    apply_dflash_gamma(&mut a, None).unwrap();
    assert_eq!(a.dflash_gamma_resolved, None);
}
