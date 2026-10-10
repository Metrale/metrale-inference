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

/// 2026-10-09: A ladder within γ - 1 is installed; one whose widest rung passes γ - 1 is
/// refused at startup, before any weight loads.
#[test]
fn a_draft_ladder_is_held_to_the_serves_gamma() {
    let dir = tempfile::tempdir().unwrap();
    drafter(dir.path());
    let d = dir.path().to_str().unwrap();
    let ladder = "1:7,2:5,3:4,4:3,5:2,6:7";
    let mut ok = args(&[
        "--dflash",
        "--draft-model",
        d,
        "--dflash-gamma",
        "8",
        "--dflash-draft-ladder",
        ladder,
    ]);
    apply_dflash_gamma(&mut ok, None).unwrap();
    assert_eq!(
        ok.dflash_draft_ladder_resolved,
        Some(metrale_speculative::dflash_ladder::DraftLadder::parse(ladder).unwrap())
    );
    let mut wide = args(&[
        "--dflash",
        "--draft-model",
        d,
        "--dflash-gamma",
        "7",
        "--dflash-draft-ladder",
        ladder,
    ]);
    assert!(apply_dflash_gamma(&mut wide, None).is_err());
}

/// 2026-10-10: An adaptive-K cost table within γ - 1 is installed; one that prices more drafts
/// than γ - 1, or that does not parse, is refused at startup; without `--dflash` the flag is
/// refused by the parser.
#[test]
fn an_adaptive_k_table_is_held_to_the_serves_gamma() {
    let dir = tempfile::tempdir().unwrap();
    drafter(dir.path());
    let d = dir.path().to_str().unwrap();
    let table = "0:32,1:44,2:50,3:56";
    let base = ["--dflash", "--draft-model", d, "--dflash-gamma"];
    let mut ok = args(&[&base[..], &["4", "--dflash-adaptive-k", table]].concat());
    apply_dflash_gamma(&mut ok, None).unwrap();
    assert_eq!(
        ok.dflash_args.dflash_adaptive_k_resolved,
        Some(metrale_speculative::dflash_adaptive_k::CostModel::parse(table).unwrap())
    );
    let mut wide = args(&[&base[..], &["3", "--dflash-adaptive-k", table]].concat());
    assert!(apply_dflash_gamma(&mut wide, None).is_err());
    let mut bad = args(&[&base[..], &["4", "--dflash-adaptive-k", "0:32,2:50"]].concat());
    assert!(apply_dflash_gamma(&mut bad, None).is_err());
    assert!(
        cli::ServeArgs::try_parse_from(["serve", "some/model", "--dflash-adaptive-k", table])
            .is_err()
    );
}

/// 2026-10-10: `--draft-tp-size`: unset is 1; `auto` is the world size; 1 and the world size
/// pass; any other count, a rank count past the world, or a split the drafter's heads cannot
/// cover is refused; 0 and junk are refused by the parser; the flag needs `--dflash`.
#[test]
fn draft_tp_size_resolves_and_refuses() {
    use cli::serve_args_dflash::{DraftTpSize, parse_draft_tp_size};
    let glm = || {
        metrale_model_arch::weight_loader::dflash_loader::parse_dflash_config(
            r#"{"hidden_size": 4096, "num_hidden_layers": 5, "num_attention_heads": 32,
                "num_key_value_heads": 8, "intermediate_size": 12288, "vocab_size": 154880,
                "head_dim": 128}"#,
        )
    };
    assert_eq!(resolve_draft_tp_size(None, 3, glm).unwrap(), 1);
    assert_eq!(
        resolve_draft_tp_size(Some(DraftTpSize::Auto), 3, glm).unwrap(),
        3
    );
    assert_eq!(
        resolve_draft_tp_size(Some(DraftTpSize::Ranks(3)), 3, glm).unwrap(),
        3
    );
    assert_eq!(
        resolve_draft_tp_size(Some(DraftTpSize::Ranks(1)), 3, glm).unwrap(),
        1
    );
    for n in [2, 4] {
        assert!(resolve_draft_tp_size(Some(DraftTpSize::Ranks(n)), 3, glm).is_err());
    }
    // 2026-10-10: Nine ranks cannot split 8 KV heads.
    assert!(resolve_draft_tp_size(Some(DraftTpSize::Auto), 9, glm).is_err());
    assert_eq!(parse_draft_tp_size("auto"), Ok(DraftTpSize::Auto));
    assert_eq!(parse_draft_tp_size("3"), Ok(DraftTpSize::Ranks(3)));
    for bad in ["0", "-1", "", "two", "3.0"] {
        assert!(parse_draft_tp_size(bad).is_err(), "{bad:?} accepted");
    }
    assert!(
        cli::ServeArgs::try_parse_from(["serve", "some/model", "--draft-tp-size", "3"]).is_err(),
        "--draft-tp-size without --dflash"
    );
    assert!(
        cli::ServeArgs::try_parse_from(["serve", "some/model", "--dflash", "--draft-tp-size", "0"])
            .is_err()
    );
}

/// 2026-10-10: `--draft-weight-quantization`: unset is `fp8` (the drafter build's behaviour
/// before the flag); every tier parses by name; a misspelt or empty value is refused by the
/// parser; an explicit value needs `--dflash`.
#[test]
fn draft_weight_quantization_parses_every_tier_and_refuses_junk() {
    use metrale_model_arch::dflash_head::DraftWeightQuantization as Q;
    let parse = |extra: &[&str]| {
        let mut argv = vec!["serve", "some/model", "--dflash"];
        argv.extend_from_slice(extra);
        cli::ServeArgs::try_parse_from(argv)
    };
    assert_eq!(
        parse(&[]).unwrap().dflash_args.draft_weight_quantization,
        Q::Fp8
    );
    for t in Q::ALL {
        assert_eq!(
            parse(&["--draft-weight-quantization", t.name()])
                .unwrap()
                .dflash_args
                .draft_weight_quantization,
            t
        );
    }
    for bad in ["", "FP8", "fp4", "bf16", "nvfp4-gate-up"] {
        assert!(
            parse(&["--draft-weight-quantization", bad]).is_err(),
            "{bad:?} accepted"
        );
    }
    assert!(
        cli::ServeArgs::try_parse_from([
            "serve",
            "some/model",
            "--draft-weight-quantization",
            "nvfp4"
        ])
        .is_err(),
        "--draft-weight-quantization without --dflash"
    );
}
