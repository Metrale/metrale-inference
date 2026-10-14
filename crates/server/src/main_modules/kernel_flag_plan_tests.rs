// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `KernelFlagPlan::from_args` on parsed, validated command lines.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::{GdnPlan, KernelFlagPlan, precision_line};
use crate::cli::{Cli, Command, validate_serve_args};
use clap::Parser;
use metrale_config::{W4a4Downcast, WeightQuantTier, WeightQuantization};
use metrale_model_layers::layers::{
    DenseQuantization, DsaQuantization, ExpertQuantization, Phased,
};

fn plan(flags: &[&str]) -> KernelFlagPlan {
    let mut argv = vec![
        "met",
        "serve",
        "org/model",
        "--activation-quantization",
        "adaptive",
    ];
    argv.extend_from_slice(flags);
    let Command::Serve(args) = Cli::try_parse_from(argv).expect("parses").command else {
        unreachable!("a serve command")
    };
    validate_serve_args(&args).expect("valid");
    KernelFlagPlan::from_args(&args)
}

#[test]
fn an_empty_command_line_publishes_nothing_the_environment_owns() {
    // 2026-09-26: A `Some` here would seal that cell's `METRALE_*` fallback on
    // every boot.
    assert_eq!(
        plan(&[]),
        KernelFlagPlan {
            gdn: None,
            weight_quant: WeightQuantTier::default(),
            expert_quantization: ExpertQuantization::Fp8,
            dense_quantization: Phased::both(DenseQuantization::Declared),
            dsa_quantization: Phased::both(DsaQuantization::Declared),
            prefill_codispatch: None,
            prefill_varlen: None,
            ssm_tail_midchunk: None,
            hermetic: false,
        }
    );
    // 2026-09-26: `auto` is the absence of a pin, not a GDN flag.
    assert_eq!(plan(&["--ssm-batched-recurrent", "auto"]).gdn, None);
}

#[test]
fn any_gdn_flag_hands_the_whole_cell_to_the_command_line() {
    let fused = GdnPlan {
        h_f16: false,
        h_f16_pool: false,
        fused_norm: true,
        batched_recurrent: None,
        exact_verify: false,
    };
    assert_eq!(plan(&["--gdn-fused-norm"]).gdn, Some(fused));
    // 2026-09-26: A GDN flag that does not name batched recurrence leaves it
    // `None` (the target default); an explicit pin is carried either way.
    assert_eq!(
        plan(&["--ssm-batched-recurrent", "off"]).gdn,
        Some(GdnPlan {
            fused_norm: false,
            batched_recurrent: Some(false),
            ..fused
        })
    );
    assert_eq!(
        plan(&["--gdn-fused-norm", "--ssm-batched-recurrent", "on"]).gdn,
        Some(GdnPlan {
            batched_recurrent: Some(true),
            ..fused
        })
    );
    assert_eq!(
        plan(&["--exact-verify"]).gdn,
        Some(GdnPlan {
            fused_norm: false,
            exact_verify: true,
            ..fused
        })
    );
    assert_eq!(
        plan(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"]).gdn,
        Some(GdnPlan {
            h_f16: true,
            ..fused
        })
    );
}

#[test]
fn each_presence_flag_publishes_its_non_default_state_only() {
    let p = plan(&[
        "--prefill-codispatch",
        "--prefill-varlen-batch",
        "--no-ssm-tail-midchunk",
        "--hermetic",
    ]);
    assert_eq!(p.prefill_codispatch, Some(true));
    assert_eq!(p.prefill_varlen, Some(true));
    assert_eq!(p.ssm_tail_midchunk, Some(false));
    assert!(p.hermetic);
}

/// 2026-09-28: `--weight-quantization` is `declared` by default and carries its lever under
/// `nvfp4`; `-wide` alone does nothing, as it always has.
#[test]
fn the_weight_quantization_tier_carries_its_lever() {
    let tier = |t, d| WeightQuantTier::new(t, d).expect("tier");
    assert_eq!(
        plan(&[]).weight_quant,
        tier(WeightQuantization::Declared, W4a4Downcast::Off)
    );
    let nv = |flags: &[&str]| {
        let mut argv = vec!["--weight-quantization", "nvfp4"];
        argv.extend_from_slice(flags);
        plan(&argv).weight_quant
    };
    assert_eq!(nv(&[]), tier(WeightQuantization::Nvfp4, W4a4Downcast::Off));
    assert_eq!(
        nv(&["--w4a4-downcast-wide"]),
        tier(WeightQuantization::Nvfp4, W4a4Downcast::Off)
    );
    assert_eq!(
        nv(&["--w4a4-downcast"]),
        tier(WeightQuantization::Nvfp4, W4a4Downcast::Narrow)
    );
    assert_eq!(
        nv(&["--w4a4-downcast", "--w4a4-downcast-wide"]),
        tier(WeightQuantization::Nvfp4, W4a4Downcast::Wide)
    );
    assert_eq!(
        plan(&["--weight-quantization", "declared"]).weight_quant,
        WeightQuantTier::default()
    );
}

/// 2026-09-28: A value outside the tiers is refused by clap, and the W4A4 lever under
/// `declared` (given or defaulted) by `validate_serve_args`.
#[test]
fn a_bad_weight_quantization_or_a_lever_under_declared_is_refused() {
    for argv in [
        vec!["met", "serve", "org/model", "--weight-quantization", "fp8"],
        vec!["met", "serve", "org/model", "--weight-quantization"],
    ] {
        assert!(Cli::try_parse_from(&argv).is_err(), "{argv:?} parsed");
    }
    for extra in [
        &["--w4a4-downcast"][..],
        &["--w4a4-downcast", "--weight-quantization", "declared"][..],
        &["--w4a4-downcast", "--w4a4-downcast-wide"][..],
    ] {
        let mut argv = vec!["met", "serve", "org/model"];
        argv.extend_from_slice(extra);
        let Command::Serve(args) = Cli::try_parse_from(&argv).expect("parses").command else {
            unreachable!("a serve command")
        };
        let err = validate_serve_args(&args).expect_err("the lever under declared is refused");
        assert!(err.contains("--weight-quantization nvfp4"), "{err}");
    }
}

#[test]
fn the_expert_quantization_tier_is_carried_and_fp8_by_default() {
    assert_eq!(plan(&[]).expert_quantization, ExpertQuantization::Fp8);
    for q in ExpertQuantization::ALL {
        assert_eq!(
            plan(&["--expert-quantization", q.name()]).expert_quantization,
            q
        );
    }
}

/// 2026-09-27: A value outside the tiers, and the removed presence flag, are refused by clap.
#[test]
fn a_bad_expert_quantization_value_is_refused() {
    for argv in [
        vec![
            "met",
            "serve",
            "org/model",
            "--expert-quantization",
            "nvfp8",
        ],
        vec!["met", "serve", "org/model", "--expert-quantization"],
        vec!["met", "serve", "org/model", "--moe-nvfp4-experts"],
    ] {
        assert!(Cli::try_parse_from(&argv).is_err(), "{argv:?} parsed");
    }
}

/// 2026-10-09: `--dense-quantization fp8` and `w4a16` reach the plan; nothing else changes.
#[test]
fn the_dense_tier_reaches_the_plan() {
    for (value, tier) in [
        ("fp8", DenseQuantization::Fp8),
        ("w4a16", DenseQuantization::W4a16),
    ] {
        let p = plan(&["--dense-quantization", value]);
        assert_eq!(p.dense_quantization, Phased::both(tier));
        assert_eq!(
            KernelFlagPlan {
                dense_quantization: Phased::both(DenseQuantization::Declared),
                ..p
            },
            plan(&[])
        );
    }
}

/// 2026-10-11: `--dsa-quantization w8a8` and `w4a16` reach the plan and move nothing else,
/// whatever `--dense-quantization` is; the two flags are independent cells.
#[test]
fn the_dsa_tier_reaches_the_plan_alone() {
    for (value, tier) in [
        ("declared", DsaQuantization::Declared),
        ("w8a8", DsaQuantization::W8a8),
        ("w4a16", DsaQuantization::W4a16),
    ] {
        let p = plan(&["--dsa-quantization", value]);
        assert_eq!(p.dsa_quantization, Phased::both(tier));
        assert_eq!(
            KernelFlagPlan {
                dsa_quantization: Phased::both(DsaQuantization::Declared),
                ..p
            },
            plan(&[])
        );
        let both = plan(&["--dense-quantization", "w4a16", "--dsa-quantization", value]);
        assert_eq!(
            (both.dense_quantization, both.dsa_quantization),
            (Phased::both(DenseQuantization::W4a16), Phased::both(tier))
        );
    }
}

/// 2026-10-11: The boot log names both tiers and what the DSA sites run as.
#[test]
fn the_precision_line_discloses_both_tiers() {
    let l = precision_line(
        Phased::both(DenseQuantization::W4a16),
        Phased::both(DsaQuantization::W8a8),
    );
    assert!(l.contains("--dense-quantization w4a16 "), "{l}");
    assert!(l.contains("--dsa-quantization w8a8 "), "{l}");
    assert!(l.contains("W8A8"), "{l}");
    let l = precision_line(
        Phased::both(DenseQuantization::Declared),
        Phased::both(DsaQuantization::Declared),
    );
    assert!(
        l.contains("--dsa-quantization declared") && l.contains("BF16"),
        "{l}"
    );
    // 2026-10-14: A split prints its flag value back and each DSA phase's precision.
    let l = precision_line(
        Phased {
            decode: DenseQuantization::W4a16,
            prefill: DenseQuantization::Fp8,
        },
        Phased {
            decode: DsaQuantization::W4a16,
            prefill: DsaQuantization::W8a8,
        },
    );
    assert!(
        l.contains("--dense-quantization decode=w4a16,prefill=fp8 "),
        "{l}"
    );
    assert!(
        l.contains("--dsa-quantization decode=w4a16,prefill=w8a8 "),
        "{l}"
    );
    assert!(
        l.contains("(DSA q_a/q_absorb/o_absorb: decode NVFP4 weights, 16-bit activations (W4A16); prefill FP8 per-channel weights, per-token FP8 activations (W8A8))"),
        "{l}"
    );
}

/// 2026-10-14: The served split reaches the plan as two tiers; an unserved one (FP8 decode under a
/// W4A16 prompt pass) is refused at parse, naming the flag.
#[test]
fn a_phase_split_reaches_the_plan_and_an_unserved_one_is_refused() {
    let p = plan(&["--dense-quantization", "decode=w4a16,prefill=fp8"]);
    assert_eq!(
        p.dense_quantization,
        Phased {
            decode: DenseQuantization::W4a16,
            prefill: DenseQuantization::Fp8
        }
    );
    assert_eq!(
        KernelFlagPlan {
            dense_quantization: Phased::both(DenseQuantization::Declared),
            ..p
        },
        plan(&[])
    );
    for (flag, value) in [
        ("--dense-quantization", "decode=fp8,prefill=w4a16"),
        ("--dense-quantization", "decode=declared,prefill=fp8"),
        ("--dsa-quantization", "decode=w8a8,prefill=w4a16"),
    ] {
        let e = Cli::try_parse_from(["met", "serve", "m", flag, value])
            .expect_err(value)
            .to_string();
        assert!(e.contains(flag), "{value}: {e}");
    }
}
