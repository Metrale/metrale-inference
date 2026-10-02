// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `disclosed_from`: the serve knobs a gate record
//! discloses, read off a rendered fixture recipe; and for the committed
//! `[benchmarks.serve_overrides]` pins, which must name `met serve` flags.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use super::{disclosed_from, rendered_overrides};
use crate::recipe::Recipe;
use std::collections::BTreeMap;

fn recipe(defaults: &str) -> Recipe {
    let text = format!(
        "recipe_version: \"2\"\nmodel: org/model\ncontainer: metrale\nruntime: metrale\n\
         metadata:\n  updated: \"2026-08-28\"\ndefaults:\n  activation_quantization: adaptive\n{defaults}"
    );
    Recipe::parse("fam/stem", &text).expect("the fixture recipe parses")
}

fn disclosed(defaults: &str, overrides: &[(&str, &str)]) -> Vec<(String, String)> {
    let overrides: BTreeMap<String, String> = overrides
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let args = recipe(defaults)
        .serve_args(&overrides)
        .unwrap_or_else(|e| panic!("the fixture renders: {e:#}"));
    disclosed_from(&args).into_iter().collect()
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs_with_tier(v, "declared")
}

/// 2026-09-28: `v` plus the `weight_quantization` key every disclosure carries, which sorts
/// after the others. 2026-09-30: And `activation_quantization`, which the fixture recipe pins
/// to `adaptive` as every committed recipe does, and which sorts before them.
fn pairs_with_tier(v: &[(&str, &str)], tier: &str) -> Vec<(String, String)> {
    [("activation_quantization", "adaptive")]
        .iter()
        .chain(v)
        .chain(&[("weight_quantization", tier)])
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// 2026-09-26: The disclosure is what the rendered command resolved: the
/// recipe as pinned, an override on top, and `--hermetic` forcing the gate.
#[test]
fn the_disclosure_is_read_off_the_rendered_serve() {
    // 2026-09-26: A recipe that pins `mtp_gate: force` discloses it.
    assert_eq!(
        disclosed("  speculative: \"true\"\n  mtp_gate: force\n", &[]),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // 2026-09-26: Speculation on and no `mtp_gate` pinned: no `mtp_gate` is
    // disclosed, rather than `auto`.
    assert_eq!(
        disclosed("  speculative: \"true\"\n", &[]),
        pairs(&[("speculative", "true")])
    );
    // 2026-09-26: An explicit `auto` is disclosed.
    assert_eq!(
        disclosed("  speculative: \"true\"\n  mtp_gate: auto\n", &[]),
        pairs(&[("mtp_gate", "auto"), ("speculative", "true")])
    );
    // 2026-09-26: A `--serve-override mtp_gate=force` on an `auto` recipe wins.
    assert_eq!(
        disclosed(
            "  speculative: \"true\"\n  mtp_gate: auto\n",
            &[("mtp_gate", "force")]
        ),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // 2026-09-26: `--hermetic` forces the gate where the recipe pinned nothing;
    // the disclosure follows `ServeArgs::mtp_gate_force`, not the raw flag.
    assert_eq!(
        disclosed("  speculative: \"true\"\n", &[("hermetic", "true")]),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // 2026-09-26: No speculation: only `speculative = false` is disclosed.
    assert_eq!(
        disclosed("  max_batch_size: \"8\"\n", &[]),
        pairs(&[("speculative", "false")])
    );
}

/// 2026-09-26: A `prefill_codispatch = "true"` serve override renders
/// `--prefill-codispatch`, and the disclosure carries `true`.
#[test]
fn the_codispatch_flag_is_disclosed_off_the_rendered_serve() {
    assert_eq!(
        disclosed(
            "  max_batch_size: \"8\"\n",
            &[("prefill_codispatch", "true")]
        ),
        pairs(&[("prefill_codispatch", "true"), ("speculative", "false")])
    );
    // 2026-09-26: An override's `false` removes the recipe's `true`: the flag is
    // not rendered, so nothing is disclosed.
    assert_eq!(
        disclosed(
            "  prefill_codispatch: \"true\"\n",
            &[("prefill_codispatch", "false")]
        ),
        pairs(&[("speculative", "false")])
    );
}

/// 2026-09-26: `--w4a4-downcast` is disclosed when on; off discloses nothing. 2026-09-28: it
/// belongs to the `nvfp4` tier, which the recipe pins beside it.
#[test]
fn the_w4a4_downcast_flag_is_disclosed_off_the_rendered_serve() {
    let nvfp4 = "  weight_quantization: nvfp4\n";
    assert_eq!(
        disclosed(nvfp4, &[("w4a4_downcast", "true")]),
        pairs_with_tier(
            &[("speculative", "false"), ("w4a4_downcast", "true")],
            "nvfp4"
        )
    );
    assert_eq!(
        disclosed(
            &format!("{nvfp4}  w4a4_downcast: \"true\"\n"),
            &[("w4a4_downcast", "false")]
        ),
        pairs_with_tier(&[("speculative", "false")], "nvfp4")
    );
}

/// 2026-09-28: `--weight-quantization` is disclosed by name for either tier: the recipe's
/// pin, an override over it, and the `declared` default when neither names it.
#[test]
fn the_weight_quantization_tier_is_disclosed_off_the_rendered_serve() {
    assert_eq!(
        disclosed("  weight_quantization: nvfp4\n", &[]),
        pairs_with_tier(&[("speculative", "false")], "nvfp4")
    );
    assert_eq!(
        disclosed(
            "  weight_quantization: nvfp4\n",
            &[("weight_quantization", "declared")]
        ),
        pairs_with_tier(&[("speculative", "false")], "declared")
    );
    assert_eq!(
        disclosed("  max_batch_size: \"8\"\n", &[]),
        pairs_with_tier(&[("speculative", "false")], "declared")
    );
}

/// 2026-09-27: `--expert-quantization` is disclosed by tier name for the NVFP4 tiers; `fp8`
/// (the default, given or not) discloses nothing.
#[test]
fn the_expert_quantization_tier_is_disclosed_off_the_rendered_serve() {
    for tier in ["nvfp4-gate-up", "nvfp4"] {
        assert_eq!(
            disclosed(
                "  max_batch_size: \"8\"\n",
                &[("expert_quantization", tier)]
            ),
            pairs(&[("expert_quantization", tier), ("speculative", "false")])
        );
    }
    assert_eq!(
        disclosed(
            "  max_batch_size: \"8\"\n",
            &[("expert_quantization", "fp8")]
        ),
        pairs(&[("speculative", "false")])
    );
    assert_eq!(
        disclosed("  max_batch_size: \"8\"\n", &[]),
        pairs(&[("speculative", "false")])
    );
}

/// 2026-09-26: Every `[benchmarks.serve_overrides]` pin in the committed `BENCH.toml` files,
/// `--hermetic` expanded as `plan_serve` expands it, renders on a minimal recipe to a `met serve`
/// command line that clap parses and `validate_serve_args` accepts. A pin naming a renamed or
/// removed serve flag, or a value the flag refuses, fails here instead of when a gate unit
/// starts its server.
#[test]
fn every_committed_serve_pin_renders_a_valid_serve() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let entries = metrale_bench::gate::bench::load_all(&root).expect("the BENCH.toml files load");
    let pinned: Vec<_> = entries
        .iter()
        .filter(|(_, e)| !e.serve_overrides.is_empty())
        .collect();
    assert!(
        pinned.len() >= 10,
        "only {} entries pin serve overrides; the scan is not reading the tree",
        pinned.len()
    );
    let mut refused = Vec::new();
    for (target, entry) in pinned {
        let overrides = crate::cli::hermetic::expand(entry.serve_overrides.clone());
        let parsed = recipe("  max_batch_size: \"8\"\n").serve_args(&overrides);
        if let Err(e) = parsed {
            refused.push(format!(
                "{target} {} {}: {e:#}",
                entry.gate, entry.checkpoint
            ));
        }
    }
    assert!(refused.is_empty(), "{}", refused.join("\n"));
}

/// 2026-09-30: A gate serve renders a recipe that names no `activation_quantization` under
/// `adaptive`, the routing it was measured under, and never overrides one the recipe or the
/// requested set names; the rule is not an override, so `requested` never carries it.
#[test]
fn a_silent_recipe_is_served_adaptive_and_a_named_value_wins() {
    let m = |kv: &[(&str, &str)]| -> BTreeMap<String, String> {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    let silent = m(&[("speculative", "true")]);
    let r = rendered_overrides(&m(&[]), &silent, 9);
    assert_eq!(
        r.get("activation_quantization").map(String::as_str),
        Some("adaptive")
    );
    assert_eq!(r.get("port").map(String::as_str), Some("9"));
    let pinned = m(&[("activation_quantization", "declared")]);
    assert!(!rendered_overrides(&m(&[]), &pinned, 9).contains_key("activation_quantization"));
    let asked = m(&[("activation_quantization", "bf16")]);
    assert_eq!(
        rendered_overrides(&asked, &silent, 9)
            .get("activation_quantization")
            .map(String::as_str),
        Some("bf16")
    );
    // 2026-09-30: The rendered argv of a silent recipe carries it.
    let text = "recipe_version: \"2\"\nmodel: org/model\ncontainer: metrale\nruntime: metrale\n\
                defaults:\n  speculative: \"true\"\n";
    let silent_recipe = Recipe::parse("fam/silent", text).expect("parses");
    assert!(
        !silent_recipe
            .defaults
            .contains_key("activation_quantization")
    );
    let args = silent_recipe
        .serve_args(&rendered_overrides(&m(&[]), &silent_recipe.defaults, 9))
        .expect("renders");
    assert!(args.activation_quantization.is_adaptive());
}
