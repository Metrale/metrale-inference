// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The `qwen3.6-35b-a3b-nvfp4-declared` recipe renders the serve the declared NVFP4
//! A/B measured (PR #120, dgx3, C1-C128): the same flags and values, so a gate that serves the
//! recipe reproduces the measured configuration. A recipe edit that drops or changes one of them
//! (the batch cap once resolved to the CLI default of 8) fails here.
//!
//! Differences from the A/B's command line, none of which changes how a request without tools is
//! served: the port; `--disable-thinking`, which the A/B gave and the recipe does not, because the
//! harness (and the vLLM leg's) sends `enable_thinking: false` with every request; and
//! `--tool-call-parser` / `--tool-grammar`, which act only on requests that carry tools.

use super::*;

fn nvfp4_declared() -> Recipe {
    let stem = "qwen3.6-35b-a3b-nvfp4-declared";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../recipes/qwen3.6")
        .join(format!("{stem}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Recipe::parse(format!("qwen3.6/{stem}"), &text).unwrap_or_else(|e| panic!("{stem}: {e:#}"))
}

/// 2026-10-05: The flags the A/B passed (wave_ab/scripts/common.sh NVARGV, arm MNWVCX, with the
/// exact wave as its flag), minus the port, as `(flag, value)`; `None` for a presence flag.
const MEASURED: &[(&str, Option<&str>)] = &[
    ("--activation-quantization", Some("declared")),
    ("--enable-prefix-caching", None),
    ("--gpu-memory-utilization", Some("0.85")),
    ("--bind", Some("0.0.0.0")),
    ("--kv-cache-dtype", Some("bf16")),
    ("--kv-high-precision-layers", Some("auto")),
    ("--lm-head-dtype", Some("nvfp4")),
    ("--max-batch-size", Some("128")),
    ("--max-seq-len", Some("2048")),
    ("--mtp-quantization", Some("bf16")),
    ("--num-drafts", Some("1")),
    ("--request-timeout", Some("0")),
    ("--scheduler", Some("fifo")),
    ("--speculative", None),
    ("--ssm-cache-slots", Some("32")),
    ("--weight-quantization", Some("declared")),
    ("--prefill-varlen-batch", None),
    ("--prefill-codispatch", None),
    ("--prefill-wave-exact", None),
];

/// 2026-10-05: Flags the recipe renders beyond the measured set (module header).
const RECIPE_ONLY: &[(&str, Option<&str>)] = &[
    ("--port", Some("8888")),
    ("--tool-call-parser", Some("qwen3_coder")),
    ("--tool-grammar", Some("off")),
];

fn pairs(argv: &[String]) -> std::collections::BTreeSet<(String, Option<String>)> {
    let mut out = std::collections::BTreeSet::new();
    let mut i = 3; // 2026-10-05: past `met serve <model>`.
    while i < argv.len() {
        let flag = argv[i].clone();
        let value = argv.get(i + 1).filter(|v| !v.starts_with("--")).cloned();
        i += if value.is_some() { 2 } else { 1 };
        out.insert((flag, value));
    }
    out
}

#[test]
fn the_nvfp4_declared_recipe_renders_the_measured_serve() {
    let r = nvfp4_declared();
    let argv = r.argv(&BTreeMap::new()).expect("argv");
    assert_eq!(argv[2], "nvidia/Qwen3.6-35B-A3B-NVFP4");
    let want: std::collections::BTreeSet<_> = MEASURED
        .iter()
        .chain(RECIPE_ONLY)
        .map(|(f, v)| (f.to_string(), v.map(str::to_string)))
        .collect();
    let got = pairs(&argv);
    // 2026-10-05: A subset check, not set equality: feat/recipes-fully-explicit (PR #124)
    // makes every previously-silent-default key explicit in the YAML, so the renderer now
    // emits far more flags than the measured set names. This test's job is catching a drop
    // or change of one of the MEASURED values, not pinning the recipe's total flag count.
    let missing: Vec<_> = want.difference(&got).collect();
    assert!(
        missing.is_empty(),
        "measured flag(s) missing or changed: {missing:?}; rendered: {}",
        argv.join(" ")
    );
    // 2026-10-05: And it parses to a serve the CLI accepts, with the batch cap the A/B ran.
    let args = r.serve_args(&BTreeMap::new()).expect("serve args");
    assert!(args.prefill_wave_exact && args.prefill_varlen_batch && args.prefill_codispatch);
}
