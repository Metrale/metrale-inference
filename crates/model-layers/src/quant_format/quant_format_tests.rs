// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The loaders' ignore lists match as each format specifies (the precision plan's
//! matcher): ModelOpt exact names and globs, compressed-tensors `re:` regexes and exact
//! names, HF `fp8` module paths on whole segments; a malformed entry is refused.

use metrale_config::precision_plan::IgnoreDialect;

use super::{
    CompressedTensorsFormat, Fp8BlockScaledFormat, IgnoreList, ModeloptFormat, QuantFormat,
};
use crate::weight_map::Nvfp4Variant;

fn list(d: IgnoreDialect, entries: &[&str]) -> IgnoreList {
    let entries: Vec<String> = entries.iter().map(|s| s.to_string()).collect();
    IgnoreList::new(d, &entries).expect("entries parse")
}

#[test]
fn modelopt_entries_are_exact_names_or_globs() {
    let l = list(
        IgnoreDialect::ModelOpt,
        &["lm_head", "model.layers.*.self_attn*", "mtp*"],
    );
    assert!(l.matches("lm_head"));
    assert!(!l.matches("lm_head_norm"));
    assert!(!l.matches("model.lm_head"));
    assert!(l.matches("model.layers.5.self_attn.q_proj"));
    assert!(!l.matches("model.layers.5.mlp.gate_proj"));
    assert!(l.matches("mtp.fc"));
}

#[test]
fn compressed_tensors_entries_are_re_patterns_or_exact_names() {
    // 2026-09-30: The Sehyo Qwen3.5 MoE and unsloth Qwen3.8 ignore lists.
    let l = list(
        IgnoreDialect::CompressedTensors,
        &[r"re:mtp\.layers\.\d+\.", "mtp.fc", "re:^mtp.*"],
    );
    assert!(l.matches("mtp.fc"), "an exact module entry");
    assert!(l.matches("mtp.layers.0.self_attn.q_proj"), "a re: pattern");
    assert!(!l.matches("model.layers.0.self_attn.q_proj"));
    let only_re = list(
        IgnoreDialect::CompressedTensors,
        &[r"re:mtp\.layers\.\d+\."],
    );
    assert!(
        !only_re.matches("model.mtp.layers.0.x"),
        "re.match anchors at the start"
    );
}

#[test]
fn hf_fp8_entries_match_whole_segments() {
    let l = list(
        IgnoreDialect::HfFp8,
        &["model.layers.0.mlp.gate", "lm_head"],
    );
    assert!(l.matches("model.layers.0.mlp.gate"));
    assert!(!l.matches("model.layers.0.mlp.gate_proj"));
    assert!(l.matches("lm_head"));
}

#[test]
fn a_malformed_entry_is_refused() {
    let bad = |d, e: &str| IgnoreList::new(d, &[e.to_string()]).is_err();
    assert!(bad(IgnoreDialect::CompressedTensors, "re:mtp.(layers"));
    assert!(bad(IgnoreDialect::HfFp8, ""));
    assert!(CompressedTensorsFormat::new(String::new(), &["re:(".to_string()]).is_err());
    // 2026-09-30: The FP8 layout's list is a ModelOpt export's (DeepSeek-V4-Flash globs).
    let dsv4 = ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"].map(String::from);
    let f = Fp8BlockScaledFormat::new(&dsv4).expect("globs parse");
    assert!(f.is_ignored("layers.3.attn.wq_a"));
    assert!(f.is_ignored("mtp.fc"));
    assert!(!f.is_ignored("layers.3.ffn.experts"));
}

#[test]
fn ignore_entries_drive_effective_variants_for_every_format() {
    let ignore = ["model.layers.5.self_attn.q_proj".to_string()];
    let formats: Vec<(Box<dyn QuantFormat>, Nvfp4Variant)> = vec![
        (
            Box::new(ModeloptFormat::new("NVFP4".into(), &ignore).unwrap()),
            Nvfp4Variant::Standard,
        ),
        (
            Box::new(CompressedTensorsFormat::new("nvfp4-pack-quantized".into(), &ignore).unwrap()),
            Nvfp4Variant::CompressedTensors,
        ),
        (
            Box::new(Fp8BlockScaledFormat::new(&ignore).unwrap()),
            Nvfp4Variant::Fp8Dequanted,
        ),
    ];
    for (format, base) in formats {
        assert_eq!(format.base_variant(), base);
        assert_eq!(
            format.variant_for("model.layers.5.self_attn.q_proj"),
            Nvfp4Variant::Bf16Raw
        );
        assert_eq!(format.variant_for("model.layers.5.mlp.gate_proj"), base);
    }
}
