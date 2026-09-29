// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The MODEL.toml `[behavior]` keys that run a model above its checkpoint's
//! declared precision, read from the checked-in tree with the build script's own parser.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The parser takes an absent or misspelled key as its default without a word, so a renamed
//! key would silently drop an exception the model's evidence depends on, and a copied
//! `[behavior]` block would silently spread it. This pins where each one is set.

#[allow(dead_code)]
#[path = "../build_parse_behavior.rs"]
mod build_parse_behavior;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-29: `hw/model` of every model directory whose MODEL.toml sets `pick`.
fn models_setting(pick: fn(&build_parse_behavior::ParsedBehavior) -> bool) -> BTreeSet<String> {
    let root = kernels_root();
    let mut out = BTreeSet::new();
    for hw in std::fs::read_dir(&root).expect("kernels/ lists").flatten() {
        let Ok(models) = std::fs::read_dir(hw.path()) else {
            continue;
        };
        for model in models.flatten() {
            let dir = model.path();
            if dir.join("MODEL.toml").is_file() && pick(&build_parse_behavior::parse_behavior(&dir))
            {
                let rel = dir.strip_prefix(&root).expect("under kernels/");
                out.insert(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    out
}

/// 2026-09-29: `expert_down_w8a16` is set on Qwen3.6-35B-A3B on GB10 only, where its evidence
/// was measured (the ssm-state-poisoning gate with the E4M3 SiLU product).
#[test]
fn expert_down_w8a16_is_set_where_its_evidence_was_measured() {
    assert_eq!(
        models_setting(|b| b.expert_down_w8a16),
        BTreeSet::from(["gb10/qwen3.6-35b-a3b".to_string()])
    );
}
