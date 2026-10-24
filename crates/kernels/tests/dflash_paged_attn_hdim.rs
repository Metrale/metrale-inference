// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The DFlash drafter's paged attention module (`attn_prefill_paged_indirect`) must
//! be compiled for the drafter's head width. The common source defaults HDIM to 256; a target
//! whose drafter has 128-wide heads needs the HDIM 128 build, or each score silently mixes two
//! heads (measured on GLM-5.3-Flash: layer-0 attention 50-90% off the drafter's math, and draft
//! acceptance at the first position 0.51 instead of 0.59).
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
use std::path::Path;

/// 2026-10-10: Targets whose DFlash drafter has 128-wide attention heads.
const HDIM_128_DRAFTER_TARGETS: [&str; 2] = ["glm-5.3-flash", "qwen3.6-27b"];

/// 2026-10-10: The HDIM the resolved `attn_prefill_paged_indirect` source compiles at: its own
/// `#define HDIM`, else the common compute header's default.
fn compiled_hdim(source: &Path, root: &Path) -> u32 {
    let parse = |text: &str, marker: &str| {
        text.lines()
            .map(str::trim)
            .find_map(|l| l.strip_prefix(marker))
            .map(|v| v.trim().parse::<u32>().expect("numeric HDIM"))
    };
    let text = std::fs::read_to_string(source).unwrap();
    parse(&text, "#define HDIM ").unwrap_or_else(|| {
        let header = root.join("kernels/gb10/common/prefill_paged_compute.cuh");
        parse(&std::fs::read_to_string(header).unwrap(), "#define HDIM ")
            .expect("prefill_paged_compute.cuh defines a default HDIM")
    })
}

#[test]
fn hdim_128_drafter_targets_build_the_paged_attention_at_hdim_128() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .canonicalize()
        .unwrap();
    for model in HDIM_128_DRAFTER_TARGETS {
        let t = metrale_closure::layout::Target {
            hardware: "gb10".into(),
            model: model.into(),
            quant: "nvfp4".into(),
        };
        let l = metrale_closure::layout::discover(&root, &t).unwrap();
        let (_, e) = l
            .modules()
            .into_iter()
            .find(|(s, _)| s == "attn_prefill_paged_indirect")
            .unwrap_or_else(|| panic!("{model}: attn_prefill_paged_indirect absent"));
        assert_eq!(
            compiled_hdim(&e.source, &root),
            128,
            "{model}: {} compiles the drafter's paged attention at another HDIM",
            e.source.display()
        );
    }
}

#[test]
fn the_common_paged_attention_source_is_not_hdim_128() {
    // 2026-10-10: The control: a target that falls through to the common source gets HDIM 256,
    // so the check above can fail.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .canonicalize()
        .unwrap();
    let common = root.join("kernels/gb10/common/attn_prefill_paged_indirect.cu");
    assert_eq!(compiled_hdim(&common, &root), 256);
}
