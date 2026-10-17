// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Build script for metrale-gpu-examples: declares the AMD target cfgs
//! and sets them from `METRALE_TARGET_HW`, the same rule metrale-model-arch's build
//! script uses.
//!
//! Owner: gpu-examples (build).
//! Invariants:
//! - `metrale_hip` is set only together with `metrale_scale`.

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    // 2026-10-17: `metrale_scale` is set for every `METRALE_TARGET_HW` that starts
    // with `strix` (`strix` and `strix-hip`). The `attn_prefill_microtest` example
    // reads it to pick its tile rows.
    println!("cargo:rustc-check-cfg=cfg(metrale_scale)");
    // 2026-10-17: `metrale_hip` is set only for `strix-hip`, the native-HIP target.
    // The `w8a16_microtest` example reads it to pick its launch geometry.
    println!("cargo:rustc-check-cfg=cfg(metrale_hip)");
    let hw = std::env::var("METRALE_TARGET_HW").unwrap_or_default();
    if hw.starts_with("strix") {
        println!("cargo:rustc-cfg=metrale_scale");
    }
    if hw == "strix-hip" {
        println!("cargo:rustc-cfg=metrale_hip");
    }
}
