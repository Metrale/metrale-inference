// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Build script for metrale-model-arch: declares the AMD target cfgs
//! and sets them from the `vendor` of `kernels/<METRALE_TARGET_HW>/HARDWARE.toml`
//! (keyed on the target name before).
//!
//! Owner: model-arch (build).
//! Invariants:
//! - `metrale_hip` is set only together with `metrale_scale`.

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    // 2026-10-12: `metrale_scale` is set for every AMD target (vendor `amd`/`rocm`/`scale`
    // for SCALE, `hip` for native HIP), the same rule gpu-runtime's build.rs uses. In this
    // crate only the `attn_prefill_microtest` example reads it.
    println!("cargo:rustc-check-cfg=cfg(metrale_scale)");
    // 2026-10-12: `metrale_hip` is set only for vendor `hip`, the native-HIP targets.
    // The Qwen3.5 loader reads it to skip `predequant_for_prefill` and the FP8
    // prefill weights (`weight_loader/qwen35/load_layers/linear_attn_arms/nvfp4.rs`).
    println!("cargo:rustc-check-cfg=cfg(metrale_hip)");
    // 2026-10-12: Both cfgs follow the target's HARDWARE.toml `vendor`, not its name. The name
    // rule (`strix*`, `strix-hip`) left a new AMD class such as `gfx1201` with neither cfg, so
    // its host binary was not a HIP binary. Same results for every existing target.
    let vendor = target_vendor();
    if matches!(vendor.as_deref(), Some("amd" | "rocm" | "scale" | "hip")) {
        println!("cargo:rustc-cfg=metrale_scale");
    }
    if vendor.as_deref() == Some("hip") {
        println!("cargo:rustc-cfg=metrale_hip");
    }
}

/// 2026-10-12: `[hardware].vendor` of `kernels/<METRALE_TARGET_HW>/HARDWARE.toml`, the key
/// metrale-kernels' build.rs selects the compiler by. `None` when `METRALE_TARGET_HW` is unset
/// (an unset target sets no AMD cfg, as before) or the file has no `vendor`. A set target whose
/// file is missing or malformed panics.
fn target_vendor() -> Option<String> {
    let hw = std::env::var("METRALE_TARGET_HW").ok()?;
    let manifest =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let hardware_toml = manifest
        .parent()
        .and_then(|crates| crates.parent())
        .expect("crates/<crate> sits two levels below the workspace root")
        .join("kernels")
        .join(hw)
        .join("HARDWARE.toml");
    println!("cargo:rerun-if-changed={}", hardware_toml.display());
    let doc = read_hardware_toml(&hardware_toml);
    Some(doc.get("hardware")?.get("vendor")?.as_str()?.to_string())
}

/// 2026-10-13: A `HARDWARE.toml` that is missing or does not parse fails the build, as it does in
/// metrale-kernels' build.rs. Returning `None` instead would skip the AMD cfgs with no error.
fn read_hardware_toml(path: &std::path::Path) -> toml::Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("invalid {}: {e}", path.display()))
}
