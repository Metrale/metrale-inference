// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Build script for metrale-target-defaults: writes `OUT_DIR/target_defaults.rs`,
//! the `TARGET_DEFAULTS` and `TARGET_SM_COUNT` consts of the `kernels/<hw>` tree
//! `METRALE_TARGET_HW` selects.
//!
//! Owner: metrale-target-defaults build.
//! Invariants:
//! - The text is `metrale_closure::hardware_defaults::generated_literal`, the same text the
//!   metrale-kernels build hashes into `METRALE_KERNEL_SET_HASH`, for the same tree.
//! - A missing tree or HARDWARE.toml yields the baseline (`hardware_defaults::read_defaults`),
//!   as it did under `METRALE_SKIP_BUILD` before the split.

use std::env;
use std::path::PathBuf;

use metrale_closure::hardware_defaults::{DEFAULT_HW, generated_literal};

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    println!("cargo:rerun-if-changed=build.rs");
    // 2026-10-17: metrale-kernel-root's `links` metadata names the directory holding `kernels/`:
    // the repository root in a checkout, that crate's own directory once published.
    let kernels_root = PathBuf::from(
        env::var("DEP_METRALE_KERNEL_TREE_ROOT")
            .expect("DEP_METRALE_KERNEL_TREE_ROOT: metrale-kernel-root is a dependency"),
    )
    .join("kernels");
    let hw = env::var("METRALE_TARGET_HW").unwrap_or_else(|_| DEFAULT_HW.into());
    let hw_toml = kernels_root.join(&hw).join("HARDWARE.toml");
    if hw_toml.exists() {
        println!("cargo:rerun-if-changed={}", hw_toml.display());
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let out = out_dir.join("target_defaults.rs");
    std::fs::write(&out, generated_literal(&kernels_root, &hw))
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", out.display()));
}
