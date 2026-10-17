// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Build script for met: hands the crate the kernel tree's root as
//! `METRALE_KERNEL_TREE_ROOT` (metrale-kernel-tree's `links` metadata), so `cli::circuit`
//! embeds KERNEL_FAMILIES.toml from it and the published crate needs no repository.

fn main() {
    let root = std::env::var("DEP_METRALE_KERNEL_TREE_ROOT")
        .expect("DEP_METRALE_KERNEL_TREE_ROOT: metrale-kernel-tree is a dependency");
    println!("cargo:rustc-env=METRALE_KERNEL_TREE_ROOT={root}");
    println!("cargo:rerun-if-changed=build.rs");
}
