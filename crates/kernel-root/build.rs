// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Build script for metrale-kernel-root: prints the directory holding `kernels/` as
//! the `root` links metadata. It reads no file under the tree, so it reruns only when this
//! script changes.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // 2026-10-17: The tree is `<crate>/kernels`: a directory in the published crate, and in a
    // checkout the committed `kernels -> ../../kernels` symlink, which canonicalizes to the
    // repository's own kernels/. A checkout made without symlinks (git `core.symlinks=false`
    // leaves the link a plain file) names the repository's directly.
    let linked = manifest.join("kernels");
    let kernels = if linked.is_dir() {
        linked
    } else {
        manifest.join("../../kernels")
    }
    .canonicalize()
    .expect("kernels/");
    let root = kernels.parent().expect("kernels/ has a parent");
    println!("cargo:root={}", root.display());
    println!("cargo:rerun-if-changed=build.rs");
}
