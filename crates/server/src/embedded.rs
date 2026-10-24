// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: What the `met` binary embeds and this library reads: the compiled kernel
//! registry (metrale-kernels) and the packed `kernels/` tree (metrale-kernel-tree).
//!
//! The library does not depend on those two crates; both rebuild on every kernel source edit,
//! and everything that depended on them rebuilt with them (DEV-LOOP-SPEED #7). The binary
//! hands them over once, as plain values, before anything runs ([`crate::serve_main`]).
//!
//! Owner: server.
//! Invariants:
//! - Installed once, by `serve_main`, before the CLI is parsed; never replaced.
//! - Reading it before then is a bug and panics, naming the missing install (unit tests
//!   read the real parts instead; see the `cfg(test)` [`get`]).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use metrale_kernel_set::KernelCatalog;

/// 2026-10-24: The binary's embedded kernel registry and kernel tree.
#[derive(Clone, Copy)]
pub struct Embedded {
    /// 2026-10-24: `metrale_kernels::CATALOG`.
    pub kernels: &'static KernelCatalog,
    /// 2026-10-24: `metrale_kernel_tree::SHA256`, the packed tree's identity.
    pub kernel_tree_sha256: &'static str,
    /// 2026-10-24: `metrale_kernel_tree::materialize`: unpack (or verify) the tree under a
    /// directory and return its root.
    pub materialize_kernel_tree: fn(&Path) -> anyhow::Result<PathBuf>,
}

static EMBEDDED: OnceLock<Embedded> = OnceLock::new();

/// 2026-10-24: Record the binary's embedded parts. A second install panics: there is one
/// binary, so a second value would be a second source of truth.
pub(crate) fn install(embedded: Embedded) {
    if EMBEDDED.set(embedded).is_err() {
        panic!("embedded::install called twice");
    }
}

/// 2026-10-24: The installed parts. Panics when `serve_main` has not installed them.
#[cfg(not(test))]
pub(crate) fn get() -> &'static Embedded {
    EMBEDDED
        .get()
        .expect("embedded::get before serve_main installed the binary's embedded kernels")
}

/// 2026-10-24: Unit tests run without `serve_main`; they read the same parts the binary
/// embeds (dev-dependencies), as they did when this crate linked them directly.
#[cfg(test)]
pub(crate) fn get() -> &'static Embedded {
    EMBEDDED.get_or_init(|| Embedded {
        kernels: &metrale_kernels::CATALOG,
        kernel_tree_sha256: metrale_kernel_tree::SHA256,
        materialize_kernel_tree: metrale_kernel_tree::materialize,
    })
}

/// 2026-10-24: The compiled kernel registry ([`Embedded::kernels`]).
pub(crate) fn kernels() -> &'static KernelCatalog {
    get().kernels
}
