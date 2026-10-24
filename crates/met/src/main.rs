// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-09-26: Entry point of the `met` binary.
//!
//! 2026-10-24: Builds the runtime and runs `metrale_server::serve_main` with the kernel
//! registry and kernel tree this binary embeds; the rest of `met` is the metrale-server
//! library.
//!
//! Owner: met (binary).
//! Invariants: the only crate that links metrale-kernels and metrale-kernel-tree into `met`.

fn main() -> anyhow::Result<()> {
    let embedded = metrale_server::embedded::Embedded {
        kernels: &metrale_kernels::CATALOG,
        kernel_tree_sha256: metrale_kernel_tree::SHA256,
        materialize_kernel_tree: metrale_kernel_tree::materialize,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(metrale_server::serve_main(embedded))
}
