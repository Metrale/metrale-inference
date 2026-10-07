// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Windows reserve for the `met` binary.
//!
//! A debug build of `met` puts a large clap tree on the main thread. The
//! default 1MB Windows stack overflows (`0xC00000FD`) before
//! `met circuit venn --help` returns. 16MB is the reserve that printed usage
//! on this host. Other targets keep the linker default.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if windows {
        println!("cargo:rustc-link-arg-bin=met=/STACK:16777216");
    }
}
