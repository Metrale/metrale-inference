// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: A backend's module set with some modules replaced by PTX files from a directory
//! (the kernels before a change), for the gates and benches that compare against them in one
//! process.
//!
//! Owner: gpu-examples.
//! Invariants: none beyond the types.

use anyhow::Result;

/// 2026-10-21: `modules` with each module whose `<name>.ptx` is in `dir` replaced by that file
/// (leaked: a backend's modules are `'static`).
///
/// # Errors
/// No module of `modules` has a PTX file in `dir`.
pub fn with_base(
    modules: &[(&'static str, &'static [u8])],
    dir: &str,
) -> Result<Vec<(&'static str, &'static [u8])>> {
    let mut n = 0;
    let out = modules
        .iter()
        .map(
            |&(name, blob)| match std::fs::read(format!("{dir}/{name}.ptx")) {
                Ok(b) => {
                    n += 1;
                    (name, &*Box::leak(b.into_boxed_slice()))
                }
                Err(_) => (name, blob),
            },
        )
        .collect();
    anyhow::ensure!(n > 0, "{dir}: no module PTX found");
    Ok(out)
}
