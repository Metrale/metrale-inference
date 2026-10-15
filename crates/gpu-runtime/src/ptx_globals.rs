// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The module-scope variables a PTX module defines, read from its text, so a
//! lookup of an optional device symbol (a kernel's published `<entry>_n_tile`) asks the
//! driver only for a symbol that exists. A `cuModuleGetGlobal_v2` that fails with
//! `CUDA_ERROR_NOT_FOUND` is a driver API error, and compute-sanitizer counts it.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - A name is listed exactly when a module-scope `.global` or `.const` definition names it,
//!   with or without `.visible`, `.weak` or `.common` linkage. An `.extern` declaration
//!   defines nothing and is not listed.
//! - Pure: no driver calls.

use std::collections::HashSet;

/// 2026-10-14: The names of the `.global` and `.const` variables `ptx` defines.
pub(crate) fn defined_globals(ptx: &str) -> HashSet<String> {
    ptx.lines().filter_map(defined_name).collect()
}

/// 2026-10-14: The variable one PTX line defines, if it is a `.global` or `.const` definition:
/// `[.visible|.weak|.common] (.global|.const) [.align N] [qualifiers] .type name[...] [= init];`.
fn defined_name(line: &str) -> Option<String> {
    let mut toks = line.split_whitespace().peekable();
    while toks
        .peek()
        .is_some_and(|t| matches!(*t, ".visible" | ".weak" | ".common"))
    {
        toks.next();
    }
    if !matches!(toks.next()?, ".global" | ".const") {
        return None;
    }
    while let Some(t) = toks.next() {
        if t == ".align" {
            toks.next();
        } else if !t.starts_with('.') {
            let name = t.split(['[', '=', ';']).next()?;
            return (!name.is_empty()).then(|| name.to_string());
        }
    }
    None
}

#[cfg(test)]
#[path = "ptx_globals_tests.rs"]
mod tests;
