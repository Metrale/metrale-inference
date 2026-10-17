// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask depgraph`: the dependency order the roles promise, judged on the
//! graphs cargo resolves for `wasm32v1-none` and for the host.
//! - Every member's normal workspace dependencies follow `Role::may_depend_on`; a pure crate's
//!   dev-dependencies follow `Role::may_dev_depend_on`.
//! - Only `device` and `tool` crates name a GPU vendor crate directly (VBOP).
//! - A pure crate reaches, through its normal and dev dependencies, only external crates on
//!   PURE_ALLOWLIST, and none on PURE_DENYLIST; the walk stops at the first offender on a path.
//!   Workspace dependencies are not walked: each is judged as a member itself.
//!
//! Offenders are keyed `crate->dependency` in `xtask/depgraph-allow.txt`.

use crate::allow::{Finding, Findings};
use crate::metadata::Graph;
use crate::roles::{Member, Role};
use std::collections::{BTreeMap, VecDeque};

/// 2026-10-17: The external crates a pure crate may reach. Grows only by a change that gives a
/// one-line reason. `thiserror` is here because the wasm32v1-none build already refuses its
/// `std` feature, which is the only part a pure crate may not use.
pub const PURE_ALLOWLIST: &[&str] = &[
    "allocator-api2",
    "autocfg",
    "bit-set",
    "bit-vec",
    "block-buffer",
    "byteorder",
    "cfg-if",
    "cpufeatures",
    "crypto-common",
    "digest",
    "equivalent",
    "fancy-regex",
    "foldhash",
    "generic-array",
    "hash32",
    "hashbrown",
    "heapless",
    "itoa",
    "memchr",
    "num-traits",
    "paste",
    "postcard",
    "ppv-lite86",
    "proc-macro2",
    "quote",
    "rand_chacha",
    "rand_core",
    "regex-automata",
    "regex-syntax",
    "ryu",
    "serde",
    "serde_core",
    "serde_derive",
    "serde_json",
    "sha2",
    "stable_deref_trait",
    "syn",
    "thiserror",
    "thiserror-impl",
    "typenum",
    "unicode-ident",
    "zerocopy",
    "zerocopy-derive",
];

/// 2026-10-17: External crates a pure crate must never reach: std-only, I/O, GPU, blocking, and
/// the runtimes and executors (ACNB: a core crate holding one could pick its own executor).
pub const PURE_DENYLIST: &[&str] = &[
    "anyhow",
    "async-executor",
    "async-io",
    "async-std",
    "async-trait",
    "axum",
    "clap",
    "crossbeam",
    "crossterm",
    "cudarc",
    "futures",
    "futures-executor",
    "futures-lite",
    "futures-util",
    "getrandom",
    "hyper",
    "hyper-util",
    "image",
    "js-sys",
    "libc",
    "log",
    "memmap2",
    "metrale-cudarc",
    "mio",
    "nix",
    "onig",
    "onig_sys",
    "openssl",
    "parking_lot",
    "pollster",
    "rand",
    "ratatui",
    "rayon",
    "reqwest",
    "ring",
    "rustix",
    "rustls",
    "serde_yaml",
    "smol",
    "std",
    "tokio",
    "tokio-util",
    "toml",
    "tower",
    "tracing",
    "tracing-subscriber",
    "ureq",
    "walkdir",
    "wasm-bindgen",
    "wasm-bindgen-futures",
    "web-sys",
];

/// 2026-10-17: `(parent, child)` edges a pure crate may cross although `child` is denied: `libc`
/// only as `cpufeatures`' runtime CPU probe on host builds (absent from the wasm graph, which
/// the wasm gate proves).
pub const PURE_VIA: &[(&str, &str)] = &[("cpufeatures", "libc")];

/// 2026-10-17: GPU vendor bindings; only `device` (and unconstrained `tool`) crates name one.
/// `metrale-cudarc` is the cudarc fork under `vendor/cudarc`: outside the workspace, so cargo
/// reports it as an external package under that name, and this rule is how it is held to
/// `device` crates.
pub const VENDOR_CRATES: &[&str] = &[
    "cuda-sys",
    "cudarc",
    "cust",
    "hip-runtime-sys",
    "hip-sys",
    "metal",
    "metrale-cudarc",
    "nvml-wrapper",
    "nvml-wrapper-sys",
    "objc2-metal",
];

/// 2026-10-17: Every depgraph offender across `graphs`, keyed `crate->dependency`. The first
/// graph that shows an offender names it.
pub fn findings(members: &[Member], graphs: &[Graph]) -> Result<Findings, String> {
    let roles: BTreeMap<&str, Role> = members.iter().map(|m| (m.name.as_str(), m.role)).collect();
    let mut out = Findings::new();
    for graph in graphs {
        for m in members {
            let id = graph.member_id(&m.name).ok_or_else(|| {
                format!(
                    "depgraph: `{}` is not a member of the {} graph",
                    m.name, graph.platform
                )
            })?;
            judge_member(graph, &roles, m, id, &mut out)?;
        }
    }
    Ok(out)
}

fn offend(out: &mut Findings, from: &str, to: &str, why: String) {
    out.entry(format!("{from}->{to}"))
        .or_insert(Finding { count: None, why });
}

fn judge_member(
    graph: &Graph,
    roles: &BTreeMap<&str, Role>,
    m: &Member,
    id: &str,
    out: &mut Findings,
) -> Result<(), String> {
    let edges = graph
        .edges
        .get(id)
        .ok_or_else(|| format!("depgraph: `{}` has no node in the resolve", m.name))?;
    let platform = &graph.platform;
    let mut external = VecDeque::new();
    for edge in edges {
        let to = graph.name(&edge.to);
        let used = edge.normal || (edge.dev && m.role.is_pure());
        if !used {
            continue;
        }
        if graph.members.contains(&edge.to) {
            let to_role = *roles
                .get(to)
                .ok_or_else(|| format!("depgraph: workspace member `{to}` has no role"))?;
            let allowed = if edge.normal {
                m.role.may_depend_on(to_role)
            } else {
                m.role.may_dev_depend_on(to_role)
            };
            if !allowed {
                let kind = if edge.normal { "" } else { "dev-" };
                offend(
                    out,
                    &m.name,
                    to,
                    format!(
                        "a `{}` crate may not {kind}depend on the `{}` crate `{to}` ({platform})",
                        m.role.name(),
                        to_role.name()
                    ),
                );
            }
            continue;
        }
        if edge.normal
            && VENDOR_CRATES.contains(&to)
            && !matches!(m.role, Role::Device | Role::Tool)
        {
            offend(
                out,
                &m.name,
                to,
                format!(
                    "VBOP: only `device` crates name the GPU vendor crate `{to}`; `{}` is `{}` ({platform})",
                    m.name,
                    m.role.name()
                ),
            );
        }
        if m.role.is_pure() {
            external.push_back((
                m.name.clone(),
                edge.to.clone(),
                format!("{} -> {to}", m.name),
            ));
        }
    }
    walk_pure(graph, m, external, out);
    Ok(())
}

/// 2026-10-17: Breadth-first over the external crates a pure crate reaches by normal edges,
/// stopping at the first offender on each path.
fn walk_pure(
    graph: &Graph,
    m: &Member,
    mut queue: VecDeque<(String, String, String)>,
    out: &mut Findings,
) {
    let mut seen = Vec::new();
    while let Some((parent, id, path)) = queue.pop_front() {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id.clone());
        let name = graph.name(&id).to_owned();
        if PURE_VIA.contains(&(parent.as_str(), name.as_str())) {
            continue;
        }
        let why = if PURE_DENYLIST.contains(&name.as_str()) {
            Some("is on PURE_DENYLIST")
        } else if PURE_ALLOWLIST.contains(&name.as_str()) {
            None
        } else {
            Some("is not on PURE_ALLOWLIST")
        };
        if let Some(why) = why {
            offend(
                out,
                &m.name,
                &name,
                format!(
                    "the `{}` crate reaches `{name}`, which {why} ({path}; {})",
                    m.role.name(),
                    graph.platform
                ),
            );
            continue;
        }
        for edge in graph.edges.get(&id).into_iter().flatten() {
            if edge.normal && !graph.members.contains(&edge.to) {
                let next = graph.name(&edge.to);
                queue.push_back((name.clone(), edge.to.clone(), format!("{path} -> {next}")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PURE_ALLOWLIST, PURE_DENYLIST, VENDOR_CRATES};

    #[test]
    fn the_lists_are_sorted_disjoint_and_free_of_duplicates() {
        for list in [PURE_ALLOWLIST, PURE_DENYLIST, VENDOR_CRATES] {
            assert!(list.windows(2).all(|w| w[0] < w[1]), "{list:?}");
        }
        for name in PURE_ALLOWLIST {
            assert!(!PURE_DENYLIST.contains(name), "{name} is on both lists");
        }
    }
}
