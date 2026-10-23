// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask depgraph`: the dependency order the roles promise, judged on the
//! graphs cargo resolves for `wasm32v1-none` and for the host.
//! - Every member's normal workspace dependencies follow `Role::may_depend_on`; a pure crate's
//!   dev-dependencies follow `Role::may_dev_depend_on`.
//! - Only `device` and `tool` crates name a GPU vendor crate directly (VBOP).
//! - A crate on LEAF_CRATES has no normal or build dependency at all.
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
    // 2026-10-22: miniz_oxide checksum; no_std.
    "adler2",
    // 2026-10-22: AEAD traits under aes-gcm (metrale-seal sealed kernels); no_std.
    "aead",
    // 2026-10-22: AES block cipher (fixsliced constant-time software backend on wasm) under
    // aes-gcm; no_std.
    "aes",
    // 2026-10-22: AES-256-GCM for sealed kernels (metrale-seal, DESIGN-encrypted-kernels.md §1.2);
    // no_std + alloc.
    "aes-gcm",
    // 2026-10-22: regex-automata's literal prefilter (metrale-precision `re:` targets, through
    // the workspace's unified `regex` features); no_std + alloc, no I/O.
    "aho-corasick",
    "allocator-api2",
    "autocfg",
    // 2026-10-22: The `kbp1.` blueprint transport; no_std + alloc, no I/O.
    "base64",
    // 2026-10-22: constant-time base64 under spki (ML-DSA key encoding); no_std.
    "base64ct",
    "bit-set",
    "bit-vec",
    "block-buffer",
    "byteorder",
    "cfg-if",
    // 2026-10-22: block/stream cipher traits under aes and ctr (metrale-seal); no_std.
    "cipher",
    // 2026-10-22: ctutils constant-time move; no_std, no I/O.
    "cmov",
    // 2026-10-22: postcard's COBS framing; no_std, no I/O.
    "cobs",
    // 2026-10-18: no_std OID constants that digest 0.11 names; no I/O, builds for wasm32v1-none.
    "const-oid",
    // 2026-10-22: compile-time word-size selection under aes 0.9 / polyval (metrale-seal); no_std,
    // no I/O.
    "cpubits",
    "cpufeatures",
    "crypto-common",
    // 2026-10-22: CTR mode under aes-gcm (metrale-seal); no_std.
    "ctr",
    // 2026-10-22: RustCrypto constant-time helpers under hybrid-array (sha2, ml-dsa); no_std.
    "ctutils",
    // 2026-10-22: Ed25519 group arithmetic (serial backend on wasm); no_std.
    "curve25519-dalek",
    // 2026-10-22: DER types under pkcs8 (ML-DSA key encoding), no I/O; no_std.
    "der",
    "digest",
    // 2026-10-22: Ed25519 signature types; no_std.
    "ed25519",
    // 2026-10-22: The Ed25519 half of the blueprint signature (2.2 as metrale-auth); no_std.
    "ed25519-dalek",
    // 2026-10-22: postcard's optional I/O traits, compiled as plain traits; no_std, no I/O.
    "embedded-io",
    "equivalent",
    "fancy-regex",
    "foldhash",
    "generic-array",
    // 2026-10-22: GHASH under aes-gcm (metrale-seal); no_std, constant-time software.
    "ghash",
    "hash32",
    "hashbrown",
    "heapless",
    // 2026-10-22: HKDF-SHA-256 (RFC 5869) for sealed-kernel keys (metrale-seal); no_std.
    "hkdf",
    // 2026-10-22: HMAC under hkdf (metrale-seal); no_std.
    "hmac",
    // 2026-10-18: The sha2 0.11 output array type (block-buffer); no_std, no I/O.
    "hybrid-array",
    // 2026-10-18: serde_json's ordered map; no_std + alloc over hashbrown and equivalent.
    "indexmap",
    // 2026-10-22: in/out buffer types under cipher (metrale-seal); no_std.
    "inout",
    "itoa",
    // 2026-10-22: Keccak permutation under SHAKE (ML-DSA); no_std.
    "keccak",
    // 2026-10-22: KEM traits under ml-kem (metrale-seal); no_std, its getrandom feature off.
    "kem",
    // 2026-10-22: num-traits float fallback under module-lattice; pure math, no_std.
    "libm",
    "memchr",
    // 2026-10-22: Pure-Rust no_std deflate for `.kbp` bodies and inline kernels (also
    // flate2's rust_backend, crates/kernel-tree/Cargo.toml).
    "miniz_oxide",
    // 2026-10-22: ML-DSA (FIPS 204), =0.1.1 as metrale-auth; no_std + alloc, deterministic.
    "ml-dsa",
    // 2026-10-22: ML-KEM-768 (FIPS 203) for the sealed-kernel hybrid wrap, 0.3.2 as metprof-
    // envelope (metrale-seal); no_std, deterministic encapsulation.
    "ml-kem",
    // 2026-10-22: ML-DSA polynomial arithmetic; no_std.
    "module-lattice",
    "num-traits",
    "paste",
    // 2026-10-22: ML-DSA key encoding types, never used for I/O; no_std.
    "pkcs8",
    // 2026-10-22: POLYVAL under ghash (metrale-seal); no_std, constant-time software.
    "polyval",
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
    // 2026-10-22: toml's span wrapper; no_std + alloc without `std`.
    "serde_spanned",
    "sha2",
    // 2026-10-22: SHA-3/SHAKE under ml-kem (metrale-seal); no_std.
    "sha3",
    // 2026-10-22: ML-DSA SHAKE XOF; no_std.
    "shake",
    // 2026-10-22: RustCrypto signature traits; no_std.
    "signature",
    // 2026-10-22: miniz_oxide checksum (portable fallback on wasm); no_std.
    "simd-adler32",
    // 2026-10-22: SPKI types under pkcs8, no I/O; no_std.
    "spki",
    // 2026-10-22: SHAKE sponge state (ML-DSA); no_std.
    "sponge-cursor",
    "stable_deref_trait",
    // 2026-10-22: constant-time comparisons (Ed25519); no_std.
    "subtle",
    "syn",
    "thiserror",
    "thiserror-impl",
    // 2026-10-22: toml 1.x is no_std + alloc without `std` (metrale-circuit parses the
    // circuit, rules and precision TOMLs it is handed); its own I/O helpers are behind `std`.
    "toml",
    // 2026-10-22: toml's datetime values; no_std without `std`.
    "toml_datetime",
    // 2026-10-22: toml's parser; no_std + alloc without `std`.
    "toml_parser",
    // 2026-10-22: toml's writer (`toml::to_string`); no_std + alloc without `std`.
    "toml_writer",
    "typenum",
    "unicode-ident",
    // 2026-10-22: universal-hash traits under ghash/polyval (metrale-seal); no_std.
    "universal-hash",
    // 2026-10-22: the parser combinators under toml_parser; no_std + alloc.
    "winnow",
    // 2026-10-22: the X25519 half of the sealed-kernel hybrid wrap (metrale-seal), on
    // curve25519-dalek 4; no_std, its getrandom feature off.
    "x25519-dalek",
    "zerocopy",
    "zerocopy-derive",
    // 2026-10-22: secure memory wiping (der, ed25519-dalek, the blueprint signer); no_std.
    "zeroize",
    // 2026-10-22: proc-macro for zeroize derives, enabled by x25519-dalek `zeroize` (metrale-seal);
    // build-time only, no I/O.
    "zeroize_derive",
    // 2026-10-18: serde_json's float formatter (the ryu successor); no_std, no I/O.
    "zmij",
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

/// 2026-10-17: Members that may have no normal or build dependency, workspace or external: the
/// design keeps `metrale-error` at the bottom of every graph, so each crate can depend on it.
/// Dev-dependencies stay allowed. A name here that is not a member is an error, so a rename
/// cannot retire the rule silently.
pub const LEAF_CRATES: &[&str] = &["metrale-error"];

/// 2026-10-23: Binaries kept apart from the serving stack, both ways: `(bin, members it may never
/// depend on)`, and no member may depend on the bin (delegated decision 2026-10-23: the knowledge
/// graph's `metrale-kg` never links met or metrale-bench, and nothing under them links it).
pub const ISOLATED_BINS: &[(&str, &[&str])] = &[("metrale-kg-mcp", &["met", "metrale-bench"])];

/// 2026-10-23: Why a normal or dev edge `from -> to` between members breaks [`ISOLATED_BINS`].
pub fn isolation(from: &str, to: &str) -> Option<String> {
    ISOLATED_BINS.iter().find_map(|(bin, banned)| {
        if to == *bin {
            Some(format!(
                "`{bin}` is an isolated binary; `{from}` may not depend on it"
            ))
        } else if from == *bin && banned.contains(&to) {
            Some(format!(
                "the isolated binary `{bin}` may not depend on `{to}`"
            ))
        } else {
            None
        }
    })
}

/// 2026-10-17: Every depgraph offender across `graphs`, keyed `crate->dependency`. The first
/// graph that shows an offender names it.
pub fn findings(members: &[Member], graphs: &[Graph]) -> Result<Findings, String> {
    let roles: BTreeMap<&str, Role> = members.iter().map(|m| (m.name.as_str(), m.role)).collect();
    if let Some(leaf) = LEAF_CRATES.iter().find(|l| !roles.contains_key(*l)) {
        return Err(format!(
            "depgraph: LEAF_CRATES names `{leaf}`, which is not a member"
        ));
    }
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
        if LEAF_CRATES.contains(&m.name.as_str()) && (edge.normal || edge.build) {
            let kind = if edge.normal { "normal" } else { "build" };
            offend(
                out,
                &m.name,
                to,
                format!(
                    "`{}` is a leaf and may have no normal or build dependency; `{to}` is a {kind} dependency ({platform})",
                    m.name
                ),
            );
        }
        if (edge.normal || edge.dev) && graph.members.contains(&edge.to) {
            if let Some(why) = isolation(&m.name, to) {
                offend(out, &m.name, to, format!("{why} ({platform})"));
            }
        }
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
    use super::{LEAF_CRATES, PURE_ALLOWLIST, PURE_DENYLIST, VENDOR_CRATES, isolation};

    #[test]
    fn the_isolated_kg_binary_is_refused_in_both_directions_and_only_there() {
        assert!(isolation("met", "metrale-kg-mcp").is_some());
        assert!(isolation("metrale-bench", "metrale-kg-mcp").is_some());
        assert!(isolation("metrale-kg-mcp", "met").is_some());
        assert!(isolation("metrale-kg-mcp", "metrale-bench").is_some());
        assert!(isolation("metrale-kg-mcp", "metrale-kg-derive").is_none());
        assert!(isolation("met", "metrale-kg-derive").is_none());
    }

    #[test]
    fn the_lists_are_sorted_disjoint_and_free_of_duplicates() {
        for list in [LEAF_CRATES, PURE_ALLOWLIST, PURE_DENYLIST, VENDOR_CRATES] {
            assert!(list.windows(2).all(|w| w[0] < w[1]), "{list:?}");
        }
        for name in PURE_ALLOWLIST {
            assert!(!PURE_DENYLIST.contains(name), "{name} is on both lists");
        }
    }
}
