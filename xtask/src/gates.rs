// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The purity gates, each a function of a workspace root so the selftest can point
//! it at a known-bad fixture workspace. A gate finds its offenders, then reconciles them with its
//! checked-in allowlist under `<root>/xtask/`; the roles gate reconciles with the pending list.

use crate::allow::{self, Finding, Findings, Problem};
use crate::depgraph;
use crate::filesize;
use crate::lints;
use crate::purity_io::{self, Built};
use crate::roles::{self, Member, Role};
use crate::shell;
use crate::vendor;
use std::path::Path;

/// 2026-10-17: The two wasm targets every pure crate builds for: one with `std`, one without.
pub const WASM_TARGETS: [&str; 2] = ["wasm32v1-none", "wasm32-unknown-unknown"];

/// 2026-10-17: One purity gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Roles,
    Depgraph,
    Wasm,
    FileSize,
    VendorBan,
    Lints,
}

/// 2026-10-17: Every gate, in the order `cargo xtask purity` runs them (cheap first).
pub const ALL: [Gate; 6] = [
    Gate::Roles,
    Gate::FileSize,
    Gate::VendorBan,
    Gate::Lints,
    Gate::Depgraph,
    Gate::Wasm,
];

impl Gate {
    /// 2026-10-17: The gate's subcommand name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Roles => "roles",
            Self::Depgraph => "depgraph",
            Self::Wasm => "wasm",
            Self::FileSize => "file-size",
            Self::VendorBan => "vendor-ban",
            Self::Lints => "lints",
        }
    }

    /// 2026-10-17: The gate named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        ALL.into_iter().find(|g| g.name() == name)
    }

    /// 2026-10-17: The gate's checked-in list, relative to the root.
    pub fn allow_file(self) -> &'static str {
        match self {
            Self::Roles => roles::PENDING_FILE,
            Self::Depgraph => "xtask/depgraph-allow.txt",
            Self::Wasm => "xtask/wasm-allow.txt",
            Self::FileSize => "xtask/file-size-allow.txt",
            Self::VendorBan => "xtask/vendor-ban-allow.txt",
            Self::Lints => "xtask/lints-allow.txt",
        }
    }

    fn counted(self) -> bool {
        matches!(self, Self::FileSize | Self::VendorBan)
    }
}

/// 2026-10-17: What a gate run found: its problems and how many entries its list holds.
pub struct Outcome {
    pub problems: Vec<Problem>,
    pub listed: usize,
}

/// 2026-10-17: Run `gate` on the workspace at `root`. `target_dir` sends the wasm builds to a
/// throwaway directory (the selftest); `None` leaves them in cargo's own.
pub fn run(gate: Gate, root: &Path, target_dir: Option<&Path>) -> Result<Outcome, String> {
    let ws = shell::load_workspace(root)?;
    let pending = roles::parse_pending(
        roles::PENDING_FILE,
        &purity_io::read_text(root, roles::PENDING_FILE)?,
    )?;
    let (members, role_problems) = roles::resolve(&ws, &pending)?;
    if gate == Gate::Roles {
        return Ok(Outcome {
            problems: role_problems,
            listed: pending.len(),
        });
    }
    let found = match gate {
        Gate::Roles => Findings::new(),
        Gate::Depgraph => {
            let graphs = [
                purity_io::cargo_metadata(root, WASM_TARGETS[0])?,
                purity_io::cargo_metadata(root, &purity_io::host_triple()?)?,
            ];
            depgraph::findings(&members, &graphs)?
        }
        Gate::Wasm => wasm(root, &members, target_dir)?,
        Gate::FileSize => file_size(root)?,
        Gate::VendorBan => vendor_ban(root, &members)?,
        Gate::Lints => lint_roots(root, &members)?,
    };
    let list = allow::parse(
        gate.allow_file(),
        &purity_io::read_text(root, gate.allow_file())?,
        gate.counted(),
    )?;
    Ok(Outcome {
        problems: allow::reconcile(gate.name(), &list, &found),
        listed: list.entries.len(),
    })
}

fn pure(members: &[Member]) -> impl Iterator<Item = &Member> {
    members.iter().filter(|m| m.role.is_pure())
}

fn wasm(root: &Path, members: &[Member], target_dir: Option<&Path>) -> Result<Findings, String> {
    let mut out = Findings::new();
    for m in pure(members) {
        for target in WASM_TARGETS {
            if let Built::Failed(why) = purity_io::cargo_build(root, &m.name, target, target_dir)? {
                out.insert(
                    format!("{}:{target}", m.name),
                    Finding {
                        count: None,
                        why: format!(
                            "the `{}` crate does not build for {target}: {why}",
                            m.role.name()
                        ),
                    },
                );
            }
        }
    }
    Ok(out)
}

fn file_size(root: &Path) -> Result<Findings, String> {
    let mut files = Vec::new();
    for path in purity_io::list_files(root)? {
        if filesize::in_scope(&path) && purity_io::exists(root, &path) {
            let bytes = purity_io::read_bytes(root, &path)?;
            files.push((path, bytes));
        }
    }
    filesize::findings(files.iter().map(|(p, b)| (p.as_str(), b.as_slice())))
}

/// 2026-10-17: The member that owns `path`: the one with the longest directory prefix.
fn owner<'a>(members: &'a [Member], path: &str) -> Option<&'a Member> {
    members
        .iter()
        .filter(|m| {
            path.strip_prefix(m.dir.as_str())
                .is_some_and(|rest| rest.starts_with('/'))
        })
        .max_by_key(|m| m.dir.len())
}

fn vendor_ban(root: &Path, members: &[Member]) -> Result<Findings, String> {
    let mut files = Vec::new();
    for path in purity_io::list_files(root)? {
        if !path.ends_with(".rs") || !purity_io::exists(root, &path) {
            continue;
        }
        let Some(m) = owner(members, &path) else {
            continue;
        };
        if matches!(m.role, Role::Device | Role::Tool) {
            continue;
        }
        let src = purity_io::read_text(root, &path)?;
        files.push((path, src, m.name.as_str(), m.role.name()));
    }
    Ok(vendor::findings(
        files
            .iter()
            .map(|(p, s, k, r)| (p.as_str(), s.as_str(), *k, *r)),
    ))
}

fn lint_roots(root: &Path, members: &[Member]) -> Result<Findings, String> {
    let mut roots = Vec::new();
    for m in pure(members) {
        let src = purity_io::read_text(root, &m.lib_root)?;
        roots.push((m.name.as_str(), m.lib_root.as_str(), src));
    }
    Ok(lints::findings(
        roots.iter().map(|(k, p, s)| (*k, *p, s.as_str())),
    ))
}

#[cfg(test)]
mod tests {
    use super::{ALL, Gate};

    // 2026-10-17: Each gate on the real tree, so `cargo test` refuses a new offender and a stale
    // or slack allowlist entry just as CI's `cargo xtask purity` does. The wasm gate is left to
    // CI and `cargo xtask wasm`: it builds every pure crate twice.
    #[test]
    fn the_repository_passes_every_gate_but_wasm_with_exact_allowlists() {
        let root = crate::shell::repo_root().unwrap();
        for gate in ALL.into_iter().filter(|g| *g != Gate::Wasm) {
            let outcome = super::run(gate, &root, None).unwrap();
            let listing: Vec<String> = outcome.problems.iter().map(ToString::to_string).collect();
            assert!(
                listing.is_empty(),
                "{}:\n{}",
                gate.name(),
                listing.join("\n")
            );
        }
    }

    #[test]
    fn every_gate_round_trips_through_its_name() {
        for gate in ALL {
            assert_eq!(Gate::parse(gate.name()), Some(gate));
        }
    }
}
