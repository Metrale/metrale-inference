// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Crate roles. Every workspace member declares one in its manifest:
//!
//! ```toml
//! [package.metadata.metrale]
//! role = "core"
//! ```
//!
//! | Role | May depend on (normal deps) | Purity |
//! |---|---|---|
//! | `abstract-io` | `abstract-io` | pure: `no_std`, PURE_ALLOWLIST only, wasm builds |
//! | `core` | `abstract-io`, `core` (dev-deps may add `host-mem`) | pure, as above |
//! | `host-mem` | `abstract-io`, `core` | std allowed; no runtime, socket or GPU |
//! | `host-std` | `abstract-io`, `core` | the only place for tokio, hyper, getrandom, fs/process |
//! | `host-worker` | `abstract-io`, `core` | the only place for wasm-bindgen and the browser APIs |
//! | `device` | `abstract-io`, `core`, `device` | FFI and GPU vendor crates allowed |
//! | `bin` | anything | the only place a host is chosen |
//! | `tool` | anything | unconstrained (xtask, bench, governance) |
//!
//! A member whose manifest is owned by another change may instead be listed in
//! `xtask/roles-pending.txt` as `package role`; the entry turns stale (and must go) once the
//! manifest declares that role, and a manifest declaring a different role is refused.

use crate::allow::{self, Kind, Problem};
use crate::manifest::{Manifest, Workspace};
use std::collections::BTreeMap;
use toml::Value;

/// 2026-10-17: The checked-in list of members whose role is not yet in their manifest.
pub const PENDING_FILE: &str = "xtask/roles-pending.txt";

const GATE: &str = "roles";

/// 2026-10-17: A crate's place in the dependency order; see the module table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    AbstractIo,
    Core,
    HostMem,
    HostStd,
    HostWorker,
    Device,
    Bin,
    Tool,
}

const ROLES: [(Role, &str); 8] = [
    (Role::AbstractIo, "abstract-io"),
    (Role::Core, "core"),
    (Role::HostMem, "host-mem"),
    (Role::HostStd, "host-std"),
    (Role::HostWorker, "host-worker"),
    (Role::Device, "device"),
    (Role::Bin, "bin"),
    (Role::Tool, "tool"),
];

impl Role {
    /// 2026-10-17: The role a manifest spells as `name`, if it is one.
    pub fn parse(name: &str) -> Option<Self> {
        ROLES.iter().find(|(_, n)| *n == name).map(|(r, _)| *r)
    }

    /// 2026-10-17: The role's manifest spelling.
    pub fn name(self) -> &'static str {
        ROLES
            .iter()
            .find(|(r, _)| *r == self)
            .map_or("?", |(_, n)| n)
    }

    /// 2026-10-17: Pure roles are `no_std`, build for wasm and depend only on PURE_ALLOWLIST.
    pub fn is_pure(self) -> bool {
        matches!(self, Self::AbstractIo | Self::Core)
    }

    /// 2026-10-17: Whether a normal dependency of a `self` crate may be a `to` crate.
    pub fn may_depend_on(self, to: Self) -> bool {
        match self {
            Self::AbstractIo => to == Self::AbstractIo,
            Self::Core | Self::HostMem | Self::HostStd | Self::HostWorker => {
                matches!(to, Self::AbstractIo | Self::Core)
            }
            Self::Device => matches!(to, Self::AbstractIo | Self::Core | Self::Device),
            Self::Bin | Self::Tool => true,
        }
    }

    /// 2026-10-17: Whether a dev-dependency of a pure `self` crate may be a `to` crate: what a
    /// normal one may be, plus the in-memory host the tests run on.
    pub fn may_dev_depend_on(self, to: Self) -> bool {
        self.may_depend_on(to) || to == Self::HostMem
    }
}

fn spellings() -> String {
    ROLES.iter().map(|(_, n)| *n).collect::<Vec<_>>().join(", ")
}

/// 2026-10-17: A workspace member with its resolved role and where its library root sits.
pub struct Member {
    pub name: String,
    pub dir: String,
    pub role: Role,
    pub lib_root: String,
}

/// 2026-10-17: Parse `xtask/roles-pending.txt`: `package role` per line.
pub fn parse_pending(file: &str, text: &str) -> Result<BTreeMap<String, Role>, String> {
    let mut out = BTreeMap::new();
    for (line_no, fields) in allow::records(text) {
        let [package, role] = fields.as_slice() else {
            return Err(format!(
                "{file}:{line_no}: expected `package role`, found `{}`",
                fields.join(" ")
            ));
        };
        let role = Role::parse(role).ok_or_else(|| {
            format!(
                "{file}:{line_no}: `{role}` is not a role; one of: {}",
                spellings()
            )
        })?;
        if out.insert((*package).to_owned(), role).is_some() {
            return Err(format!("{file}:{line_no}: `{package}` is listed twice"));
        }
    }
    Ok(out)
}

/// 2026-10-17: What a manifest says about its role: absent, a role, or something else.
enum Declared {
    Absent,
    Role(Role),
    Invalid(String),
}

fn declared(m: &Manifest) -> Declared {
    let Some(meta) = m
        .table
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|md| md.get("metrale"))
    else {
        return Declared::Absent;
    };
    match meta.get("role") {
        None => Declared::Absent,
        Some(Value::String(s)) => Role::parse(s).map_or_else(
            || Declared::Invalid(format!("role `{s}` is not one of: {}", spellings())),
            Declared::Role,
        ),
        Some(other) => Declared::Invalid(format!("`role` is a {}, not a string", other.type_str())),
    }
}

fn package_name(m: &Manifest) -> Result<String, String> {
    m.table
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{}: no [package] name", m.file))
}

fn lib_root(m: &Manifest, dir: &str) -> String {
    let rel = m
        .table
        .get("lib")
        .and_then(|l| l.get("path"))
        .and_then(Value::as_str)
        .unwrap_or("src/lib.rs");
    format!("{dir}/{rel}")
}

/// 2026-10-17: Resolve every member's role from its manifest or the pending list. Returns the
/// members that have a role and the problems the `roles` gate reports.
pub fn resolve(
    ws: &Workspace,
    pending: &BTreeMap<String, Role>,
) -> Result<(Vec<Member>, Vec<Problem>), String> {
    let mut members = Vec::new();
    let mut problems = Vec::new();
    let mut refuse = |kind, key: &str, detail: String| {
        problems.push(Problem {
            gate: GATE,
            kind,
            key: key.to_owned(),
            detail,
        });
    };
    let mut seen = Vec::new();
    for m in &ws.members {
        let name = package_name(m)?;
        let dir = m.file.trim_end_matches("Cargo.toml").trim_end_matches('/');
        let listed = pending.get(&name).copied();
        let role = match (declared(m), listed) {
            (Declared::Role(r), None) => Some(r),
            (Declared::Role(r), Some(p)) if r == p => {
                refuse(
                    Kind::Stale,
                    &name,
                    format!(
                        "{} now declares role `{}`; remove it from {PENDING_FILE}",
                        m.file,
                        r.name()
                    ),
                );
                Some(r)
            }
            (Declared::Role(r), Some(p)) => {
                refuse(
                    Kind::Refused,
                    &name,
                    format!(
                        "{} declares role `{}` but {PENDING_FILE} says `{}`",
                        m.file,
                        r.name(),
                        p.name()
                    ),
                );
                Some(r)
            }
            (Declared::Absent, Some(p)) => Some(p),
            (Declared::Absent, None) => {
                refuse(
                    Kind::Refused,
                    &name,
                    format!(
                        "{} declares no [package.metadata.metrale] role; one of: {}",
                        m.file,
                        spellings()
                    ),
                );
                None
            }
            (Declared::Invalid(why), _) => {
                refuse(Kind::Refused, &name, format!("{}: {why}", m.file));
                None
            }
        };
        if let Some(role) = role {
            members.push(Member {
                lib_root: lib_root(m, dir),
                name: name.clone(),
                dir: dir.to_owned(),
                role,
            });
        }
        seen.push(name);
    }
    for package in pending.keys() {
        if !seen.contains(package) {
            refuse(
                Kind::Stale,
                package,
                format!("is not a workspace member; remove it from {PENDING_FILE}"),
            );
        }
    }
    Ok((members, problems))
}

#[cfg(test)]
mod tests {
    use super::Role;

    #[test]
    fn every_role_round_trips_through_its_spelling() {
        for (role, name) in super::ROLES {
            assert_eq!(Role::parse(name), Some(role));
            assert_eq!(role.name(), name);
        }
        assert_eq!(Role::parse("Core"), None);
    }

    #[test]
    fn the_core_never_depends_on_a_host_device_bin_or_tool() {
        for to in [
            Role::HostMem,
            Role::HostStd,
            Role::HostWorker,
            Role::Device,
            Role::Bin,
            Role::Tool,
        ] {
            assert!(!Role::Core.may_depend_on(to), "{to:?}");
            assert!(!Role::AbstractIo.may_depend_on(to), "{to:?}");
        }
        assert!(!Role::AbstractIo.may_depend_on(Role::Core));
        assert!(Role::Core.may_dev_depend_on(Role::HostMem));
        assert!(!Role::Core.may_dev_depend_on(Role::HostStd));
        assert!(!Role::Device.may_depend_on(Role::HostStd));
        assert!(Role::Bin.may_depend_on(Role::HostStd));
    }
}
