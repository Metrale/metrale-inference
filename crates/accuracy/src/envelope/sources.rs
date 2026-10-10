// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The sources a family's entry points compile from, and their digest: the union of
//! its `[[family.point]] files` in KERNEL_FAMILIES.toml that are kernel sources (under
//! `kernels/`), sorted. SCHEDULES.toml records the digest at sweep time; a family whose digest
//! changed since is stale and the bake drops its schedules.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Business logic only: every byte arrives through [`Repo`] (SBIO).
//! - The digest is SHA-256 over, for each file in listed order, its repo-relative path's bytes
//!   then the file's bytes (the SPEC's definition, no separators).
//! - A family with no kernel source cannot be checked for staleness and is refused.

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::venn::{Repo, parse_families};
use sha2::{Digest, Sha256};

use super::schedules::{Schedules, SchedulesError, Source};

/// 2026-10-10: The digest of `files` (path, text) in the given order.
pub fn digest(files: &[(String, String)]) -> String {
    let mut h = Sha256::new();
    for (path, text) in files {
        h.update(path.as_bytes());
        h.update(text.as_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// 2026-10-10: The sources of each of `families` on `hw`, read now.
pub fn sources_of(
    repo: &dyn Repo,
    hw: &str,
    families: &[&str],
) -> Result<BTreeMap<String, Source>, SchedulesError> {
    let manifest = format!("kernels/{hw}/common/KERNEL_FAMILIES.toml");
    let fams = parse_families(&repo.read(&manifest).map_err(SchedulesError::Load)?)
        .map_err(|e| SchedulesError::Load(format!("{manifest}: {e}")))?;
    let mut out = BTreeMap::new();
    for &id in families {
        let fam =
            fams.families
                .iter()
                .find(|f| f.id == id)
                .ok_or_else(|| SchedulesError::Invalid {
                    at: format!("sources.{id}"),
                    problem: format!("no family `{id}` in {manifest}"),
                })?;
        let files: BTreeSet<&String> = fam
            .points
            .iter()
            .flat_map(|p| &p.files)
            .filter(|f| f.starts_with("kernels/"))
            .collect();
        if files.is_empty() {
            return Err(SchedulesError::Invalid {
                at: format!("sources.{id}"),
                problem: "the family's points name no kernel source under kernels/".into(),
            });
        }
        let mut read = Vec::with_capacity(files.len());
        for f in &files {
            read.push(((*f).clone(), repo.read(f).map_err(SchedulesError::Load)?));
        }
        out.insert(
            id.to_string(),
            Source {
                files: files.into_iter().cloned().collect(),
                sha256: digest(&read),
            },
        );
    }
    Ok(out)
}

/// 2026-10-10: The families of `file` whose sources differ from `now` (another file list or
/// another digest), or that `now` no longer has: their schedules are stale.
pub fn stale(now: &BTreeMap<String, Source>, file: &Schedules) -> Vec<String> {
    file.sources
        .iter()
        .filter(|(fam, src)| now.get(*fam) != Some(*src))
        .map(|(fam, _)| fam.clone())
        .collect()
}

#[cfg(test)]
#[path = "sources_tests.rs"]
mod tests;
