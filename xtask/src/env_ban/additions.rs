// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The allowlist may only shrink. An entry in the allowlist being judged that is
//! not in the merge-base's is an addition, and the commit that introduced it must carry an
//! `Env-Ban-Allow: <reason>` trailer. Pure: the merge-base's allowlist and each commit that
//! changed it in, a verdict out.
//!
//! When the merge-base has no allowlist, the range introduces the gate: the commit that creates
//! the file is its inventory, and the entries it adds need no trailer. Every later addition
//! does. Once the file is on the base branch this mode cannot fire again.

use super::judge::entries;
use std::collections::BTreeSet;
use std::fmt;

/// 2026-10-17: The trailer key a commit names its reason under.
pub const TRAILER: &str = "Env-Ban-Allow";

/// 2026-10-17: One commit that changed the allowlist: its sha, the file across its parents and
/// at the commit (`None` where absent), and its `Env-Ban-Allow` trailer value (empty if none).
pub struct Change {
    pub sha: String,
    pub before: Option<String>,
    pub after: Option<String>,
    pub reason: String,
}

/// 2026-10-17: An added entry and the commit that introduced it (`None`: no commit in the
/// range did, e.g. an uncommitted edit).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Added {
    pub entry: String,
    pub sha: Option<String>,
}

/// 2026-10-17: The additions, split by how they are accounted for.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// 2026-10-17: `(sha, entries)` of the inventory commit, when the merge-base has no list.
    pub inventory: Option<(String, usize)>,
    pub explained: Vec<(Added, String)>,
    pub unexplained: Vec<Added>,
}

fn set(text: Option<&str>) -> BTreeSet<&str> {
    text.map(|t| entries(t).map(|(_, e)| e).collect())
        .unwrap_or_default()
}

/// 2026-10-17: The verdict on `current` (the allowlist being judged) against the merge-base's
/// list `base` and the `changes` in the range, oldest first.
pub fn judge(base: Option<&str>, changes: &[Change], current: &str) -> Verdict {
    let base_set = set(base);
    let inventory = changes
        .first()
        .filter(|c| base.is_none() && c.before.is_none() && c.after.is_some());
    let mut verdict = Verdict {
        inventory: inventory.map(|c| (c.sha.clone(), 0)),
        ..Verdict::default()
    };
    for entry in set(Some(current)).difference(&base_set) {
        let introducer = changes.iter().rev().find(|c| {
            !set(c.before.as_deref()).contains(entry) && set(c.after.as_deref()).contains(entry)
        });
        let added = Added {
            entry: (*entry).to_owned(),
            sha: introducer.map(|c| c.sha.clone()),
        };
        match (introducer, &mut verdict.inventory) {
            (Some(c), Some((sha, n))) if *sha == c.sha => *n += 1,
            (Some(c), _) if !c.reason.trim().is_empty() => {
                verdict.explained.push((added, c.reason.trim().to_owned()));
            }
            _ => verdict.unexplained.push(added),
        }
    }
    verdict
}

impl fmt::Display for Added {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = super::ALLOWLIST;
        match &self.sha {
            Some(sha) => write!(
                f,
                "{list}: `{}` was added by {sha} without an `{TRAILER}: <reason>` trailer; the list \
                 may only shrink unless a commit says why",
                self.entry
            ),
            None => write!(
                f,
                "{list}: `{}` is not in the merge-base's list and no commit in the range added it; \
                 commit it with an `{TRAILER}: <reason>` trailer",
                self.entry
            ),
        }
    }
}

#[cfg(test)]
#[path = "additions_tests.rs"]
mod tests;
