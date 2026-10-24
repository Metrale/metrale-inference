// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Checked-in allowlists that may only shrink. Each purity gate finds its offenders
//! as keyed values; `reconcile` compares them with the gate's list under `xtask/` and refuses a
//! new offender, a counted entry that grew, a counted entry that shrank (the list is tightened in
//! the same change, so it never holds slack), and an entry that no longer matches anything.

use std::collections::BTreeMap;
use std::fmt;

/// 2026-10-17: Why a gate refused a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// An offender the allowlist does not hold.
    Refused,
    /// A counted offender above its allowlisted count.
    Grown,
    /// A counted offender below its allowlisted count: tighten the entry.
    Shrunk,
    /// An allowlist entry that matches no offender: remove it.
    Stale,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "refused",
            Self::Grown => "grown",
            Self::Shrunk => "shrunk",
            Self::Stale => "stale",
        })
    }
}

/// 2026-10-17: One refusal: the gate, why, the offender's key and a sentence for the reader.
#[derive(Debug)]
pub struct Problem {
    pub gate: &'static str,
    pub kind: Kind,
    pub key: String,
    pub detail: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] {}: {}",
            self.gate, self.kind, self.key, self.detail
        )
    }
}

/// 2026-10-17: An offender a gate found: its count for a counted gate, and why it offends.
pub struct Finding {
    pub count: Option<u64>,
    pub why: String,
}

/// 2026-10-17: A gate's offenders, keyed as its allowlist keys them.
pub type Findings = BTreeMap<String, Finding>;

/// 2026-10-17: A parsed allowlist: its path (for messages) and each key with its count.
pub struct Allowlist {
    pub file: String,
    pub entries: BTreeMap<String, Option<u64>>,
}

/// 2026-10-17: Parse `text`. One entry per line: `key` for an uncounted list, `key count` for a
/// counted one; `#` starts a comment. A duplicate key, a missing or extra field, or a count that
/// is not a number is an error, so a typo cannot widen the list.
pub fn parse(file: &str, text: &str, counted: bool) -> Result<Allowlist, String> {
    let mut entries = BTreeMap::new();
    for (line_no, fields) in records(text) {
        let line = fields.join(" ");
        let (key, count) = match (counted, fields.as_slice()) {
            (false, [key]) => (*key, None),
            (true, [key, count]) => {
                let n = count.parse::<u64>().map_err(|e| {
                    format!("{file}:{line_no}: count `{count}` is not a number: {e}")
                })?;
                (*key, Some(n))
            }
            _ => {
                let shape = if counted { "`key count`" } else { "`key`" };
                return Err(format!(
                    "{file}:{line_no}: expected {shape}, found `{line}`"
                ));
            }
        };
        if entries.insert(key.to_owned(), count).is_some() {
            return Err(format!("{file}:{line_no}: `{key}` is listed twice"));
        }
    }
    Ok(Allowlist {
        file: file.to_owned(),
        entries,
    })
}

/// 2026-10-17: Each entry line of a checked-in list: its 1-based number and its
/// whitespace-separated fields, with `#` comments and blank lines dropped.
pub fn records(text: &str) -> impl Iterator<Item = (usize, Vec<&str>)> {
    text.lines().enumerate().filter_map(|(index, raw)| {
        let fields: Vec<&str> = raw
            .split('#')
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .collect();
        (!fields.is_empty()).then(|| (index.saturating_add(1), fields))
    })
}

/// 2026-10-17: Compare what `gate` found with its allowlist; every mismatch is a problem.
pub fn reconcile(gate: &'static str, allow: &Allowlist, found: &Findings) -> Vec<Problem> {
    let mut out = Vec::new();
    let problem = |kind, key: &str, detail: String| Problem {
        gate,
        kind,
        key: key.to_owned(),
        detail,
    };
    for (key, finding) in found {
        match (allow.entries.get(key), finding.count) {
            (None, _) => out.push(problem(Kind::Refused, key, finding.why.clone())),
            (Some(Some(allowed)), Some(count)) if count > *allowed => out.push(problem(
                Kind::Grown,
                key,
                format!(
                    "grew from {allowed} to {count} ({}); {} lists it at {allowed} and it may only shrink",
                    finding.why, allow.file
                ),
            )),
            (Some(Some(allowed)), Some(count)) if count < *allowed => out.push(problem(
                Kind::Shrunk,
                key,
                format!(
                    "shrank from {allowed} to {count}; tighten {} to `{key} {count}`",
                    allow.file
                ),
            )),
            _ => {}
        }
    }
    for key in allow.entries.keys() {
        if !found.contains_key(key) {
            out.push(problem(
                Kind::Stale,
                key,
                format!("no longer offends; remove it from {}", allow.file),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(entries: &[(&str, Option<u64>)]) -> Findings {
        entries
            .iter()
            .map(|(k, c)| {
                (
                    (*k).to_owned(),
                    Finding {
                        count: *c,
                        why: "why".to_owned(),
                    },
                )
            })
            .collect()
    }

    fn kinds(problems: &[Problem]) -> Vec<(Kind, &str)> {
        problems.iter().map(|p| (p.kind, p.key.as_str())).collect()
    }

    #[test]
    fn a_counted_list_refuses_new_grown_shrunk_and_stale_and_passes_an_exact_match() {
        let allow = parse(
            "a.txt",
            "# header\nexact 10\ngrown 5 # note\nshrunk 7\nstale 3\n",
            true,
        )
        .unwrap();
        let got = reconcile(
            "g",
            &allow,
            &found(&[
                ("exact", Some(10)),
                ("grown", Some(6)),
                ("shrunk", Some(6)),
                ("new", Some(1)),
            ]),
        );
        assert_eq!(
            kinds(&got),
            vec![
                (Kind::Grown, "grown"),
                (Kind::Refused, "new"),
                (Kind::Shrunk, "shrunk"),
                (Kind::Stale, "stale"),
            ]
        );
        assert!(got[2].detail.contains("`shrunk 6`"), "{}", got[2].detail);
    }

    #[test]
    fn an_uncounted_list_refuses_only_new_and_stale_keys() {
        let allow = parse("a.txt", "kept\ngone\n", false).unwrap();
        let got = reconcile("g", &allow, &found(&[("kept", None), ("new", None)]));
        assert_eq!(
            kinds(&got),
            vec![(Kind::Refused, "new"), (Kind::Stale, "gone")]
        );
    }

    #[test]
    fn malformed_lines_are_errors_not_entries() {
        for (text, counted, want) in [
            ("a 1\na 2\n", true, "listed twice"),
            ("a\n", true, "expected `key count`"),
            ("a x\n", true, "is not a number"),
            ("a 1\n", false, "expected `key`"),
            ("a -1\n", true, "is not a number"),
        ] {
            let err = parse("a.txt", text, counted).err().unwrap_or_default();
            assert!(err.contains(want), "{text:?}: {err}");
        }
    }
}
