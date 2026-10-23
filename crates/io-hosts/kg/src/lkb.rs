// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The LKB facts no file holds (P1): the kernel inventory from the checkout's own
//! `kernel_perf.py --json`, and git dates (last commit per kernel file, date per commit).
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - Both commands run in the checkout with an explicit interpreter and `python -I`; neither
//!   writes to it.
//! - A command that fails is `KgSource` with its exit status and the last stderr line, never an
//!   empty inventory.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::lkb_derive::{LkbFacts, LkbSources};

/// 2026-10-23: Run `cmd`: its stdout, or (as a value, not an error) how it refused: the exit
/// status, the count of stderr lines and the first one. Only a command that cannot start is an
/// error.
fn run_outcome(what: &str, cmd: &mut Command) -> Result<core::result::Result<String, String>> {
    let out = cmd.output().map_err(|e| {
        error!(
            ErrorCode::KgSource,
            what.to_string(),
            format!("did not start: {}", e.kind())
        )
    })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = err.lines().filter(|l| !l.trim().is_empty()).collect();
        let first = lines.first().copied().unwrap_or("").trim();
        return Ok(Err(format!(
            "{}, {} stderr lines, first: {first}",
            out.status,
            lines.len()
        )));
    }
    String::from_utf8(out.stdout)
        .map(Ok)
        .map_err(|_| error!(ErrorCode::KgSource, what.to_string(), "printed non-UTF-8"))
}

/// 2026-10-23: [`run_outcome`] where a refusal is a host failure (git).
fn run(what: &str, cmd: &mut Command) -> Result<String> {
    run_outcome(what, cmd)?.map_err(|why| error!(ErrorCode::KgSource, what.to_string(), why))
}

/// 2026-10-23: `(file → (sha, date))` for every file under `dir` from one `git log --name-only`:
/// the first commit listing a file is its last change.
pub fn parse_name_log(log: &str) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    let mut cur: Option<(String, String)> = None;
    for l in log.lines() {
        if let Some(h) = l.strip_prefix('>') {
            cur = h
                .split_once(' ')
                .map(|(s, d)| (s.to_string(), d.to_string()));
        } else if let (false, Some(c)) = (l.is_empty(), &cur) {
            out.entry(l.to_string()).or_insert_with(|| c.clone());
        }
    }
    out
}

/// 2026-10-23: The freshness stamp of the LKB facts: the checkout's HEAD and a hash of its
/// dirty-file list under the hardware's kernels, so an edit or a commit there re-derives.
pub fn repo_stamp(root: &Path, cfg: &LkbSources) -> Result<(String, u64, u128)> {
    let repo = root.join(&cfg.repo);
    let head = run(
        "git rev-parse",
        Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["rev-parse", "HEAD"]),
    )?;
    let kernels_dir = format!("kernels/{}", cfg.hardware);
    let dirty = run(
        "git status",
        Command::new("git").arg("-C").arg(&repo).args([
            "status",
            "--porcelain",
            "--",
            &kernels_dir,
            "scripts",
        ]),
    )?;
    let d = metrale_kg_derive::hash::sha256(format!("{head}\n{dirty}").as_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&d.0[..16]);
    Ok((format!("{}@git", cfg.repo), 0, u128::from_le_bytes(b)))
}

/// 2026-10-23: Gather the facts for `cfg` under `root`.
pub fn facts(root: &Path, cfg: &LkbSources, python: &str) -> Result<LkbFacts> {
    let repo = root.join(&cfg.repo);
    let inventory = run_outcome(
        "kernel inventory",
        Command::new(python)
            .arg("-I")
            .arg(repo.join(&cfg.inventory))
            .args(["--json", "--partial"])
            .current_dir(&repo),
    )?;
    let kernels_dir = format!("kernels/{}", cfg.hardware);
    let log = run(
        "git log of kernel files",
        Command::new("git").arg("-C").arg(&repo).args([
            "log",
            "--format=>%H %ct",
            "--name-only",
            "--",
            &kernels_dir,
        ]),
    )?;
    let file_last = parse_name_log(&log);
    let all = run(
        "git log of commits",
        Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["log", "--all", "--format=%H %ct"]),
    )?;
    let commit_date = all
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    Ok(LkbFacts {
        inventory,
        file_last,
        commit_date,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_commit_naming_a_file_is_its_last_change() {
        let log = ">bbb 1792000000\n\nk/a.cu\n>aaa 1790000000\n\nk/a.cu\nk/b.cu\n";
        let m = parse_name_log(log);
        assert_eq!(m["k/a.cu"], ("bbb".into(), "1792000000".into()));
        assert_eq!(m["k/b.cu"].0, "aaa");
        assert_eq!(m.len(), 2);
    }
}
