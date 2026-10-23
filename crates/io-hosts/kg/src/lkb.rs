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

fn head(repo: &Path) -> Result<String> {
    Ok(run(
        "git rev-parse",
        Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["rev-parse", "HEAD"]),
    )?
    .trim()
    .to_string())
}

/// 2026-10-23: `metrale-kg refresh`: move the read-only LKB checkout to `<remote>/<branch>`
/// (`git fetch`, then a detached checkout). Refuses a checkout with tracked changes, which would
/// mean it is somebody's working tree, not a source. Returns (old HEAD, new HEAD).
pub fn refresh(root: &Path, cfg: &LkbSources) -> Result<(String, String)> {
    let repo = root.join(&cfg.repo);
    let dirty = run(
        "git status",
        Command::new("git").arg("-C").arg(&repo).args([
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]),
    )?;
    if !dirty.trim().is_empty() {
        return Err(error!(
            ErrorCode::KgSource,
            cfg.repo.clone(),
            format!(
                "has {} tracked changes; refresh only moves a clean read-only checkout",
                dirty.lines().count()
            )
        ));
    }
    let old = head(&repo)?;
    run(
        "git fetch",
        Command::new("git").arg("-C").arg(&repo).args([
            "fetch",
            "--quiet",
            &cfg.remote,
            &cfg.branch,
        ]),
    )?;
    let target = format!("refs/remotes/{}/{}", cfg.remote, cfg.branch);
    run(
        "git checkout",
        Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["checkout", "--quiet", "--detach", &target]),
    )?;
    Ok((old, head(&repo)?))
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

/// 2026-10-23: [`facts`] through a cache in `cache_dir`, keyed by everything the facts are a
/// function of: the checkout's HEAD, its full `git status` (tracked changes and untracked files)
/// and every ref (a fetch adds commits that date measurements). The cache is derived and
/// rebuildable; only the newest entry is kept. A first derive after a refresh pays the inventory.
pub fn facts_cached(
    root: &Path,
    cfg: &LkbSources,
    python: &str,
    cache_dir: &Path,
) -> Result<LkbFacts> {
    let repo = root.join(&cfg.repo);
    let git =
        |what: &str, args: &[&str]| run(what, Command::new("git").arg("-C").arg(&repo).args(args));
    let key_src = format!(
        "{}\n{}\n{}\n{}\n{}",
        head(&repo)?,
        git(
            "git status",
            &["status", "--porcelain", "--untracked-files=all"]
        )?,
        git(
            "git for-each-ref",
            &["for-each-ref", "--format=%(objectname) %(refname)"]
        )?,
        cfg.inventory,
        cfg.hardware
    );
    let key = metrale_kg_derive::hash::sha256(key_src.as_bytes()).hex();
    let path = cache_dir.join(format!("lkb-facts-{key}.json"));
    if let Some(f) = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| from_json(&t))
    {
        return Ok(f);
    }
    let f = facts(root, cfg, python)?;
    let shown = cache_dir.display().to_string();
    let werr = |e: std::io::Error| error!(ErrorCode::KgStore, shown.clone(), e.kind().to_string());
    std::fs::create_dir_all(cache_dir).map_err(werr)?;
    let tmp = cache_dir.join(format!(".lkb-facts-{key}.{}", std::process::id()));
    std::fs::write(&tmp, to_json(&f)).map_err(werr)?;
    std::fs::rename(&tmp, &path).map_err(werr)?;
    if let Ok(rd) = std::fs::read_dir(cache_dir) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("lkb-facts-") && e.path() != path {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    Ok(f)
}

fn to_json(f: &LkbFacts) -> String {
    use serde_json::{Value, json};
    let pairs = |m: &BTreeMap<String, (String, String)>| -> Value {
        m.iter()
            .map(|(k, (a, b))| (k.clone(), json!([a, b])))
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    let inv = match &f.inventory {
        Ok(s) => json!({"ok": s}),
        Err(s) => json!({"refused": s}),
    };
    json!({"head": f.head, "inventory": inv, "file_last": pairs(&f.file_last), "commit_date": f.commit_date}).to_string()
}

fn from_json(t: &str) -> Option<LkbFacts> {
    use serde_json::Value;
    let v: Value = serde_json::from_str(t).ok()?;
    let inv = &v["inventory"];
    let inventory = match (inv["ok"].as_str(), inv["refused"].as_str()) {
        (Some(s), None) => Ok(s.to_string()),
        (None, Some(s)) => Err(s.to_string()),
        _ => return None,
    };
    let mut file_last = BTreeMap::new();
    for (k, p) in v["file_last"].as_object()? {
        file_last.insert(
            k.clone(),
            (p[0].as_str()?.to_string(), p[1].as_str()?.to_string()),
        );
    }
    let mut commit_date = BTreeMap::new();
    for (k, d) in v["commit_date"].as_object()? {
        commit_date.insert(k.clone(), d.as_str()?.to_string());
    }
    Some(LkbFacts {
        head: v["head"].as_str()?.to_string(),
        inventory,
        file_last,
        commit_date,
    })
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
    let head = head(&repo)?;
    Ok(LkbFacts {
        head,
        inventory,
        file_last,
        commit_date,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_facts_round_trip_both_inventory_outcomes() {
        for inventory in [
            Ok("{\"kernels\":[]}".to_string()),
            Err("exit status: 2".to_string()),
        ] {
            let f = LkbFacts {
                head: "abc".into(),
                inventory,
                file_last: [("k.cu".to_string(), ("s".to_string(), "1".to_string()))].into(),
                commit_date: [("s".to_string(), "1".to_string())].into(),
            };
            assert_eq!(from_json(&to_json(&f)), Some(f));
        }
        assert_eq!(from_json("{\"head\":\"x\"}"), None);
    }

    #[test]
    fn the_newest_commit_naming_a_file_is_its_last_change() {
        let log = ">bbb 1792000000\n\nk/a.cu\n>aaa 1790000000\n\nk/a.cu\nk/b.cu\n";
        let m = parse_name_log(log);
        assert_eq!(m["k/a.cu"], ("bbb".into(), "1792000000".into()));
        assert_eq!(m["k/b.cu"].0, "aaa");
        assert_eq!(m.len(), 2);
    }
}
