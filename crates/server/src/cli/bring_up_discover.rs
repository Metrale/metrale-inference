// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Finding the local Metrale serve when `model-bring-up-bench` is given no `--url`:
//! every `met serve` process's own argv names its port (or takes `ServeArgs`'s default), and the
//! serve on that port must answer `GET /serve-config` with that process's pid and argv digest.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants:
//! - A port is never guessed: only a serve that reports its own identity on the port its argv
//!   names is used, and none or several is an error that lists what was found.
//! - Ranks above 0 of a multi-box serve are skipped; rank 0 serves the API.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};
use clap::CommandFactory as _;
use metrale_bench::TargetEndpoint;
use metrale_bench::serve_identity::{ServeIdentity, argv_fingerprint};

/// 2026-10-10: A `met serve` process, as its argv describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub pid: u32,
    pub port: u16,
    pub argv: Vec<String>,
    /// 2026-10-10: `--model-from-path`, from which the tokenizer can be derived.
    pub model_from_path: Option<PathBuf>,
}

/// 2026-10-10: The value of `--flag v` or `--flag=v` in `argv`.
fn flag(argv: &[String], name: &str) -> Option<String> {
    let eq = format!("{name}=");
    argv.iter().enumerate().find_map(|(i, a)| {
        if a == name {
            argv.get(i + 1).cloned()
        } else {
            a.strip_prefix(&eq).map(str::to_string)
        }
    })
}

/// 2026-10-10: `met serve`'s `--port` default, read from this binary's own clap definition.
pub(crate) fn default_port() -> Result<u16> {
    let cmd = super::Cli::command();
    let serve = cmd
        .find_subcommand("serve")
        .ok_or_else(|| anyhow::anyhow!("no serve command"))?;
    let port = serve
        .get_arguments()
        .find(|a| a.get_id() == "port")
        .and_then(|a| a.get_default_values().first())
        .and_then(|v| v.to_str())
        .ok_or_else(|| anyhow::anyhow!("serve --port has no default"))?;
    Ok(port.parse()?)
}

/// 2026-10-10: A process whose argv is `<…/met> serve …` at rank 0, with its port.
pub(crate) fn candidate(pid: u32, argv: Vec<String>, default_port: u16) -> Option<Candidate> {
    let program = argv.first().map(|p| p.rsplit('/').next().unwrap_or(p))?;
    if program != "met" || argv.get(1).map(String::as_str) != Some("serve") {
        return None;
    }
    if flag(&argv, "--rank").is_some_and(|r| r != "0") {
        return None;
    }
    let port = match flag(&argv, "--port") {
        Some(p) => p.parse().ok()?,
        None => default_port,
    };
    let model_from_path = flag(&argv, "--model-from-path").map(PathBuf::from);
    Some(Candidate {
        pid,
        port,
        argv,
        model_from_path,
    })
}

/// 2026-10-10: Whether a `/serve-config` answer is the candidate's own.
pub(crate) fn is_own(c: &Candidate, reported: &ServeIdentity) -> bool {
    reported.pid == c.pid && reported.argv_sha256 == argv_fingerprint(&c.argv[1..])
}

/// 2026-10-10: Every `met serve` candidate in `/proc`.
fn scan() -> Result<Vec<Candidate>> {
    let default = default_port()?;
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Ok(out);
    };
    for e in entries.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(raw) = std::fs::read(e.path().join("cmdline")) else {
            continue;
        };
        let argv: Vec<String> = raw
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        out.extend(candidate(pid, argv, default));
    }
    Ok(out)
}

/// 2026-10-10: The one local serve that confirms its identity, as a base URL.
pub(crate) async fn local_serve() -> Result<(String, Candidate)> {
    let found = scan()?;
    let mut verified = Vec::new();
    let mut refused = Vec::new();
    for c in found {
        let target = TargetEndpoint::local(c.port, "");
        let got =
            metrale_bench::http::get_json(&target, "/serve-config", Duration::from_secs(5)).await;
        match got.and_then(|v| Ok(serde_json::from_value::<ServeIdentity>(v)?)) {
            Ok(id) if is_own(&c, &id) => verified.push((target.base_url.clone(), c)),
            Ok(id) => refused.push(format!(
                "pid {} port {}: /serve-config answered for pid {}",
                c.pid, c.port, id.pid
            )),
            Err(e) => refused.push(format!("pid {} port {}: {e:#}", c.pid, c.port)),
        }
    }
    match verified.len() {
        1 => Ok(verified.remove(0)),
        0 => bail!(
            "no local Metrale serve confirmed its identity on its own port{}; pass --url",
            if refused.is_empty() {
                " (no `met serve` process is running)".to_string()
            } else {
                format!(": {}", refused.join("; "))
            }
        ),
        _ => bail!(
            "{} local Metrale serves answer ({}); pass --url to pick one",
            verified.len(),
            verified
                .iter()
                .map(|(u, c)| format!("{u} pid {}", c.pid))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
#[path = "bring_up_discover_tests.rs"]
mod tests;
