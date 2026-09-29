// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: `met circuit venn`: the kernel Venn diagram of a model being added against models
//! the engine already supports (`.claude/skills/new-model/SKILL.md`, step 3). The I/O side of
//! `metrale_circuit::venn`: it reads the repository's TOMLs, kernel sources and, for a
//! checkpoint directory, its `config.json` / `hf_quant_config.json`, and writes or checks the
//! Markdown report.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here classifies or estimates: `metrale_circuit::venn::report_text` does, over the
//!   [`FsRepo`] this file supplies (the tests run it over the same tree).
//! - The repository files are read from the working tree, not embedded: a new model's circuit is
//!   edited and re-run without a rebuild.
//! - `--check` compares and never writes; a stale or missing report is an error.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_circuit::venn::{Repo, VennArgs, checkpoint_id_of, report_text};

use super::{CircuitMode, CircuitVennArgs};

/// 2026-09-29: The repository on disk.
pub(crate) struct FsRepo {
    root: PathBuf,
}

impl Repo for FsRepo {
    fn read(&self, rel: &str) -> std::result::Result<String, String> {
        std::fs::read_to_string(self.root.join(rel)).map_err(|e| format!("{rel}: {e}"))
    }

    fn list(&self, rel: &str) -> std::result::Result<Vec<String>, String> {
        let mut out = Vec::new();
        let mut stack = vec![self.root.join(rel)];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            for entry in entries {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(r) = path.strip_prefix(&self.root) {
                    out.push(r.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

fn find_root(start: &Path) -> Result<PathBuf> {
    start
        .ancestors()
        .find(|d| d.join("kernels/circuits/INSTANCES.toml").is_file())
        .map(Path::to_path_buf)
        .with_context(|| {
            format!(
                "no kernels/circuits/INSTANCES.toml above {}; pass --root",
                start.display()
            )
        })
}

fn mode_of(m: CircuitMode) -> metrale_circuit::Mode {
    match m {
        CircuitMode::Decode => metrale_circuit::Mode::Decode,
        CircuitMode::MultiSeq => metrale_circuit::Mode::MultiSeq,
        CircuitMode::Verify => metrale_circuit::Mode::Verify,
        CircuitMode::Draft => metrale_circuit::Mode::Draft,
    }
}

/// 2026-09-29: A checkpoint directory's `config.json` and, when it ships one,
/// `hf_quant_config.json`.
struct CheckpointTexts {
    config: String,
    hf_quant: Option<String>,
}

/// 2026-09-29: The arguments the crate takes, with a checkpoint directory replaced by the
/// checkpoint id it holds, plus that directory's config texts.
fn resolve_args(a: &CircuitVennArgs) -> Result<(VennArgs, Option<CheckpointTexts>)> {
    let dir = Path::new(&a.target);
    let (target, texts) = if dir.join("config.json").is_file() {
        let id = checkpoint_id_of(&a.target)
            .with_context(|| format!("cannot tell which checkpoint {} holds", a.target))?;
        let config = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading {}/config.json", a.target))?;
        let hfq = match std::fs::read_to_string(dir.join("hf_quant_config.json")) {
            Ok(t) => Some(t),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).context("reading hf_quant_config.json"),
        };
        (
            id,
            Some(CheckpointTexts {
                config,
                hf_quant: hfq,
            }),
        )
    } else {
        (a.target.clone(), None)
    };
    let args = VennArgs {
        target,
        against: a.against.clone(),
        modes: a.mode.iter().copied().map(mode_of).collect(),
        rows: a.rows.clone(),
        verify_rows: a.verify_rows.clone(),
        out: a.out.clone(),
    };
    Ok((args, texts))
}

/// 2026-09-29: Run `met circuit venn`.
pub(crate) fn run(a: CircuitVennArgs) -> Result<()> {
    let root = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let (args, texts) = resolve_args(&a)?;
    let checkpoint = texts
        .as_ref()
        .map(|t| (t.config.as_str(), t.hf_quant.as_deref()));
    let text = report_text(&FsRepo { root: root.clone() }, &args, checkpoint)?;
    let path = root.join(&a.out);
    if a.check {
        let on_disk = std::fs::read_to_string(&path)
            .with_context(|| format!("--check: reading {}", path.display()))?;
        if on_disk != text {
            let line = on_disk
                .lines()
                .zip(text.lines())
                .position(|(x, y)| x != y)
                .map_or_else(|| "the end".to_string(), |i| format!("line {}", i + 1));
            bail!(
                "{} is stale (first difference at {line}); regenerate it with `{}`",
                a.out,
                args.command()
            );
        }
        eprintln!("{}: current", a.out);
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {}", a.out);
    Ok(())
}

#[cfg(test)]
#[path = "circuit_venn_tests.rs"]
mod circuit_venn_tests;
