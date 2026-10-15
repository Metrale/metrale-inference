// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KB side of the command line (DESIGN §3.1): the required
//! global `--kb KB.TOML|off`, loading `kb.toml` and the `embedding.toml` it
//! names (through the `Files` port), and `metprof kb
//! selfcheck|replay|export|seal|verify`.
//!
//! Exit statuses: 0 = recorded and materialised; 2 = a usage or
//! configuration error (nothing written); 3 = recorded only, or not recorded
//! (the reason is on stderr; any analysis is still on stdout);
//! `VERIFY_FAILED` (4) = `kb verify` did not verify the sealed export: it is
//! missing, does not open (tampered with, or sealed under another key) or
//! does not seal the current export, or could not be read. A busy lease is
//! still 3.

use metprof_kb::config::{self as embedding_toml, EmbeddingConfig};
use metprof_kb::kb_toml::{self, KbConfig, SealConfig};
use metprof_kb::port::{KbReport, RUN_SCHEMA, RunRecord, UNKNOWN};
use serde_json::{Map, json};

use crate::cli::args::Spec;
use crate::cli::{Command, Io, Outcome, Run, record};
use crate::core::error::{Error, Result};
use crate::core::report::render;
use crate::fail;

pub const REQUIRED: &str = "--kb is required (kb.toml path, or 'off')";
pub const OFF_NOTICE: &str = "metprof: --kb off: this run is not recorded in the KB\n";
/// `kb verify` ran and the sealed export failed it (distinct from 3, a KB
/// that was busy or not in the state asked for).
pub const VERIFY_FAILED: i32 = 4;

/// The value of `--kb`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KbArg {
    Off,
    Path(String),
}

/// Take the global `--kb VALUE` (or `--kb=VALUE`) out of `argv`; it may stand
/// anywhere before a `--`. Given twice, or without a value, is an error.
pub fn split(argv: &[String]) -> Result<(Option<KbArg>, Vec<String>)> {
    let mut kb: Option<KbArg> = None;
    let mut rest = Vec::with_capacity(argv.len());
    let mut it = argv.iter();
    while let Some(t) = it.next() {
        let value = match t.as_str() {
            "--" => {
                rest.push(t.clone());
                rest.extend(it.by_ref().cloned());
                break;
            }
            "--kb" => match it.next() {
                Some(v) if !v.starts_with('-') => v.clone(),
                _ => fail!("argument --kb: expected one argument"),
            },
            _ => match t.strip_prefix("--kb=") {
                Some(v) => v.to_string(),
                None => {
                    rest.push(t.clone());
                    continue;
                }
            },
        };
        if kb.is_some() {
            fail!("argument --kb: given more than once");
        }
        if value.is_empty() {
            fail!("argument --kb: expected one argument");
        }
        kb = Some(if value == "off" {
            KbArg::Off
        } else {
            KbArg::Path(value)
        });
    }
    Ok((kb, rest))
}

/// A loaded `kb.toml` and its embedder.
pub struct Setup {
    pub config: KbConfig,
    pub embedding: EmbeddingConfig,
}

fn read_text(io: &Io, what: &str, path: &str) -> Result<String> {
    let bytes = io
        .files
        .read(path)
        .map_err(|e| Error(format!("--kb: cannot read {what} {path}: {e}")))?;
    String::from_utf8(bytes).map_err(|_| Error(format!("--kb: {what} {path} is not UTF-8")))
}

/// Read and check `kb.toml` and the `embedding.toml` it names.
pub fn load(io: &Io, path: &str) -> Result<Setup> {
    let config = kb_toml::parse(&read_text(io, "kb.toml", path)?)
        .map_err(|e| Error(format!("--kb {path}: {e}")))?;
    let emb_path = config.embedding.to_string_lossy().into_owned();
    let embedding = embedding_toml::parse(&read_text(io, "embedding.toml", &emb_path)?)
        .map_err(|e| Error(format!("--kb {path}: {e}")))?;
    Ok(Setup { config, embedding })
}

/// A `metprof kb` subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KbOp {
    Selfcheck,
    Replay,
    Export,
    Seal,
    Verify,
}

const fn kb_command(name: &'static str, about: &'static str, op: KbOp) -> Command {
    Command {
        spec: Spec {
            name,
            about,
            positionals: &[],
            opts: &[],
        },
        run: Run::Kb(op),
    }
}

pub const COMMANDS: &[&Command] = &[
    &kb_command(
        "kb selfcheck",
        "record a sentinel run through the whole write path and read it back",
        KbOp::Selfcheck,
    ),
    &kb_command(
        "kb replay",
        "materialise every spooled run (idempotent), export and seal",
        KbOp::Replay,
    ),
    &kb_command(
        "kb export",
        "export (and seal) the durable store as it is",
        KbOp::Export,
    ),
    &kb_command(
        "kb seal",
        "seal the current export under the local export key",
        KbOp::Seal,
    ),
    &kb_command(
        "kb verify",
        "open the sealed export and require it to be the current export",
        KbOp::Verify,
    ),
];

fn usage(name: &str, msg: String) -> Outcome {
    Outcome {
        code: 2,
        stdout: String::new(),
        stderr: format!("metprof: error: {name}: {msg}\n"),
        files: Vec::new(),
    }
}

fn incomplete(name: &str, stdout: String, reason: &str) -> Outcome {
    Outcome {
        code: 3,
        stdout,
        stderr: format!("metprof: {name}: {reason}\n"),
        files: Vec::new(),
    }
}

/// The sentinel `kb selfcheck` records: a run with no arguments and no
/// files, stamped like any other.
fn sentinel(io: &Io, setup: &Setup) -> Result<RunRecord> {
    let stamps = record::stamps(io, &setup.config.identity, UNKNOWN, None)?;
    let mut result = Map::new();
    result.insert("sentinel".into(), json!(stamps.recorded_at));
    Ok(RunRecord {
        schema: RUN_SCHEMA.to_string(),
        command: "kb selfcheck".to_string(),
        args: Vec::new(),
        result,
        files: Vec::new(),
        stamps,
    })
}

/// Run one `metprof kb` subcommand against the KB of `setup`.
pub fn run_op(name: &str, op: KbOp, setup: Option<&Setup>, io: &Io, json: bool) -> Outcome {
    let Some(setup) = setup else {
        return usage(name, "needs a kb.toml; --kb off names no KB".to_string());
    };
    if matches!(op, KbOp::Seal | KbOp::Verify) && setup.config.seal == SealConfig::Off {
        return usage(name, "kb.toml has seal = \"off\"".to_string());
    }
    let kb = match io.kb.open(&setup.config, &setup.embedding) {
        Ok(kb) => kb,
        Err(e) => return usage(name, e.0),
    };
    let report: std::result::Result<KbReport, String> = match op {
        KbOp::Selfcheck => match sentinel(io, setup) {
            Ok(s) => kb.selfcheck(&s).map_err(|e| e.0),
            Err(e) => Err(e.0),
        },
        KbOp::Replay => kb.replay().map_err(|e| e.0),
        KbOp::Export => kb.export().map_err(|e| e.0),
        KbOp::Seal => kb.seal().map_err(|e| e.0),
        KbOp::Verify => kb.verify().map_err(|e| e.0),
    };
    let report = match report {
        Ok(r) => r,
        Err(e) if op == KbOp::Verify => {
            return Outcome {
                code: VERIFY_FAILED,
                stdout: String::new(),
                stderr: format!("metprof: {name}: FAILED: {e}\n"),
                files: Vec::new(),
            };
        }
        Err(e) => return incomplete(name, String::new(), &e),
    };
    let stdout = match render(&report.result, json) {
        Ok(text) => format!("{text}\n"),
        Err(e) => return incomplete(name, String::new(), &e.0),
    };
    match report.incomplete {
        None => Outcome {
            code: 0,
            stdout,
            stderr: String::new(),
            files: Vec::new(),
        },
        Some(reason) => incomplete(name, stdout, &reason),
    }
}
