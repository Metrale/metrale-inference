// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the `RunRecord` of one command (DESIGN §3.1–3.2). A file
//! argument is recorded as its SHA-256 and base name, never as its path; a
//! directory as its base name. The stamps say when, by which metprof build,
//! on which family table and capture, and for whom (`kb.toml` [identity]).
//! A fact metprof cannot know is the explicit `unknown`.

use std::collections::HashMap;
use std::path::Path;

use metprof_kb::port::{Arg, FileRef, Identity, RUN_SCHEMA, RunRecord, Stamps, UNKNOWN};
use serde_json::{Map, Value};

use crate::cli::Io;
use crate::cli::args::{Args, Item, Kind};
use crate::core::error::{Error, Result};
use crate::core::provenance::sha256_hex;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The commit metprof was built from (`build.rs`), or `unknown`.
pub const GIT_SHA: &str = env!("METPROF_GIT_SHA");

/// This build, as the sealed export names its publisher.
pub fn build_id() -> String {
    format!("metprof {VERSION}+{GIT_SHA}")
}

/// The options whose file is the family table a result was computed with.
const CONFIG_OPTS: [&str; 2] = ["families", "config"];
/// The option or positional naming the capture.
const CAPTURE: &str = "capture";

pub fn stamps(
    io: &Io,
    identity: &Identity,
    config_sha256: &str,
    capture_sha256: Option<String>,
) -> Result<Stamps> {
    Ok(Stamps {
        recorded_at: io.clock.now_rfc3339().map_err(|e| Error(e.0))?,
        metprof_version: VERSION.to_string(),
        metprof_git_sha: GIT_SHA.to_string(),
        config_sha256: config_sha256.to_string(),
        capture_sha256,
        identity: identity.clone(),
    })
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| UNKNOWN.to_string(), |n| n.to_string_lossy().into_owned())
}

/// Digests of the files a run names, each read once. A file the command
/// writes is digested from the bytes it writes.
struct Digests<'a> {
    io: &'a Io<'a>,
    written: &'a [(String, Vec<u8>)],
    seen: HashMap<String, String>,
}

impl Digests<'_> {
    fn of(&mut self, path: &str) -> Result<String> {
        if let Some(d) = self.seen.get(path) {
            return Ok(d.clone());
        }
        let d = match self.written.iter().find(|(p, _)| p == path) {
            Some((_, bytes)) => sha256_hex(bytes),
            None => self
                .io
                .files
                .sha256(path)
                .map_err(|e| Error(format!("cannot digest {path}: {e}")))?,
        };
        self.seen.insert(path.to_string(), d.clone());
        Ok(d)
    }
}

/// Replace every string equal to a recorded path by its base name, so the
/// result keeps no path the arguments did not keep.
fn scrub(v: &mut Value, paths: &[String]) {
    match v {
        Value::String(s) if paths.contains(s) => *s = basename(s),
        Value::Array(xs) => xs.iter_mut().for_each(|x| scrub(x, paths)),
        Value::Object(m) => m.values_mut().for_each(|x| scrub(x, paths)),
        _ => {}
    }
}

/// The record of `command` run with `args`, its `result` and the files it
/// writes.
pub fn run_record(
    io: &Io,
    identity: &Identity,
    command: &str,
    args: &Args,
    result: &Map<String, Value>,
    written: &[(String, Vec<u8>)],
) -> Result<RunRecord> {
    let mut digests = Digests {
        io,
        written,
        seen: HashMap::new(),
    };
    let mut out = Vec::new();
    let mut paths = Vec::new();
    let (mut config, mut capture) = (None, None);
    let mut opt: Option<&str> = None;
    for item in args.items() {
        let (name, kind, value) = match item {
            Item::Opt(name) => {
                opt = Some(name);
                out.push(Arg::Word(format!("--{name}")));
                continue;
            }
            Item::Val(kind, v) => (opt, *kind, v),
            Item::Pos(name, v) => (Some(*name), Kind::File, v),
        };
        let arg = match kind {
            Kind::File | Kind::FilePair => {
                let sha256 = digests.of(value)?;
                if name.is_some_and(|n| CONFIG_OPTS.contains(&n)) {
                    config = Some(sha256.clone());
                }
                if name == Some(CAPTURE) {
                    capture = Some(sha256.clone());
                }
                Arg::File {
                    sha256,
                    basename: basename(value),
                }
            }
            Kind::Dir => Arg::Dir {
                basename: basename(value),
            },
            _ => {
                out.push(Arg::Word(value.clone()));
                continue;
            }
        };
        paths.push(value.clone());
        out.push(arg);
    }
    let files = written
        .iter()
        .map(|(path, bytes)| FileRef {
            name: basename(path),
            sha256: sha256_hex(bytes),
            // Only a JSON result file (the kernel tree) carries content; any
            // other file is recorded by its digest alone.
            json: serde_json::from_slice(bytes).ok(),
        })
        .collect();
    let mut result = result.clone();
    result.values_mut().for_each(|v| scrub(v, &paths));
    Ok(RunRecord {
        schema: RUN_SCHEMA.to_string(),
        command: command.to_string(),
        args: out,
        result,
        files,
        stamps: stamps(io, identity, config.as_deref().unwrap_or(UNKNOWN), capture)?,
    })
}
