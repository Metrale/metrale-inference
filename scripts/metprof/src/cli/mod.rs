// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: command line `metprof <command>`. Every threshold is an explicit
//! argument or a config key; nothing analytical has a default.
//!
//! `run` is the whole program over six ports (traces, files, chat, salt, the
//! KB and the clock), so the tests drive exactly the code `main` runs, with
//! in-memory ports. Every command needs the global `--kb KB.TOML|off`; a
//! command run with a `kb.toml` records exactly one `RunRecord` (`record`).

pub mod args;
mod busy;
mod diff;
mod families;
pub mod kb;
mod load;
mod record;

pub use record::build_id;
mod segment;
mod slope;
mod tree;
mod ttft;

use metprof_kb::port::{Clock, Kb, KbOpen};
use serde_json::{Map, Value};

use crate::cli::args::{Args, Parsed, Spec};
use crate::cli::kb::{KbArg, KbOp, Setup};
use crate::core::error::Result;
use crate::core::report::render;
use crate::core::repr::quote;
use crate::fail;
use crate::ports::{ChatTransport, Files, SaltSource, TraceSource};

pub struct Io<'a> {
    pub traces: &'a dyn TraceSource,
    pub files: &'a dyn Files,
    pub chat: &'a dyn ChatTransport,
    pub salt: &'a dyn SaltSource,
    /// Opens the KB the `--kb` kb.toml names.
    pub kb: &'a dyn KbOpen,
    pub clock: &'a dyn Clock,
}

/// A command's result, and the files it asks the caller to write (path,
/// bytes). The core never writes; `main` does, after `run` returns.
pub struct Done {
    pub result: Map<String, Value>,
    pub files: Vec<(String, Vec<u8>)>,
}

#[derive(Clone, Copy)]
pub enum Run {
    /// A command that only reports.
    Report(fn(&Args, &Io) -> Result<Map<String, Value>>),
    /// A command that also produces files.
    Files(fn(&Args, &Io) -> Result<Done>),
    /// A `metprof kb` subcommand.
    Kb(KbOp),
}

pub struct Command {
    pub spec: Spec,
    pub run: Run,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
    /// Files to write (path, bytes), in order; empty on exit status 2.
    pub files: Vec<(String, Vec<u8>)>,
}

fn commands() -> Vec<&'static Command> {
    [
        busy::COMMANDS,
        families::COMMANDS,
        segment::COMMANDS,
        diff::COMMANDS,
        tree::COMMANDS,
        slope::COMMANDS,
        ttft::COMMANDS,
        kb::COMMANDS,
    ]
    .concat()
}

fn groups(cmds: &[&Command]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for c in cmds {
        let g = c.spec.name.split(' ').next().unwrap_or(c.spec.name);
        if !out.contains(&g) {
            out.push(g);
        }
    }
    out
}

fn top_help(cmds: &[&Command]) -> String {
    let mut out = vec![
        "usage: metprof --kb KB.TOML|off [--json] <command> ...".to_string(),
        String::new(),
        "profiling harness over nsys sqlite exports".to_string(),
        String::new(),
        "commands:".to_string(),
    ];
    out.extend(
        cmds.iter()
            .map(|c| format!("  {:<20}{}", c.spec.name, c.spec.about)),
    );
    out.push(String::new());
    out.push("`metprof <command> --help` describes a command's arguments.".to_string());
    out.push(
        "--kb names the kb.toml every run is recorded through; `--kb off` records nothing."
            .to_string(),
    );
    out.join("\n")
}

fn invalid_choice(what: &str, got: &str, choices: &[&str]) -> String {
    let list: Vec<String> = choices.iter().map(|c| quote(c)).collect();
    format!(
        "argument {what}: invalid choice: {} (choose from {})",
        quote(got),
        list.join(", ")
    )
}

/// Pick the command and parse its arguments. `Ok(None)` = help text.
fn select<'c>(
    argv: &[String],
    cmds: &[&'c Command],
    json: &mut bool,
) -> Result<(Option<&'c Command>, Parsed)> {
    let mut i = 0;
    let mut words: Vec<&str> = Vec::new();
    while let Some(t) = argv.get(i).map(String::as_str) {
        match t {
            "--json" => *json = true,
            "-h" | "--help" => {
                let prefix = words.join(" ");
                let sub: Vec<&Command> = cmds
                    .iter()
                    .copied()
                    .filter(|c| c.spec.name.starts_with(&prefix))
                    .collect();
                return Ok((
                    None,
                    Parsed::Help(top_help(if words.is_empty() { cmds } else { &sub })),
                ));
            }
            _ if t.starts_with('-') => fail!("unrecognized arguments: {t}"),
            _ => {
                words.push(t);
                let name = words.join(" ");
                if let Some(c) = cmds.iter().find(|c| c.spec.name == name) {
                    return Ok((Some(c), args::parse(&c.spec, &argv[i + 1..], json)?));
                }
                let prefix = format!("{name} ");
                let deeper: Vec<&str> = cmds
                    .iter()
                    .filter_map(|c| c.spec.name.strip_prefix(prefix.as_str()))
                    .collect();
                if deeper.is_empty() {
                    if words.len() == 1 {
                        fail!("{}", invalid_choice("command", t, &groups(cmds)));
                    }
                    let modes: Vec<&str> = cmds
                        .iter()
                        .filter_map(|c| {
                            c.spec
                                .name
                                .strip_prefix(&format!("{} ", words[..words.len() - 1].join(" ")))
                        })
                        .collect();
                    fail!("{}", invalid_choice("mode", t, &modes));
                }
            }
        }
        i += 1;
    }
    let what = if words.is_empty() { "command" } else { "mode" };
    fail!("the following arguments are required: {what}");
}

fn refused(stderr: &str, msg: &str) -> Outcome {
    Outcome {
        code: 2,
        stdout: String::new(),
        stderr: format!("{stderr}metprof: error: {msg}\n"),
        files: Vec::new(),
    }
}

/// Record the run in the KB: exit 0 when it is materialised, 3 (with the
/// reason on stderr, the analysis kept) when it is only spooled or not
/// recorded at all.
fn record_run(
    kb: &dyn Kb,
    setup: &Setup,
    io: &Io,
    cmd: &Command,
    a: &Args,
    done: &Done,
) -> (i32, String) {
    let rec = record::run_record(
        io,
        &setup.config.identity,
        cmd.spec.name,
        a,
        &done.result,
        &done.files,
    )
    .map_err(|e| e.0)
    .and_then(|r| kb.record(&r).map_err(|e| e.0));
    match rec {
        Ok(r) if r.materialized => (0, String::new()),
        Ok(r) => (
            3,
            format!(
                "metprof: kb: recorded in {} but not materialised: {}\n",
                r.spool_path,
                r.reason.as_deref().unwrap_or("no reason given")
            ),
        ),
        Err(e) => (3, format!("metprof: kb: this run is not recorded: {e}\n")),
    }
}

pub fn run(argv: &[String], io: &Io) -> Outcome {
    let cmds = commands();
    let mut json = false;
    let (kb_arg, argv) = match kb::split(argv) {
        Ok(x) => x,
        Err(e) => return refused("", &e.0),
    };
    let (cmd, parsed) = match select(&argv, &cmds, &mut json) {
        Ok(x) => x,
        Err(e) => return refused("", &e.0),
    };
    let (cmd, a) = match (cmd, parsed) {
        (_, Parsed::Help(text)) => {
            return Outcome {
                code: 0,
                stdout: format!("{text}\n"),
                stderr: String::new(),
                files: Vec::new(),
            };
        }
        (Some(cmd), Parsed::Args(a)) => (cmd, a),
        (None, Parsed::Args(_)) => {
            return refused("", "the following arguments are required: command");
        }
    };
    let (setup, notice) = match kb_arg {
        None => return refused("", kb::REQUIRED),
        Some(KbArg::Off) => (None, kb::OFF_NOTICE),
        Some(KbArg::Path(p)) => match kb::load(io, &p) {
            Ok(s) => (Some(s), ""),
            Err(e) => return refused("", &e.0),
        },
    };
    if let Run::Kb(op) = cmd.run {
        return kb::run_op(cmd.spec.name, op, setup.as_ref(), io, json);
    }
    // The KB is opened before the analysis, so a bad key or embedder is a
    // configuration error with nothing written.
    let opened = match &setup {
        Some(s) => match io.kb.open(&s.config, &s.embedding) {
            Ok(kb) => Some((kb, s)),
            Err(e) => return refused(notice, &format!("--kb: {}", e.0)),
        },
        None => None,
    };
    let done = match cmd.run {
        Run::Report(f) => f(&a, io).map(|result| Done {
            result,
            files: Vec::new(),
        }),
        Run::Files(f) => f(&a, io),
        Run::Kb(_) => return refused(notice, "a kb command reached the analysis"),
    };
    let (done, text) = match done.and_then(|d| {
        let text = render(&d.result, json)?;
        Ok((d, text))
    }) {
        Ok(x) => x,
        Err(e) => return refused(notice, &e.0),
    };
    let (code, stderr) = match &opened {
        Some((kb, setup)) => record_run(kb.as_ref(), setup, io, cmd, &a, &done),
        None => (0, notice.to_string()),
    };
    Outcome {
        code,
        stdout: format!("{text}\n"),
        stderr,
        files: done.files,
    }
}
