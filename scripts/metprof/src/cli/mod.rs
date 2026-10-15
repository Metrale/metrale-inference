// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: command line `metprof <command>`. Every threshold is an explicit
//! argument or a config key; nothing analytical has a default.
//!
//! `run` is the whole program over two ports (traces, files), so the tests
//! drive exactly the code `main` runs, with in-memory ports.

pub mod args;
mod busy;
mod diff;
mod families;
mod load;
mod segment;
mod tree;

use serde_json::{Map, Value};

use crate::cli::args::{Args, Parsed, Spec};
use crate::core::error::Result;
use crate::core::report::render;
use crate::core::repr::quote;
use crate::fail;
use crate::ports::{Files, TraceSource};

pub struct Io<'a> {
    pub traces: &'a dyn TraceSource,
    pub files: &'a dyn Files,
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
    /// Files to write (path, bytes), in order; empty on any error.
    pub files: Vec<(String, Vec<u8>)>,
}

fn commands() -> Vec<&'static Command> {
    [
        busy::COMMANDS,
        families::COMMANDS,
        segment::COMMANDS,
        diff::COMMANDS,
        tree::COMMANDS,
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
        "usage: metprof [--json] <command> ...".to_string(),
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

pub fn run(argv: &[String], io: &Io) -> Outcome {
    let cmds = commands();
    let mut json = false;
    let err = |msg: String| Outcome {
        code: 2,
        stdout: String::new(),
        stderr: format!("metprof: error: {msg}\n"),
        files: Vec::new(),
    };
    let (cmd, parsed) = match select(argv, &cmds, &mut json) {
        Ok(x) => x,
        Err(e) => return err(e.0),
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
            return err("the following arguments are required: command".to_string());
        }
    };
    let done = match cmd.run {
        Run::Report(f) => f(&a, io).map(|result| Done {
            result,
            files: Vec::new(),
        }),
        Run::Files(f) => f(&a, io),
    };
    match done.and_then(|d| Ok((render(&d.result, json)?, d.files))) {
        Ok((text, files)) => Outcome {
            code: 0,
            stdout: format!("{text}\n"),
            stderr: String::new(),
            files,
        },
        Err(e) => err(e.0),
    }
}
