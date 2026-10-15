// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `diff`, `envdiff`, `verdict`: capture A vs B, the one-lever env
//! assert, the A/B verdict.

use regex::Regex;
use serde_json::{Map, Value, json};

use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::busy::{DEVICE, FAMILIES};
use crate::cli::{Command, Io, Run, load};
use crate::core::diff::{Side, capture_diff};
use crate::core::envdiff::{EnvChange, env_diff, one_lever, parse_env};
use crate::core::error::{Error, Result};
use crate::core::families::FamilyTable;
use crate::core::model::{Kernel, ms_to_ns};
use crate::core::repr::quote_list;
use crate::core::segment::split_requests;
use crate::core::verdict::decide;
use crate::fail;

const IGNORE: Opt = Opt {
    name: "ignore",
    kind: Kind::Append,
    required: false,
    metavar: "REGEX",
    help: "drop env keys matching this pattern before comparing",
};

const REQUEST_ARGS: [&str; 3] = ["gap-ms", "expect-a", "expect-b"];

pub const COMMANDS: &[&Command] = &[
    &Command {
        spec: Spec {
            name: "diff",
            about: "capture A vs B per family and per kernel; kernels only in A or only in B",
            positionals: &[
                ("a", "nsys sqlite export, control"),
                ("b", "nsys sqlite export, treatment"),
            ],
            opts: &[
                DEVICE,
                FAMILIES,
                Opt {
                    name: "per",
                    kind: Kind::Choice(&["window", "request"]),
                    required: true,
                    metavar: "{window,request}",
                    help: "normalize by the whole window or by the asserted request count",
                },
                Opt {
                    name: "gap-ms",
                    kind: Kind::Float,
                    required: false,
                    metavar: "MS",
                    help: "--per request: request boundary gap",
                },
                Opt {
                    name: "expect-a",
                    kind: Kind::Int,
                    required: false,
                    metavar: "N",
                    help: "--per request: requests in A",
                },
                Opt {
                    name: "expect-b",
                    kind: Kind::Int,
                    required: false,
                    metavar: "N",
                    help: "--per request: requests in B",
                },
                Opt {
                    name: "top",
                    kind: Kind::Int,
                    required: true,
                    metavar: "N",
                    help: "kernel rows, by |delta ms|",
                },
                Opt {
                    name: "env",
                    kind: Kind::FilePair,
                    required: false,
                    metavar: "ENV_A ENV_B",
                    help: "also assert that the two arms' environments differ in exactly one lever",
                },
                IGNORE,
            ],
        },
        run: Run::Report(diff),
    },
    &Command {
        spec: Spec {
            name: "envdiff",
            about: "assert two environments differ in exactly one lever",
            positionals: &[
                ("a", "KEY=VALUE lines or /proc/PID/environ content, control"),
                ("b", "the same, treatment"),
            ],
            opts: &[IGNORE],
        },
        run: Run::Report(envdiff),
    },
    &Command {
        spec: Spec {
            name: "verdict",
            about: "win / neutral / loss / unproven / invalid / parity-broken from an A/B record",
            positionals: &[],
            opts: &[Opt {
                name: "record",
                kind: Kind::File,
                required: true,
                metavar: "TOML",
                help: "TOML record of one A/B; every field is required",
            }],
        },
        run: Run::Report(verdict),
    },
];

fn lever_json(c: &EnvChange) -> Value {
    json!({"key": c.key, "a": c.a, "b": c.b})
}

fn lever(io: &Io, env: &[String], ignore: &[String]) -> Result<Value> {
    let mut patterns = Vec::with_capacity(ignore.len());
    for p in ignore {
        match Regex::new(p) {
            Ok(r) => patterns.push(r),
            Err(e) => fail!("--ignore {}: invalid regex: {e}", quote_list(&[p])),
        }
    }
    let mut envs = Vec::with_capacity(2);
    for p in env {
        envs.push(parse_env(&load::read_bytes(io, p)?, p)?);
    }
    let [a, b] = envs.as_slice() else {
        fail!("an environment diff needs exactly two environments")
    };
    Ok(lever_json(one_lever(&env_diff(a, b, &patterns))?))
}

fn side<'k>(
    name: &str,
    kernels: &'k [Kernel],
    labels: Vec<&'k str>,
    a: &Args,
    expect: Option<i64>,
) -> Result<Side<'k>> {
    let units = match (a.str("per")?, expect) {
        ("request", Some(n)) => split_requests(kernels, ms_to_ns(a.float("gap-ms")?), n)
            .map_err(|e| e.context(&format!("side {name}")))?
            .len() as i64,
        _ => 1,
    };
    Ok(Side {
        kernels,
        labels,
        units,
    })
}

fn window<'t>(
    io: &Io,
    name: &str,
    path: &str,
    a: &Args,
    table: &'t FamilyTable,
) -> Result<(Vec<Kernel>, Vec<&'t str>)> {
    load::classified_window(io, path, a.int("device")?, table)
        .map_err(|e: Error| e.context(&format!("side {name}")))
}

fn diff(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let per = a.str("per")?;
    let given: Vec<String> = REQUEST_ARGS
        .iter()
        .filter(|n| a.opt_str(n).is_some())
        .map(|n| format!("--{n}"))
        .collect();
    if per == "request" && given.len() != REQUEST_ARGS.len() {
        fail!("--per request needs --gap-ms, --expect-a and --expect-b");
    }
    if per == "window" && !given.is_empty() {
        fail!(
            "--per window takes no request arguments, got {}",
            quote_list(&given)
        );
    }
    let env = a.list("env");
    if !a.list("ignore").is_empty() && env.is_empty() {
        fail!("--ignore applies to --env only");
    }
    let lever = if env.is_empty() {
        None
    } else {
        Some(lever(io, env, a.list("ignore"))?)
    };
    let table = load::load_families(io, a.str("families")?)?;
    let (ka, la) = window(io, "A", a.pos(0)?, a, &table)?;
    let sa = side("A", &ka, la, a, a.opt_int("expect-a")?)?;
    let (kb, lb) = window(io, "B", a.pos(1)?, a, &table)?;
    let sb = side("B", &kb, lb, a, a.opt_int("expect-b")?)?;
    let mut out = Map::new();
    out.insert("per".to_string(), json!(per));
    if let Some(l) = lever {
        out.insert("lever".to_string(), l);
    }
    out.extend(capture_diff(&sa, &sb, a.int("top")?)?);
    Ok(out)
}

fn envdiff(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let env = [a.pos(0)?.to_string(), a.pos(1)?.to_string()];
    let mut out = Map::new();
    out.insert("lever".to_string(), lever(io, &env, a.list("ignore"))?);
    Ok(out)
}

fn verdict(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    decide(&load::read_toml(io, a.str("record")?)?)
}
