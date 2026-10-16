// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: a small argument parser (no dependency). Options are long
//! (`--name VALUE` or `--name=VALUE`), never abbreviated; a value may be a
//! negative number. Typed values are checked while parsing, so a command never
//! sees a malformed number. Nothing analytical has a default: an option is
//! either required or optional-and-absent.

use std::collections::HashMap;

use crate::core::error::Result;
use crate::core::repr::quote;
use crate::fail;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Int,
    Float,
    Str,
    /// A path to one file (recorded in the KB as its digest and base name).
    File,
    /// A path to a directory (recorded as its base name).
    Dir,
    Choice(&'static [&'static str]),
    /// Repeatable, each occurrence takes one value.
    Append,
    /// One occurrence taking exactly two file paths.
    FilePair,
    /// Repeatable, each occurrence one `ROWS=PATH`: a whole number of rows
    /// and a path to one file (recorded as the rows and the file's digest
    /// and base name).
    RowsFile,
}

#[derive(Debug, Clone, Copy)]
pub struct Opt {
    pub name: &'static str,
    pub kind: Kind,
    pub required: bool,
    pub metavar: &'static str,
    pub help: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct Spec {
    /// The command words, e.g. `segment decode`.
    pub name: &'static str,
    pub about: &'static str,
    /// Every positional argument is a file path (the KB records it as a
    /// digest, never as a path).
    pub positionals: &'static [(&'static str, &'static str)],
    pub opts: &'static [Opt],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pos: Vec<String>,
    vals: HashMap<&'static str, Vec<String>>,
    items: Vec<Item>,
}

/// One argument as given, in order (`--name=value` as two items; `--json`,
/// `--help` and `--` are not arguments of the run).
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Opt(&'static str),
    Val(Kind, String),
    /// A positional argument, by its name.
    Pos(&'static str, String),
}

pub enum Parsed {
    Help(String),
    Args(Args),
}

fn is_number(s: &str) -> bool {
    s.parse::<f64>().is_ok()
}

/// `ROWS=PATH` split at the first `=`: rows a whole number of at least 1,
/// path non-empty.
pub fn rows_file(v: &str) -> Result<(u64, &str)> {
    match v.split_once('=') {
        Some((rows, path)) if !path.is_empty() => match rows.parse::<u64>() {
            Ok(r) if r >= 1 => Ok((r, path)),
            _ => fail!(
                "{}: ROWS must be a whole number of rows, at least 1, got {}",
                quote(v),
                quote(rows)
            ),
        },
        _ => fail!("{}: expected ROWS=PATH", quote(v)),
    }
}

fn check(opt: &Opt, v: &str) -> Result<()> {
    if opt.kind == Kind::RowsFile {
        if let Err(e) = rows_file(v) {
            fail!("argument --{}: {}", opt.name, e.0);
        }
        return Ok(());
    }
    let ok = match opt.kind {
        Kind::Int => v.parse::<i64>().is_ok(),
        Kind::Float => v.parse::<f64>().is_ok(),
        Kind::Choice(cs) => {
            if !cs.contains(&v) {
                let list: Vec<String> = cs.iter().map(|c| quote(c)).collect();
                fail!(
                    "argument --{}: invalid choice: {} (choose from {})",
                    opt.name,
                    quote(v),
                    list.join(", ")
                );
            }
            true
        }
        Kind::Str | Kind::File | Kind::Dir | Kind::Append | Kind::FilePair | Kind::RowsFile => true,
    };
    if !ok {
        let what = if opt.kind == Kind::Int {
            "int"
        } else {
            "float"
        };
        fail!(
            "argument --{}: invalid {what} value: {}",
            opt.name,
            quote(v)
        );
    }
    Ok(())
}

pub fn help(spec: &Spec) -> String {
    let mut usage = format!("usage: metprof {}", spec.name);
    for (p, _) in spec.positionals {
        usage.push_str(&format!(" {p}"));
    }
    for o in spec.opts {
        let arg = format!("--{} {}", o.name, o.metavar);
        usage.push_str(&if o.required {
            format!(" {arg}")
        } else {
            format!(" [{arg}]")
        });
    }
    let mut out = vec![usage, String::new(), spec.about.to_string()];
    if !spec.positionals.is_empty() {
        out.push(String::new());
        out.push("positional arguments:".to_string());
        out.extend(
            spec.positionals
                .iter()
                .map(|(p, h)| format!("  {p:<22}{h}")),
        );
    }
    out.push(String::new());
    out.push("options:".to_string());
    out.push(format!(
        "  {:<22}{}",
        "-h, --help", "show this help and exit"
    ));
    out.push(format!("  {:<22}{}", "--json", "print the result as JSON"));
    out.push(format!(
        "  {:<22}{}",
        "--kb KB.TOML|off", "record this run in the KB kb.toml names, or not (required)"
    ));
    for o in spec.opts {
        let tail = match o.kind {
            Kind::Append => " (repeatable)",
            Kind::RowsFile if o.required => " (repeatable, required)",
            Kind::RowsFile => " (repeatable)",
            _ if o.required => " (required)",
            _ => "",
        };
        out.push(format!("  {:<22}{}{tail}", format!("--{}", o.name), o.help));
    }
    out.join("\n")
}

/// Parse `tokens` (everything after the command words) against `spec`.
/// `--json` is accepted anywhere and reported through `json`.
pub fn parse(spec: &Spec, tokens: &[String], json: &mut bool) -> Result<Parsed> {
    let mut a = Args {
        pos: Vec::new(),
        vals: HashMap::new(),
        items: Vec::new(),
    };
    let mut extra: Vec<String> = Vec::new();
    let mut i = 0;
    let mut only_pos = false;
    while i < tokens.len() {
        let t = tokens[i].as_str();
        i += 1;
        if only_pos || !t.starts_with('-') || t == "-" || is_number(t) {
            if let Some((name, _)) = spec.positionals.get(a.pos.len()) {
                a.items.push(Item::Pos(name, t.to_string()));
                a.pos.push(t.to_string());
            } else {
                extra.push(t.to_string());
            }
            continue;
        }
        match t {
            "--" => {
                only_pos = true;
                continue;
            }
            "-h" | "--help" => return Ok(Parsed::Help(help(spec))),
            "--json" => {
                *json = true;
                continue;
            }
            _ => {}
        }
        let (name, inline) = match t
            .strip_prefix("--")
            .map(|r| r.split_once('=').unwrap_or((r, "")))
        {
            Some((n, v)) if t.contains('=') => (n, Some(v.to_string())),
            Some((n, _)) => (n, None),
            None => ("", None),
        };
        let Some(opt) = spec.opts.iter().find(|o| o.name == name) else {
            extra.push(t.to_string());
            continue;
        };
        let want = if opt.kind == Kind::FilePair { 2 } else { 1 };
        let mut values = Vec::new();
        if let Some(v) = inline {
            values.push(v);
        }
        while values.len() < want && i < tokens.len() {
            let v = tokens[i].as_str();
            if v.starts_with('-') && v != "-" && !is_number(v) {
                break;
            }
            values.push(v.to_string());
            i += 1;
        }
        if values.len() < want {
            let n = if want == 1 {
                "one argument".to_string()
            } else {
                format!("{want} arguments")
            };
            fail!("argument --{}: expected {n}", opt.name);
        }
        for v in &values {
            check(opt, v)?;
        }
        a.items.push(Item::Opt(opt.name));
        a.items
            .extend(values.iter().map(|v| Item::Val(opt.kind, v.clone())));
        if matches!(opt.kind, Kind::Append | Kind::RowsFile) {
            a.vals.entry(opt.name).or_default().extend(values);
        } else {
            a.vals.insert(opt.name, values);
        }
    }
    let mut missing: Vec<String> = spec
        .positionals
        .iter()
        .skip(a.pos.len())
        .map(|(p, _)| p.to_string())
        .collect();
    missing.extend(
        spec.opts
            .iter()
            .filter(|o| o.required && !a.vals.contains_key(o.name))
            .map(|o| format!("--{}", o.name)),
    );
    if !missing.is_empty() {
        fail!(
            "the following arguments are required: {}",
            missing.join(", ")
        );
    }
    if !extra.is_empty() {
        fail!("unrecognized arguments: {}", extra.join(" "));
    }
    Ok(Parsed::Args(a))
}

impl Args {
    /// Every argument as given, in order.
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    pub fn pos(&self, i: usize) -> Result<&str> {
        match self.pos.get(i) {
            Some(p) => Ok(p),
            None => fail!("positional argument {} is missing", i + 1),
        }
    }

    fn one(&self, name: &str) -> Option<&str> {
        self.vals
            .get(name)
            .and_then(|v| v.first())
            .map(String::as_str)
    }

    pub fn opt_str(&self, name: &str) -> Option<&str> {
        self.one(name)
    }

    pub fn str(&self, name: &str) -> Result<&str> {
        match self.one(name) {
            Some(v) => Ok(v),
            None => fail!("argument --{name} is required"),
        }
    }

    pub fn opt_int(&self, name: &str) -> Result<Option<i64>> {
        match self.one(name).map(str::parse::<i64>) {
            None => Ok(None),
            Some(Ok(v)) => Ok(Some(v)),
            Some(Err(_)) => fail!("argument --{name}: invalid int value"),
        }
    }

    pub fn int(&self, name: &str) -> Result<i64> {
        match self.opt_int(name)? {
            Some(v) => Ok(v),
            None => fail!("argument --{name} is required"),
        }
    }

    pub fn opt_float(&self, name: &str) -> Result<Option<f64>> {
        match self.one(name).map(str::parse::<f64>) {
            None => Ok(None),
            Some(Ok(v)) => Ok(Some(v)),
            Some(Err(_)) => fail!("argument --{name}: invalid float value"),
        }
    }

    pub fn float(&self, name: &str) -> Result<f64> {
        match self.opt_float(name)? {
            Some(v) => Ok(v),
            None => fail!("argument --{name} is required"),
        }
    }

    /// Every value of a repeatable or two-value option (empty when absent).
    pub fn list(&self, name: &str) -> &[String] {
        self.vals.get(name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Every `ROWS=PATH` value of a repeatable option, as given.
    pub fn rows_files(&self, name: &str) -> Result<Vec<(u64, &str)>> {
        self.list(name).iter().map(|v| rows_file(v)).collect()
    }
}
