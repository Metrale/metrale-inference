// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The raw power trace: every source's declared capabilities and samples, the
//! measurement windows, the launch checkpoints and the run's metadata (policy, order, seed,
//! fingerprint). A row is recomputable from the trace alone (`derive::derive`), so the trace is
//! what persists beside the row as `energy/<record>.trace.csv`, named by `trace_sha256` (HPAR).
//!
//! Text form, one record per line, `,`-separated, deterministic (floats print in Rust's
//! shortest round-trip form):
//! ```text
//! # metrale-power-trace 1
//! meta,<key>,<value...>
//! source,<id>,<kind>,<version>
//! counter,<id>,<rail>,<unit_j>,<update_period_ms>
//! sensor,<id>,<rail>,<update_period_ms>,<average_window_ms|->
//! field,<id>,<field>
//! unavailable,<id>,<field>,<reason...>
//! window,<idle|load>,<box>,<repeat>,<from_ns>,<to_ns>,<launches|->
//! launches,<at_ns>,<cumulative launches>
//! s,<id>,<at_ns>,<key=value;...>
//! ```
//!
//! Owner: metrale-power.
//! Invariants: `decode(encode(t)) == t`; a sample is pushed only after its source's capabilities
//! accept it.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, Field, PowerCapabilities, PowerSample, Rail, RailCaps, SourceId, SourceKind,
};
use metrale_error::{ErrorCode, error};
use sha2::{Digest, Sha256};

use crate::Result;
use crate::trace_sample::{decode_sample, encode_sample};

/// 2026-10-22: The first line of every trace.
pub const HEADER: &str = "# metrale-power-trace 1";

/// 2026-10-22: What a window measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// 2026-10-22: Context up, nothing running: an idle baseline.
    Idle,
    /// 2026-10-22: The kernel looping.
    Load,
}

impl Phase {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Load => "load",
        }
    }
}

/// 2026-10-22: One measurement window on one box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub phase: Phase,
    /// 2026-10-22: The box (hierarchical level above the run).
    pub box_id: u32,
    /// 2026-10-22: The repeat (run) within the box; idle windows carry the repeat they precede.
    pub repeat: u32,
    pub from: MonoNs,
    pub to: MonoNs,
    /// 2026-10-22: Launches completed inside a load window; `None` for idle, and for a load
    /// window whose recorder did not count them (attribution per launch then refuses it).
    pub launches: Option<u64>,
}

impl Window {
    /// 2026-10-22: Length in seconds.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        secs(self.from, self.to)
    }
}

/// 2026-10-22: Seconds from `a` to `b` (0 when `b` is earlier).
#[must_use]
pub fn secs(a: MonoNs, b: MonoNs) -> f64 {
    b.0.saturating_sub(a.0) as f64 * 1e-9
}

/// 2026-10-22: A cumulative launch count at one instant (the harness records one per sync).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub at: MonoNs,
    pub launches: u64,
}

/// 2026-10-22: One source's declaration and samples.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceTrace {
    pub caps: PowerCapabilities,
    pub samples: Vec<PowerSample>,
}

/// 2026-10-22: A whole recorded run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PowerTrace {
    pub meta: BTreeMap<String, String>,
    pub sources: Vec<SourceTrace>,
    pub windows: Vec<Window>,
    pub checkpoints: Vec<Checkpoint>,
}

impl PowerTrace {
    /// 2026-10-22: Add a source; its index is the return value.
    pub fn add_source(&mut self, caps: PowerCapabilities) -> usize {
        self.sources.push(SourceTrace {
            caps,
            samples: Vec::new(),
        });
        self.sources.len() - 1
    }

    /// 2026-10-22: Record `s` for source `i` once its capabilities accept it.
    pub fn push(&mut self, i: usize, s: PowerSample) -> Result<()> {
        let src = self.sources.get_mut(i).ok_or_else(|| {
            error!(
                ErrorCode::EnergyInputInvalid,
                format!("no source {i} in the trace")
            )
        })?;
        src.caps.check(&s)?;
        src.samples.push(s);
        Ok(())
    }

    /// 2026-10-22: The source with id `id`.
    #[must_use]
    pub fn source(&self, id: &str) -> Option<&SourceTrace> {
        self.sources.iter().find(|s| s.caps.source().0 == id)
    }

    /// 2026-10-22: The text form.
    #[must_use]
    pub fn encode(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "{HEADER}");
        for (k, v) in &self.meta {
            let _ = writeln!(o, "meta,{k},{v}");
        }
        for s in &self.sources {
            encode_caps(&mut o, &s.caps);
        }
        for w in &self.windows {
            let l = w
                .launches
                .map_or_else(|| "-".to_string(), |n| n.to_string());
            let _ = writeln!(
                o,
                "window,{},{},{},{},{},{l}",
                w.phase.as_str(),
                w.box_id,
                w.repeat,
                w.from.0,
                w.to.0
            );
        }
        for c in &self.checkpoints {
            let _ = writeln!(o, "launches,{},{}", c.at.0, c.launches);
        }
        for s in &self.sources {
            for x in &s.samples {
                let _ = writeln!(o, "s,{},{},{}", s.caps.source().0, x.at.0, encode_sample(x));
            }
        }
        o
    }

    /// 2026-10-22: Parse the text form; every line is checked, every sample against its source.
    pub fn decode(text: &str) -> Result<Self> {
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, HEADER)) => {}
            _ => return Err(bad(1, "missing the `# metrale-power-trace 1` header")),
        }
        let mut t = PowerTrace::default();
        let mut decls: Vec<Decl> = Vec::new();
        for (i, line) in lines {
            let n = i + 1;
            if line.is_empty() {
                continue;
            }
            let (tag, rest) = line
                .split_once(',')
                .ok_or_else(|| bad(n, "no record tag"))?;
            match tag {
                "meta" => {
                    let (k, v) = rest
                        .split_once(',')
                        .ok_or_else(|| bad(n, "meta needs a key"))?;
                    t.meta.insert(k.to_string(), v.to_string());
                }
                "source" | "counter" | "sensor" | "field" | "unavailable" => {
                    decl_line(&mut decls, tag, rest, n)?;
                }
                "window" => t.windows.push(window_line(rest, n)?),
                "launches" => {
                    let f: Vec<&str> = rest.split(',').collect();
                    let [at, count] = f.as_slice() else {
                        return Err(bad(n, "launches needs at_ns and a count"));
                    };
                    t.checkpoints.push(Checkpoint {
                        at: MonoNs(num(at, n)?),
                        launches: num(count, n)?,
                    });
                }
                "s" => {
                    if !decls.is_empty() {
                        for d in decls.drain(..) {
                            t.add_source(d.build()?);
                        }
                    }
                    let mut f = rest.splitn(3, ',');
                    let (id, at, kv) = (f.next(), f.next(), f.next());
                    let (Some(id), Some(at)) = (id, at) else {
                        return Err(bad(n, "a sample needs a source and at_ns"));
                    };
                    let i = t
                        .sources
                        .iter()
                        .position(|s| s.caps.source().0 == id)
                        .ok_or_else(|| bad(n, &format!("sample for undeclared source {id}")))?;
                    let s = decode_sample(MonoNs(num(at, n)?), kv.unwrap_or(""), n)?;
                    t.push(i, s)?;
                }
                other => return Err(bad(n, &format!("unknown record `{other}`"))),
            }
        }
        for d in decls {
            t.add_source(d.build()?);
        }
        Ok(t)
    }
}

/// 2026-10-22: Lowercase hex SHA-256 of `bytes` (the row's `trace_sha256`).
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in d.iter() {
        let _ = write!(s, "{b:02x}");
    }
    s
}

pub(crate) fn bad(line: usize, why: &str) -> metrale_error::Error {
    error!(ErrorCode::PowerTraceFormat, line, why.to_string())
}

pub(crate) fn num<T: core::str::FromStr>(s: &str, line: usize) -> Result<T> {
    s.parse()
        .map_err(|_| bad(line, &format!("`{s}` is not a number")))
}

fn encode_caps(o: &mut String, c: &PowerCapabilities) {
    let id = &c.source().0;
    let _ = writeln!(o, "source,{id},{},{}", c.kind().as_str(), c.version());
    for f in c.available() {
        match f {
            Field::Energy(r) => {
                if let Some(s) = c.counter(*r) {
                    let _ = writeln!(
                        o,
                        "counter,{id},{},{:?},{:?}",
                        r.as_str(),
                        s.unit_j,
                        s.update_period_ms
                    );
                }
            }
            Field::InstantPower(r) => {
                if let Some(s) = c.sensor(*r) {
                    let w = s
                        .average_window_ms
                        .map_or_else(|| "-".to_string(), |v| format!("{v:?}"));
                    let _ = writeln!(o, "sensor,{id},{},{:?},{w}", r.as_str(), s.update_period_ms);
                }
            }
            other => {
                let _ = writeln!(o, "field,{id},{}", other.name());
            }
        }
    }
    for u in c.unavailable() {
        let _ = writeln!(o, "unavailable,{id},{},{}", u.field.name(), u.reason);
    }
}

/// 2026-10-22: A source declaration being read.
struct Decl {
    id: String,
    kind: SourceKind,
    version: String,
    /// 2026-10-22: (tag, fields, the line it was read from).
    lines: Vec<(String, Vec<String>, usize)>,
}

impl Decl {
    fn build(self) -> Result<PowerCapabilities> {
        let mut b = PowerCapabilities::builder(SourceId(self.id), self.kind, self.version);
        let mut universe = Vec::new();
        for (tag, f, line) in &self.lines {
            let line = *line;
            let field = |name: &str| Field::parse(name).ok_or_else(|| bad(line, "unknown field"));
            let rail = |name: &str| Rail::parse(name).ok_or_else(|| bad(line, "unknown rail"));
            match (tag.as_str(), f.as_slice()) {
                ("counter", [r, unit, period]) => {
                    let r = rail(r)?;
                    universe.push(Field::Energy(r));
                    b = b.counter(
                        r,
                        CounterSpec {
                            unit_j: num(unit, line)?,
                            update_period_ms: num(period, line)?,
                        },
                    );
                }
                ("sensor", [r, period, window]) => {
                    let r = rail(r)?;
                    universe.push(Field::InstantPower(r));
                    let average_window_ms = match window.as_str() {
                        "-" => None,
                        w => Some(num(w, line)?),
                    };
                    b = b.sensor(
                        r,
                        RailCaps {
                            update_period_ms: num(period, line)?,
                            average_window_ms,
                        },
                    );
                }
                ("field", [name]) => {
                    let f = field(name)?;
                    universe.push(f);
                    b = b.field(f);
                }
                ("unavailable", [name, reason]) => {
                    let f = field(name)?;
                    universe.push(f);
                    b = b.unavailable(f, reason);
                }
                _ => return Err(bad(line, &format!("malformed `{tag}` declaration"))),
            }
        }
        b.build(&universe)
    }
}

fn decl_line(decls: &mut Vec<Decl>, tag: &str, rest: &str, n: usize) -> Result<()> {
    if tag == "source" {
        let f: Vec<&str> = rest.splitn(3, ',').collect();
        let [id, kind, version] = f.as_slice() else {
            return Err(bad(n, "source needs id, kind and version"));
        };
        let kind = SourceKind::parse(kind).ok_or_else(|| bad(n, "unknown source kind"))?;
        decls.push(Decl {
            id: (*id).to_string(),
            kind,
            version: (*version).to_string(),
            lines: Vec::new(),
        });
        return Ok(());
    }
    // 2026-10-22: An `unavailable` reason is free text and may hold commas: it is the remainder
    // after the source and the field. Every other declaration has at most four fields.
    let limit = if tag == "unavailable" { 3 } else { 5 };
    let f: Vec<&str> = rest.splitn(limit, ',').collect();
    let (id, fields) = f
        .split_first()
        .ok_or_else(|| bad(n, "declaration needs a source"))?;
    let d = decls
        .iter_mut()
        .rev()
        .find(|d| d.id == *id)
        .ok_or_else(|| bad(n, &format!("declaration for undeclared source {id}")))?;
    d.lines.push((
        tag.to_string(),
        fields.iter().map(|s| (*s).to_string()).collect(),
        n,
    ));
    Ok(())
}

fn window_line(rest: &str, n: usize) -> Result<Window> {
    let f: Vec<&str> = rest.split(',').collect();
    let [phase, box_id, repeat, from, to, launches] = f.as_slice() else {
        return Err(bad(
            n,
            "window needs phase, box, repeat, from, to, launches",
        ));
    };
    let phase = match *phase {
        "idle" => Phase::Idle,
        "load" => Phase::Load,
        _ => return Err(bad(n, "window phase is idle or load")),
    };
    let launches = match *launches {
        "-" => None,
        l => Some(num(l, n)?),
    };
    if phase == Phase::Idle && launches.is_some() {
        return Err(bad(n, "an idle window has no launches"));
    }
    Ok(Window {
        phase,
        box_id: num(box_id, n)?,
        repeat: num(repeat, n)?,
        from: MonoNs(num(from, n)?),
        to: MonoNs(num(to, n)?),
        launches,
    })
}
