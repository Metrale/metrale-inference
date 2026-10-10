// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `kernels/<hw>/common/SCHEDULES.toml`: the envelope sweep's winners per cell and
//! row range, with the sources each winning family compiles from and their digest at sweep time.
//! The file is generated ([`render`]), never hand-edited; the kernel build reads it back
//! ([`parse`]) and drops a schedule whose family's sources changed since ([`super::sources`]).
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Byte-stable: [`render`] orders schedules by (op, weight, activation, k, n, rows lo) and
//!   sources by family, whatever order they arrive in; `parse(render(s))` renders identically.
//! - Every field is stated; unknown keys and a schema other than [`SCHEMA`] are refused.
//! - Bit-identical by default: `enabled = "default"` only for `same` / `bit_identical`
//!   ([`Enabled::of`]); [`check`] refuses any other pairing.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Deserialize;

/// 2026-10-10: Schema of SCHEDULES.toml.
pub const SCHEMA: u32 = 1;

/// 2026-10-10: How the winner's output relates to the default's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Numerics {
    /// 2026-10-10: The winner is the default.
    Same,
    /// 2026-10-10: Another entry point whose output bytes equal the default's on every class.
    BitIdentical,
    /// 2026-10-10: Faster but numerics-changing.
    Differs,
    /// 2026-10-10: No default exists at this cell (no served plan).
    New,
}

impl Numerics {
    /// 2026-10-10: The file spelling.
    pub fn name(self) -> &'static str {
        match self {
            Numerics::Same => "same",
            Numerics::BitIdentical => "bit_identical",
            Numerics::Differs => "differs",
            Numerics::New => "new",
        }
    }
}

/// 2026-10-10: Whether the bake routes the winner by default or only on opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enabled {
    /// 2026-10-10: Routed by default.
    Default,
    /// 2026-10-10: Routed only when opted in.
    OptIn,
}

impl Enabled {
    /// 2026-10-10: The owner's numerics policy, in one place: only a winner whose bytes equal
    /// today's default's is enabled by default.
    pub fn of(numerics: Numerics) -> Enabled {
        match numerics {
            Numerics::Same | Numerics::BitIdentical => Enabled::Default,
            Numerics::Differs | Numerics::New => Enabled::OptIn,
        }
    }

    /// 2026-10-10: The file spelling.
    pub fn name(self) -> &'static str {
        match self {
            Enabled::Default => "default",
            Enabled::OptIn => "opt_in",
        }
    }
}

/// 2026-10-10: A family's kernel sources and their digest ([`super::sources::digest`]).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// 2026-10-10: Repo-relative paths under `kernels/`, sorted.
    pub files: Vec<String>,
    /// 2026-10-10: Lowercase hex SHA-256.
    pub sha256: String,
}

/// 2026-10-10: The op class and widths of a cell: a schedule without its rows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape {
    /// 2026-10-10: Op base name.
    pub op: String,
    /// 2026-10-10: Weight format.
    pub weight: String,
    /// 2026-10-10: Activation format.
    pub activation: String,
    /// 2026-10-10: Input width.
    pub k: u64,
    /// 2026-10-10: Output width.
    pub n: u64,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} x {} k={} n={}",
            self.op, self.weight, self.activation, self.k, self.n
        )
    }
}

/// 2026-10-10: One decision over an inclusive row range.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schedule {
    /// 2026-10-10: Op base name.
    pub op: String,
    /// 2026-10-10: Weight format.
    pub weight: String,
    /// 2026-10-10: Activation format.
    pub activation: String,
    /// 2026-10-10: Input width.
    pub k: u64,
    /// 2026-10-10: Output width.
    pub n: u64,
    /// 2026-10-10: Inclusive row range `[lo, hi]`.
    pub rows: [u64; 2],
    /// 2026-10-10: The winning entry point.
    pub kernel: String,
    /// 2026-10-10: Its family.
    pub family: String,
    /// 2026-10-10: Today's routed entry point, `""` when no served plan has one.
    pub default: String,
    /// 2026-10-10: Numerics class.
    pub numerics: Numerics,
    /// 2026-10-10: [`Enabled::of`] the numerics.
    pub enabled: Enabled,
    /// 2026-10-10: The winner's median, microseconds.
    pub median_us: f64,
    /// 2026-10-10: The default's median, 0 when there is no default.
    pub default_us: f64,
    /// 2026-10-10: Roofline floor, microseconds.
    pub floor_us: f64,
    /// 2026-10-10: `<host> <at>` of the record the decision rests on.
    pub measured: String,
}

impl Schedule {
    /// 2026-10-10: The schedule's shape.
    pub fn shape(&self) -> Shape {
        Shape {
            op: self.op.clone(),
            weight: self.weight.clone(),
            activation: self.activation.clone(),
            k: self.k,
            n: self.n,
        }
    }

    /// 2026-10-10: The entry point a default build launches here: the winner when it is
    /// enabled by default, otherwise today's default (`""` for a `new` cell).
    pub fn routed(&self) -> &str {
        match self.enabled {
            Enabled::Default => &self.kernel,
            Enabled::OptIn => &self.default,
        }
    }

    fn order(&self) -> (Shape, u64) {
        (self.shape(), self.rows[0])
    }
}

/// 2026-10-10: The whole file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schedules {
    /// 2026-10-10: [`SCHEMA`].
    pub schema: u32,
    /// 2026-10-10: Hardware class.
    pub hardware: String,
    /// 2026-10-10: The command that generated the file.
    pub generated_by: String,
    /// 2026-10-10: Sources per family id.
    pub sources: BTreeMap<String, Source>,
    /// 2026-10-10: The decisions.
    pub schedule: Vec<Schedule>,
}

/// 2026-10-10: A SCHEDULES file that cannot be used, or sources that cannot be digested.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchedulesError {
    /// 2026-10-10: Not TOML, or not the schema's shape.
    #[error("SCHEDULES.toml: {0}")]
    Parse(String),
    /// 2026-10-10: A field breaks a schema rule.
    #[error("SCHEDULES.toml {at}: {problem}")]
    Invalid {
        /// 2026-10-10: Where (`schedule <shape> rows [lo, hi]`, `sources.<family>`).
        at: String,
        /// 2026-10-10: What is wrong.
        problem: String,
    },
    /// 2026-10-10: A repository file could not be read.
    #[error("{0}")]
    Load(String),
}

fn invalid(at: impl Into<String>, problem: impl Into<String>) -> SchedulesError {
    SchedulesError::Invalid {
        at: at.into(),
        problem: problem.into(),
    }
}

/// 2026-10-10: Parse SCHEDULES.toml text and [`check`] it. Schedules keep file order.
pub fn parse(text: &str) -> Result<Schedules, SchedulesError> {
    let s: Schedules = toml::from_str(text).map_err(|e| SchedulesError::Parse(e.to_string()))?;
    check(&s)?;
    Ok(s)
}

fn is_hex_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// 2026-10-10: Every schema rule: the schema number, sources well formed, each schedule's
/// numerics consistent with its default, enabled and times, its family's sources present, and
/// no two row ranges of one shape overlapping.
pub fn check(s: &Schedules) -> Result<(), SchedulesError> {
    if s.schema != SCHEMA {
        return Err(SchedulesError::Parse(format!(
            "schema {} (this build reads {SCHEMA})",
            s.schema
        )));
    }
    if s.hardware.is_empty() || s.generated_by.is_empty() {
        return Err(invalid(
            "header",
            "`hardware` and `generated_by` are required",
        ));
    }
    for (fam, src) in &s.sources {
        let at = format!("sources.{fam}");
        if src.files.is_empty() {
            return Err(invalid(at, "no files"));
        }
        if src.files.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid(at, "files must be sorted and distinct"));
        }
        if let Some(f) = src.files.iter().find(|f| !f.starts_with("kernels/")) {
            return Err(invalid(at, format!("`{f}` is not a kernel source")));
        }
        if !is_hex_digest(&src.sha256) {
            return Err(invalid(at, "sha256 must be 64 lowercase hex digits"));
        }
    }
    for e in &s.schedule {
        check_schedule(s, e)?;
    }
    let mut sorted: Vec<&Schedule> = s.schedule.iter().collect();
    sorted.sort_by_key(|e| e.order());
    for w in sorted.windows(2) {
        if w[0].shape() == w[1].shape() && w[1].rows[0] <= w[0].rows[1] {
            return Err(invalid(
                format!("schedule {}", w[1].shape()),
                format!("rows {:?} overlap rows {:?}", w[1].rows, w[0].rows),
            ));
        }
    }
    Ok(())
}

fn check_schedule(s: &Schedules, e: &Schedule) -> Result<(), SchedulesError> {
    let at = format!("schedule {} rows {:?}", e.shape(), e.rows);
    let bad = |p: &str| Err(invalid(at.clone(), p));
    if e.rows[0] == 0 || e.rows[0] > e.rows[1] {
        return bad("rows must be [lo, hi] with 1 <= lo <= hi");
    }
    if e.k == 0 || e.n == 0 || e.op.is_empty() || e.weight.is_empty() || e.activation.is_empty() {
        return bad("op, weight, activation, k and n are required");
    }
    if e.kernel.is_empty() || e.measured.is_empty() {
        return bad("kernel and measured are required");
    }
    if !s.sources.contains_key(&e.family) {
        return bad("its family has no `[sources]` entry");
    }
    if e.enabled != Enabled::of(e.numerics) {
        return bad("enabled must be `default` exactly for same / bit_identical");
    }
    let finite = |x: f64| x.is_finite() && x >= 0.0;
    if !(finite(e.median_us) && e.median_us > 0.0 && finite(e.floor_us) && finite(e.default_us)) {
        return bad("times must be finite, non-negative, and the median positive");
    }
    match e.numerics {
        Numerics::New if !e.default.is_empty() || e.default_us != 0.0 => {
            bad("a `new` cell has default = \"\" and default_us = 0")
        }
        Numerics::Same | Numerics::BitIdentical | Numerics::Differs
            if e.default.is_empty() || e.default_us <= 0.0 =>
        {
            bad("a cell with numerics other than `new` states its default and default_us")
        }
        Numerics::Same if e.kernel != e.default => bad("`same` needs kernel = default"),
        Numerics::BitIdentical | Numerics::Differs if e.kernel == e.default => {
            bad("the winner is the default: numerics is `same`")
        }
        _ => Ok(()),
    }
}

/// 2026-10-10: A TOML basic string.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                let _ = write!(out, "\\u{:04X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn key(s: &str) -> String {
    let bare = !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if bare { s.to_string() } else { quote(s) }
}

/// 2026-10-10: Rust's shortest round-trip spelling, which always carries a `.` or an
/// exponent, so TOML reads it back as the same float.
fn float(x: f64) -> String {
    format!("{x:?}")
}

/// 2026-10-10: The file text: header, sources by family, schedules in [`Schedule`] order.
pub fn render(s: &Schedules) -> String {
    let mut o = String::new();
    o.push_str("# SPDX-License-Identifier: MIT OR Apache-2.0\n\n");
    o.push_str("# Generated by the kernel envelope sweep; never hand-edited.\n");
    o.push_str("# Schema: crates/accuracy/src/envelope/schedules.rs.\n\n");
    let _ = writeln!(o, "schema = {}", s.schema);
    let _ = writeln!(o, "hardware = {}", quote(&s.hardware));
    let _ = writeln!(o, "generated_by = {}", quote(&s.generated_by));
    o.push_str(
        "\n# The sources each family's entry points compile from, and their SHA-256 at sweep time\n\
         # (sha256 over the files' bytes in listed order, each file's repo-relative path then its bytes).\n\
         # The bake drops every schedule whose family's sources changed since (stale), with a warning.\n",
    );
    for (fam, src) in &s.sources {
        let files: Vec<String> = src.files.iter().map(|f| quote(f)).collect();
        let _ = writeln!(o, "[sources.{}]", key(fam));
        let _ = writeln!(o, "files = [{}]", files.join(", "));
        let _ = writeln!(o, "sha256 = {}", quote(&src.sha256));
    }
    let mut sorted: Vec<&Schedule> = s.schedule.iter().collect();
    sorted.sort_by_key(|e| e.order());
    for e in sorted {
        o.push_str("\n[[schedule]]\n");
        let _ = writeln!(o, "op = {}", quote(&e.op));
        let _ = writeln!(o, "weight = {}", quote(&e.weight));
        let _ = writeln!(o, "activation = {}", quote(&e.activation));
        let _ = writeln!(o, "k = {}", e.k);
        let _ = writeln!(o, "n = {}", e.n);
        let _ = writeln!(o, "rows = [{}, {}]", e.rows[0], e.rows[1]);
        let _ = writeln!(o, "kernel = {}", quote(&e.kernel));
        let _ = writeln!(o, "family = {}", quote(&e.family));
        let _ = writeln!(o, "default = {}", quote(&e.default));
        let _ = writeln!(o, "numerics = {}", quote(e.numerics.name()));
        let _ = writeln!(o, "enabled = {}", quote(e.enabled.name()));
        let _ = writeln!(o, "median_us = {}", float(e.median_us));
        let _ = writeln!(o, "default_us = {}", float(e.default_us));
        let _ = writeln!(o, "floor_us = {}", float(e.floor_us));
        let _ = writeln!(o, "measured = {}", quote(&e.measured));
    }
    o
}

#[cfg(test)]
#[path = "schedules_tests.rs"]
mod tests;
