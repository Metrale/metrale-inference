// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `kernels/<hw>/common/SCHEDULES.toml` (schema 1), parsed, verified against the
//! current kernel sources and baked into `metrale_kernels::TARGET_SCHEDULES` /
//! `TARGET_SCHEDULES_STALE`.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - An unknown key, a missing key, a wrong type, `schema != 1`, a `hardware` other than the
//!   directory's, a schedule naming a family with no `[sources]` entry, an `enabled = "default"`
//!   entry whose numerics change bits, or two entries of one shape with overlapping rows is an
//!   `Err` naming the file; the build turns it into a panic.
//! - A family whose sources' digest no longer matches loses every schedule (stale) and is named in
//!   [`Baked::stale`]; nothing else is dropped.
//! - Only [`read_tree`] touches the filesystem; [`parse`], [`source_digest`], [`bake`] and
//!   [`literal`] are pure.
//!
//! Included via `#[path = "build_schedules.rs"] mod build_schedules;`. Its own file, with no
//! `super::` dependencies, so `tests/schedules_bake.rs` compiles the same code: cargo never runs a
//! build script's `#[cfg(test)]` modules. Self-contained on purpose: a build script cannot depend
//! on metrale-accuracy, which writes the file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

/// 2026-10-10: The one schema this bake reads.
pub(crate) const SCHEMA: i64 = 1;

/// 2026-10-10: `[sources.<family>]`: the files a family's entry points compile from, and their
/// digest at sweep time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Sources {
    pub family: String,
    pub files: Vec<String>,
    pub sha256: String,
}

/// 2026-10-10: One `[[schedule]]` entry, owned (build-time shape of `schedules::Schedule`).
/// `numerics` and `enabled` hold the TOML spelling, validated by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    pub op: String,
    pub weight: String,
    pub activation: String,
    pub k: u32,
    pub n: u32,
    pub rows_lo: u32,
    pub rows_hi: u32,
    pub kernel: String,
    pub family: String,
    pub default: String,
    pub numerics: String,
    pub enabled: String,
}

/// 2026-10-10: A parsed SCHEDULES.toml.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Parsed {
    pub sources: Vec<Sources>,
    pub schedules: Vec<Entry>,
}

/// 2026-10-10: What the bake keeps: the schedules of every family whose sources are unchanged,
/// and the names of the families dropped as stale (sorted).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Baked {
    pub schedules: Vec<Entry>,
    pub stale: Vec<String>,
}

const TOP_KEYS: &[&str] = &["schema", "hardware", "generated_by", "sources", "schedule"];
const SOURCE_KEYS: &[&str] = &["files", "sha256"];
const ENTRY_KEYS: &[&str] = &[
    "op",
    "weight",
    "activation",
    "k",
    "n",
    "rows",
    "kernel",
    "family",
    "default",
    "numerics",
    "enabled",
    "median_us",
    "default_us",
    "floor_us",
    "measured",
];
const NUMERICS: &[&str] = &["same", "bit_identical", "differs", "new"];
const ENABLED: &[&str] = &["default", "opt_in"];
/// 2026-10-10: The numerics an `enabled = "default"` entry may carry (bit-identical by default).
const DEFAULT_SAFE: &[&str] = &["same", "bit_identical"];

type Table = toml::map::Map<String, toml::Value>;

fn only_keys(file: &str, at: &str, t: &Table, allowed: &[&str]) -> Result<(), String> {
    match t.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(format!("{file}: {at} has no key `{k}` (schema {SCHEMA})")),
        None => Ok(()),
    }
}

fn get<'a>(file: &str, at: &str, t: &'a Table, key: &str) -> Result<&'a toml::Value, String> {
    t.get(key)
        .ok_or_else(|| format!("{file}: {at} is missing `{key}`"))
}

fn string(file: &str, at: &str, t: &Table, key: &str) -> Result<String, String> {
    get(file, at, t, key)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{file}: {at} `{key}` must be a string"))
}

fn uint(file: &str, at: &str, v: &toml::Value, key: &str) -> Result<u32, String> {
    let n = v
        .as_integer()
        .ok_or_else(|| format!("{file}: {at} `{key}` must be an integer"))?;
    u32::try_from(n).map_err(|_| format!("{file}: {at} `{key}` = {n} is not a u32"))
}

fn one_of(file: &str, at: &str, key: &str, v: String, allowed: &[&str]) -> Result<String, String> {
    if allowed.contains(&v.as_str()) {
        Ok(v)
    } else {
        Err(format!(
            "{file}: {at} `{key}` = \"{v}\" is not one of {allowed:?}"
        ))
    }
}

/// 2026-10-10: A listed source must stay inside the repository (CWE-22): relative, no `..`.
fn repo_relative(file: &str, at: &str, p: &str) -> Result<String, String> {
    let ok = !p.is_empty()
        && Path::new(p)
            .components()
            .all(|c| matches!(c, Component::Normal(_)));
    if ok {
        Ok(p.to_string())
    } else {
        Err(format!(
            "{file}: {at} lists `{p}`, which is not a repo-relative path"
        ))
    }
}

fn parse_sources(file: &str, family: &str, v: &toml::Value) -> Result<Sources, String> {
    let at = format!("[sources.{family}]");
    let t = v
        .as_table()
        .ok_or_else(|| format!("{file}: {at} must be a table"))?;
    only_keys(file, &at, t, SOURCE_KEYS)?;
    let list = get(file, &at, t, "files")?
        .as_array()
        .ok_or_else(|| format!("{file}: {at} `files` must be an array"))?;
    let files = list
        .iter()
        .map(|f| {
            let s = f
                .as_str()
                .ok_or_else(|| format!("{file}: {at} `files` must hold strings"))?;
            repo_relative(file, &at, s)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if files.is_empty() {
        return Err(format!("{file}: {at} `files` is empty"));
    }
    let sha256 = string(file, &at, t, "sha256")?;
    Ok(Sources {
        family: family.to_string(),
        files,
        sha256,
    })
}

fn parse_entry(file: &str, i: usize, v: &toml::Value) -> Result<Entry, String> {
    let at = format!("[[schedule]] #{i}");
    let t = v
        .as_table()
        .ok_or_else(|| format!("{file}: {at} must be a table"))?;
    only_keys(file, &at, t, ENTRY_KEYS)?;
    let rows = get(file, &at, t, "rows")?
        .as_array()
        .filter(|r| r.len() == 2)
        .ok_or_else(|| format!("{file}: {at} `rows` must be [lo, hi]"))?;
    let (rows_lo, rows_hi) = (
        uint(file, &at, &rows[0], "rows")?,
        uint(file, &at, &rows[1], "rows")?,
    );
    if rows_lo > rows_hi {
        return Err(format!(
            "{file}: {at} `rows` = [{rows_lo}, {rows_hi}] is empty"
        ));
    }
    for key in ["median_us", "default_us", "floor_us"] {
        let x = get(file, &at, t, key)?;
        if !(x.is_float() || x.is_integer()) {
            return Err(format!("{file}: {at} `{key}` must be a number"));
        }
    }
    string(file, &at, t, "measured")?;
    let numerics = one_of(
        file,
        &at,
        "numerics",
        string(file, &at, t, "numerics")?,
        NUMERICS,
    )?;
    let enabled = one_of(
        file,
        &at,
        "enabled",
        string(file, &at, t, "enabled")?,
        ENABLED,
    )?;
    if enabled == "default" && !DEFAULT_SAFE.contains(&numerics.as_str()) {
        return Err(format!(
            "{file}: {at} is enabled by default with numerics \"{numerics}\"; only \
             {DEFAULT_SAFE:?} may be, a bits-changing winner is opt_in"
        ));
    }
    Ok(Entry {
        op: string(file, &at, t, "op")?,
        weight: string(file, &at, t, "weight")?,
        activation: string(file, &at, t, "activation")?,
        k: uint(file, &at, get(file, &at, t, "k")?, "k")?,
        n: uint(file, &at, get(file, &at, t, "n")?, "n")?,
        rows_lo,
        rows_hi,
        kernel: string(file, &at, t, "kernel")?,
        family: string(file, &at, t, "family")?,
        default: string(file, &at, t, "default")?,
        numerics,
        enabled,
    })
}

/// 2026-10-10: Two entries of one (op, weight, activation, k, n) whose row ranges intersect would
/// make `schedules::lookup` order-dependent.
fn check_overlaps(file: &str, entries: &[Entry]) -> Result<(), String> {
    for (i, a) in entries.iter().enumerate() {
        for b in &entries[i + 1..] {
            let same = (&a.op, &a.weight, &a.activation, a.k, a.n)
                == (&b.op, &b.weight, &b.activation, b.k, b.n);
            if same && a.rows_lo <= b.rows_hi && b.rows_lo <= a.rows_hi {
                return Err(format!(
                    "{file}: two schedules of {} {} {} k={} n={} overlap at rows \
                     [{}, {}] and [{}, {}]",
                    a.op,
                    a.weight,
                    a.activation,
                    a.k,
                    a.n,
                    a.rows_lo,
                    a.rows_hi,
                    b.rows_lo,
                    b.rows_hi
                ));
            }
        }
    }
    Ok(())
}

/// 2026-10-10: Parse a SCHEDULES.toml's text. `file` names it in every error; `hw` is the
/// `kernels/<hw>` directory it was read from and must equal its `hardware`.
pub(crate) fn parse(file: &str, hw: &str, text: &str) -> Result<Parsed, String> {
    let top: Table = toml::from_str(text).map_err(|e| format!("{file}: not TOML: {e}"))?;
    only_keys(file, "the top level", &top, TOP_KEYS)?;
    let schema = get(file, "the top level", &top, "schema")?
        .as_integer()
        .ok_or_else(|| format!("{file}: `schema` must be an integer"))?;
    if schema != SCHEMA {
        return Err(format!(
            "{file}: schema {schema} is not the schema this build reads ({SCHEMA})"
        ));
    }
    let hardware = string(file, "the top level", &top, "hardware")?;
    if hardware != hw {
        return Err(format!(
            "{file}: hardware \"{hardware}\" is not this directory's \"{hw}\""
        ));
    }
    string(file, "the top level", &top, "generated_by")?;
    let sources = match top.get("sources") {
        None => Vec::new(),
        Some(v) => v
            .as_table()
            .ok_or_else(|| format!("{file}: `sources` must be a table"))?
            .iter()
            .map(|(family, v)| parse_sources(file, family, v))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let schedules = match top.get("schedule") {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .ok_or_else(|| format!("{file}: `schedule` must be an array of tables"))?
            .iter()
            .enumerate()
            .map(|(i, v)| parse_entry(file, i, v))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if let Some(e) = schedules
        .iter()
        .find(|e| !sources.iter().any(|s| s.family == e.family))
    {
        return Err(format!(
            "{file}: schedule {} names family `{}`, which has no [sources.{}]",
            e.kernel, e.family, e.family
        ));
    }
    check_overlaps(file, &schedules)?;
    Ok(Parsed { sources, schedules })
}

/// 2026-10-10: SHA-256 over each file's repo-relative path bytes then its content bytes, in
/// listed order, as lower-case hex. The digest the generator records in `[sources]`.
pub(crate) fn source_digest<'a>(files: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> String {
    let mut h = Sha256::new();
    for (path, bytes) in files {
        h.update(path.as_bytes());
        h.update(bytes);
    }
    metrale_closure::hex_lower(&h.finalize())
}

/// 2026-10-10: Keep the schedules of every family whose listed files (`contents`, keyed by
/// repo-relative path; an absent key is a deleted file) still hash to the recorded digest.
pub(crate) fn bake(parsed: &Parsed, contents: &BTreeMap<String, Vec<u8>>) -> Baked {
    let mut stale = BTreeSet::new();
    for s in &parsed.sources {
        let current: Option<Vec<(&str, &[u8])>> = s
            .files
            .iter()
            .map(|f| contents.get(f).map(|b| (f.as_str(), b.as_slice())))
            .collect();
        let fresh = current.is_some_and(|c| source_digest(c) == s.sha256.to_ascii_lowercase());
        if !fresh {
            stale.insert(s.family.clone());
        }
    }
    Baked {
        schedules: parsed
            .schedules
            .iter()
            .filter(|e| !stale.contains(&e.family))
            .cloned()
            .collect(),
        stale: stale.into_iter().collect(),
    }
}

fn variant(spelling: &str) -> &'static str {
    match spelling {
        "same" => "Numerics::Same",
        "bit_identical" => "Numerics::BitIdentical",
        "differs" => "Numerics::Differs",
        "new" => "Numerics::New",
        "default" => "Enabled::Default",
        "opt_in" => "Enabled::OptIn",
        other => unreachable!("`{other}` was refused by parse"),
    }
}

/// 2026-10-10: The generated `TARGET_SCHEDULES` / `TARGET_SCHEDULES_STALE` items. String fields
/// are written with `{:?}`, which escapes them into valid Rust literals.
pub(crate) fn literal(baked: &Baked) -> String {
    let mut out = String::from(
        "// Auto-generated by build.rs from kernels/<hw>/common/SCHEDULES.toml — do not edit.\n\
         /// The swept kernel schedules whose family sources are unchanged since the sweep\n\
         /// (`metrale_kernels::schedules`). Empty when the target has no SCHEDULES.toml.\n\
         pub const TARGET_SCHEDULES: &[Schedule] = &[\n",
    );
    for e in &baked.schedules {
        let default = if e.default.is_empty() {
            "None".to_string()
        } else {
            format!("Some({:?})", e.default)
        };
        out.push_str(&format!(
            "    Schedule {{ op: {:?}, weight: {:?}, activation: {:?}, k: {}, n: {}, \
             rows_lo: {}, rows_hi: {}, kernel: {:?}, family: {:?}, default: {default}, \
             numerics: {}, enabled: {} }},\n",
            e.op,
            e.weight,
            e.activation,
            e.k,
            e.n,
            e.rows_lo,
            e.rows_hi,
            e.kernel,
            e.family,
            variant(&e.numerics),
            variant(&e.enabled),
        ));
    }
    out.push_str(
        "];\n\
         /// Families whose schedules the bake dropped because their sources changed since the\n\
         /// sweep. Each was also reported as a `cargo:warning`.\n\
         pub const TARGET_SCHEDULES_STALE: &[&str] = &[",
    );
    let names: Vec<String> = baked.stale.iter().map(|s| format!("{s:?}")).collect();
    out.push_str(&names.join(", "));
    out.push_str("];\n");
    out
}

/// 2026-10-10: The repo-relative path of a hardware tree's SCHEDULES.toml.
pub(crate) fn schedules_path(hw: &str) -> String {
    format!("kernels/{hw}/common/SCHEDULES.toml")
}

/// 2026-10-10: Read and bake `kernels/<hw>/common/SCHEDULES.toml` under `workspace_root`. The I/O
/// shell around [`parse`] and [`bake`]. An absent file is an empty [`Baked`]. Also returns the
/// paths the result depends on, for `cargo:rerun-if-changed`: the file (or, while it is absent,
/// its directory, so creating it reruns the build) and every listed source that exists (or its
/// directory).
pub(crate) fn read_tree(workspace_root: &Path, hw: &str) -> Result<(Baked, Vec<PathBuf>), String> {
    let rel = schedules_path(hw);
    let path = workspace_root.join(&rel);
    let watch = |p: PathBuf| -> Option<PathBuf> {
        if p.exists() {
            Some(p)
        } else {
            p.parent().filter(|d| d.is_dir()).map(Path::to_path_buf)
        }
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok((Baked::default(), watch(path).into_iter().collect()));
    };
    let parsed = parse(&rel, hw, &text)?;
    let mut deps = vec![path];
    let mut contents = BTreeMap::new();
    for f in parsed.sources.iter().flat_map(|s| &s.files) {
        let p = workspace_root.join(f);
        if let Ok(bytes) = std::fs::read(&p) {
            contents.insert(f.clone(), bytes);
        }
        deps.extend(watch(p));
    }
    deps.sort();
    deps.dedup();
    Ok((bake(&parsed, &contents), deps))
}
