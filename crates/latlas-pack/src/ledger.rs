// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: LEDGER.tsv as `tools/ledger.py` reads it: the table, its `check`, and the two
//! closures `tainted` and `deps`, with ledger.py's exact iteration orders (Python dicts keep the
//! first insertion's position and the last value; `list.pop()` takes from the end) and its exact
//! message and output text. The P0 gate proves byte parity on the real ledger and on sabotaged
//! copies; this module is the Rust half of that gate.
//!
//! Owner: metrale-latlas-pack (the metrale pack; was metrale-kg-derive).
//! Invariants:
//! - [`Table::parse`] never drops a line: a malformed row is kept and reported by [`Table::check`].
//! - The closures refuse a table [`Table::check`] rejects (ledger.py exits "ledger invalid").

use crate::prelude::*;

/// 2026-10-23: The column count ledger.py checks (16: `gate` and `resolves` added 2026-10-23).
pub const NCOLS: usize = 16;
/// 2026-10-23: The index of `depends_on`.
pub const DEPENDS_ON: usize = 12;
/// 2026-10-23: The index of `retracts`.
pub const RETRACTS: usize = 13;
/// 2026-10-23: The index of `gate` (the row's claims, `crate::claims` grammar).
pub const GATE: usize = 14;
/// 2026-10-23: The index of `resolves` (earlier BLOCKED rows this row unblocks).
pub const RESOLVES: usize = 15;
/// 2026-10-23: The index of `verdict`.
pub const VERDICT: usize = 7;
/// 2026-10-23: The trailing header columns ledger.py requires.
pub const TAIL: [&str; 4] = ["depends_on", "retracts", "gate", "resolves"];

/// 2026-10-23: ledger.py `blocked_verdict`.
pub fn blocked_verdict(r: &[String]) -> bool {
    r.get(VERDICT).is_some_and(|v| v.starts_with("BLOCKED"))
}

/// 2026-10-23: The ledger as split text: header fields and row fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// 2026-10-23: ledger.py's `refs`: `[]` for `""`/`-`, else the comma split.
pub fn refs(field: &str) -> Vec<String> {
    if field.is_empty() || field == "-" {
        Vec::new()
    } else {
        field.split(',').map(str::to_string).collect()
    }
}

/// 2026-10-23: An insertion-ordered map: Python dict semantics (first position, last value).
#[derive(Default)]
struct PyDict<V> {
    keys: Vec<String>,
    vals: BTreeMap<String, V>,
}

impl<V: Clone> PyDict<V> {
    fn insert(&mut self, k: &str, v: V) {
        if self.vals.insert(k.to_string(), v).is_none() {
            self.keys.push(k.to_string());
        }
    }
    fn get(&self, k: &str) -> Option<&V> {
        self.vals.get(k)
    }
    fn items(&self) -> Vec<(String, V)> {
        self.keys
            .iter()
            .map(|k| (k.clone(), self.vals[k].clone()))
            .collect()
    }
}

impl Table {
    /// 2026-10-23: `text.rstrip("\n").split("\n")`, each line split on tabs.
    pub fn parse(text: &str) -> Table {
        let body = text.trim_end_matches('\n');
        let mut lines = body.split('\n');
        let header = lines
            .next()
            .unwrap_or("")
            .split('\t')
            .map(str::to_string)
            .collect();
        let rows = lines
            .map(|l| l.split('\t').map(str::to_string).collect())
            .collect();
        Table { header, rows }
    }

    /// 2026-10-23: ledger.py `check`: the error lines, in its order (empty = valid).
    pub fn check(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let h = &self.header;
        let tail_ok = h.len() >= TAIL.len() && h[h.len() - TAIL.len()..] == TAIL;
        if h.len() != NCOLS || !tail_ok {
            errs.push(format!(
                "header has {} columns, expected {NCOLS} ending {}",
                h.len(),
                TAIL.join(",")
            ));
        }
        let mut ids: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, r) in self.rows.iter().enumerate() {
            let n = i + 2;
            if r.len() != NCOLS {
                errs.push(format!(
                    "line {n}: {} fields, expected {NCOLS} (missing newline or tab in a field?)",
                    r.len()
                ));
                continue;
            }
            if let Some(first) = ids.get(r[2].as_str()) {
                errs.push(format!(
                    "line {n}: duplicate id {} (first on line {first})",
                    r[2]
                ));
            }
            ids.insert(r[2].as_str(), n);
            if let Err(e) = crate::claims::parse(&r[GATE]) {
                errs.push(format!("line {n}: gate `{}`: {e}", r[GATE]));
            }
        }
        for (i, r) in self.rows.iter().enumerate() {
            let n = i + 2;
            if r.len() != NCOLS {
                continue;
            }
            for col in [DEPENDS_ON, RETRACTS, RESOLVES] {
                let name = h.get(col).map_or("?", String::as_str);
                for x in refs(&r[col]) {
                    if !ids.contains_key(x.as_str()) {
                        errs.push(format!("line {n}: {name} names unknown id {x}"));
                    } else if x == r[2] {
                        errs.push(format!("line {n}: {} references itself", r[2]));
                    }
                }
            }
        }
        let mut resolver: BTreeMap<String, String> = BTreeMap::new();
        for (i, r) in self.rows.iter().enumerate() {
            let n = i + 2;
            if r.len() != NCOLS || refs(&r[RESOLVES]).is_empty() {
                continue;
            }
            if blocked_verdict(r) {
                errs.push(format!(
                    "line {n}: {} resolves rows but is itself BLOCKED",
                    r[2]
                ));
            }
            for x in refs(&r[RESOLVES]) {
                let Some(&at) = ids.get(x.as_str()) else {
                    continue;
                };
                if x == r[2] {
                    continue;
                }
                if at > n {
                    errs.push(format!(
                        "line {n}: resolves {x}, which is a later row (line {at})"
                    ));
                } else if !blocked_verdict(&self.rows[at - 2]) {
                    errs.push(format!(
                        "line {n}: resolves {x}, whose verdict is not BLOCKED"
                    ));
                }
                if let Some(by) = resolver.get(&x) {
                    errs.push(format!("line {n}: resolves {x}, already resolved by {by}"));
                }
                resolver.entry(x).or_insert_with(|| r[2].clone());
            }
        }
        let graph = self.graph(|r| r.len() == NCOLS);
        let mut state: BTreeMap<String, u8> = BTreeMap::new();
        for (v, _) in graph.items() {
            visit(&graph, &v, &mut Vec::new(), &mut state, &mut errs);
        }
        errs
    }

    fn graph(&self, keep: impl Fn(&Vec<String>) -> bool) -> PyDict<Vec<String>> {
        let mut g = PyDict::default();
        for r in self.rows.iter().filter(|r| keep(r)) {
            let deps = r.get(DEPENDS_ON).map(|f| refs(f)).unwrap_or_default();
            g.insert(r.get(2).map_or("", String::as_str), deps);
        }
        g
    }

    /// 2026-10-23: The well-formed rows' links, in file order (the closures' input).
    pub fn links(&self) -> Vec<Links> {
        self.rows
            .iter()
            .filter(|r| r.len() == NCOLS)
            .map(|r| Links {
                id: r[2].clone(),
                depends_on: refs(&r[DEPENDS_ON]),
                retracts: refs(&r[RETRACTS]),
            })
            .collect()
    }
}

/// 2026-10-23: One ledger row's references, in field order: what the closures read. The table
/// supplies them ([`Table::links`]) and so does the bundle (`queries::ledger_links`), so one
/// closure implementation serves ledger.py parity and `la_impact`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Links {
    pub id: String,
    pub depends_on: Vec<String>,
    pub retracts: Vec<String>,
}

/// 2026-10-23: ledger.py `tainted`: every retracted row and why, sorted by id.
pub fn tainted(rows: &[Links]) -> Vec<(String, String)> {
    let mut direct: PyDict<String> = PyDict::default();
    let mut dependents: PyDict<Vec<String>> = PyDict::default();
    for r in rows {
        for x in &r.retracts {
            direct.insert(x, r.id.clone());
        }
    }
    for r in rows {
        for d in &r.depends_on {
            let mut v = dependents.get(d).cloned().unwrap_or_default();
            v.push(r.id.clone());
            dependents.insert(d, v);
        }
    }
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    let mut stack = direct.items();
    while let Some((v, why)) = stack.pop() {
        if out.contains_key(&v) {
            continue;
        }
        out.insert(v.clone(), why);
        for w in dependents.get(&v).cloned().unwrap_or_default() {
            stack.push((w, format!("depends on {v}")));
        }
    }
    out.into_iter().collect()
}

/// 2026-10-23: ledger.py `deps <id>`: the DFS order of the dependency chain.
pub fn deps(rows: &[Links], id: &str) -> Vec<String> {
    let mut graph: PyDict<Vec<String>> = PyDict::default();
    for r in rows {
        graph.insert(&r.id, r.depends_on.clone());
    }
    let mut seen: Vec<String> = Vec::new();
    let mut stack = vec![id.to_string()];
    while let Some(v) = stack.pop() {
        if !seen.contains(&v) {
            stack.extend(graph.get(&v).cloned().unwrap_or_default());
            seen.push(v);
        }
    }
    seen
}

fn visit(
    graph: &PyDict<Vec<String>>,
    v: &str,
    path: &mut Vec<String>,
    state: &mut BTreeMap<String, u8>,
    errs: &mut Vec<String>,
) {
    match state.get(v) {
        Some(1) => {
            let mut p = path.clone();
            p.push(v.to_string());
            errs.push(format!("depends_on cycle: {}", p.join(" -> ")));
            return;
        }
        Some(2) => return,
        _ => {}
    }
    state.insert(v.to_string(), 1);
    path.push(v.to_string());
    for w in graph.get(v).cloned().unwrap_or_default() {
        visit(graph, &w, path, state, errs);
    }
    path.pop();
    state.insert(v.to_string(), 2);
}

/// 2026-10-23: ledger.py `blocked`: `(id, verdict)` of every BLOCKED row no row resolves, in
/// file order.
pub fn blocked(t: &Table) -> Vec<(String, String)> {
    let resolved: BTreeSet<String> = t
        .rows
        .iter()
        .filter(|r| r.len() == NCOLS)
        .flat_map(|r| refs(&r[RESOLVES]))
        .collect();
    t.rows
        .iter()
        .filter(|r| r.len() == NCOLS && blocked_verdict(r) && !resolved.contains(&r[2]))
        .map(|r| (r[2].clone(), r[VERDICT].clone()))
        .collect()
}

/// 2026-10-23: The stdout of `ledger.py blocked`.
pub fn render_blocked(rows: &[(String, String)]) -> String {
    rows.iter().map(|(id, v)| format!("{id}\t{v}\n")).collect()
}

/// 2026-10-23: The stdout of `ledger.py check` (it prints, then exits 1 when non-empty).
pub fn render_check(t: &Table) -> String {
    let errs = t.check();
    if errs.is_empty() {
        format!("ok: {} rows\n", t.rows.len())
    } else {
        format!("{}\n", errs.join("\n"))
    }
}

/// 2026-10-23: The stdout of `ledger.py tainted`.
pub fn render_tainted(rows: &[(String, String)]) -> String {
    let mut s = String::new();
    for (v, why) in rows {
        if why.starts_with("depends") {
            s.push_str(&format!("{v}\t{why}\n"));
        } else {
            s.push_str(&format!("{v}\tretracted by {why}\n"));
        }
    }
    s
}

/// 2026-10-23: The stdout of `ledger.py deps <id>`.
pub fn render_deps(chain: &[String]) -> String {
    format!("{}\n", chain.join(" <- "))
}

/// 2026-10-23: The message ledger.py exits with when `tainted`/`deps` meet an invalid ledger.
pub const INVALID: &str = "ledger invalid, run `ledger.py check`";
