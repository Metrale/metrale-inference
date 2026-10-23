// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg-bundle/v1` in memory: nodes, edges, texts, the source digests and the derive
//! errors, built through [`Builder`] so that every invariant holds by construction
//! (DESIGN-kernel-knowledge-graph.md §4.1; queries part §4).
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Every node has a `src` (`file:line`) and a `sha` of its canonical record text.
//! - Every edge endpoint is a node: a reference to an undefined id of a label that sources must
//!   define becomes an `Unresolved` node plus a `KgDanglingRef` error, never a dropped edge.
//! - No string in the bundle matches the secrets denylist: the builder drops the field and
//!   records `KgSecretRedacted` with the pattern name only.
//! - The result does not depend on the order adapters ran in: records are sorted, duplicates
//!   resolve to the smallest `src`, implicit nodes take the smallest mention site.

use crate::hash::{Sha256, point_id, record_sha};
use crate::id::{KgId, LABEL_LIMIT, Label, Rel};
use crate::prelude::*;
use crate::secrets::secret_pattern;
use metrale_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: The bundle schema id.
pub const SCHEMA: &str = "kg-bundle/v1";

/// 2026-10-23: One node.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: KgId,
    /// 2026-10-23: `path:line` of the source record (root-relative path).
    pub src: String,
    pub sha: Sha256,
    /// 2026-10-23: The payload; keys sorted.
    pub p: BTreeMap<String, Value>,
}

/// 2026-10-23: One edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub from: KgId,
    pub rel: Rel,
    pub to: KgId,
    pub w: f32,
}

/// 2026-10-23: One embeddable text of a node field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Text {
    pub node: KgId,
    pub field: String,
    pub sha: Sha256,
    /// 2026-10-23: `"<Label> <key> <field>: <text>"`.
    pub text: String,
}

/// 2026-10-23: A source file's digest.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceDigest {
    pub path: String,
    pub sha: Sha256,
    /// 2026-10-23: The commit the file is at, when the host knows it.
    pub git: Option<String>,
}

/// 2026-10-23: A derive finding: a registry code, where, and the rendered message.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Finding {
    pub code: &'static str,
    pub at: String,
    pub msg: String,
}

/// 2026-10-23: A derived bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct Bundle {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub texts: Vec<Text>,
    pub sources: Vec<SourceDigest>,
    pub errors: Vec<Finding>,
}

impl Bundle {
    /// 2026-10-23: The node with `id`, by binary search over the sorted nodes.
    pub fn node(&self, id: &KgId) -> Option<&Node> {
        self.nodes
            .binary_search_by(|n| n.id.cmp(id))
            .ok()
            .map(|i| &self.nodes[i])
    }

    /// 2026-10-23: The edges leaving `id` with relation `rel`.
    pub fn out<'a>(&'a self, id: &'a KgId, rel: Rel) -> impl Iterator<Item = &'a Edge> + 'a {
        let start = self.edges.partition_point(|e| e.from < *id);
        self.edges[start..]
            .iter()
            .take_while(move |e| e.from == *id)
            .filter(move |e| e.rel == rel)
    }

    /// 2026-10-23: The edges entering `id` with relation `rel` (a scan).
    pub fn into<'a>(&'a self, id: &'a KgId, rel: Rel) -> impl Iterator<Item = &'a Edge> + 'a {
        self.edges
            .iter()
            .filter(move |e| e.rel == rel && e.to == *id)
    }

    /// 2026-10-23: The nodes of one label.
    pub fn of(&self, label: Label) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(move |n| n.id.label() == label)
    }

    /// 2026-10-23: A string payload field.
    pub fn field<'a>(&'a self, id: &KgId, key: &str) -> Option<&'a str> {
        self.node(id)?.p.get(key)?.as_str()
    }
}

/// 2026-10-23: Labels whose nodes exist because something mentions them (a commit sha, an env
/// lever, a gate name), rather than because a source file defines them.
fn implicit(label: Label) -> bool {
    matches!(
        label,
        Label::Commit
            | Label::Lever
            | Label::Gate
            | Label::Profile
            | Label::Kernel
            | Label::Paper
            | Label::Tool
            | Label::Box
            | Label::Design
            | Label::Op
            | Label::Hardware
    )
}

/// 2026-10-23: Collects records from the adapters and seals them into a [`Bundle`].
#[derive(Default)]
pub struct Builder {
    defined: BTreeMap<KgId, Node>,
    mentioned: BTreeMap<KgId, String>,
    edges: BTreeMap<(KgId, Rel, KgId), (f32, String)>,
    texts: BTreeMap<(KgId, String), String>,
    errors: BTreeSet<Finding>,
}

/// 2026-10-23: Whether an undefined target of `label` is a forward reference rather than an
/// integrity error: memory and feedback-rule links name memories worth writing later (the memory
/// rules: "a [[name]] that doesn't match is fine"), so they never count against integrity.
pub fn is_forward_link(label: Label) -> bool {
    matches!(label, Label::Memory | Label::Rule)
}

impl Builder {
    /// 2026-10-23: Define a node from its source record. A second definition of the same id
    /// keeps the smaller `src` and records `KgDuplicateId`.
    pub fn define(&mut self, id: KgId, src: String, canonical: &str, p: BTreeMap<String, Value>) {
        let node = Node {
            sha: record_sha(canonical),
            id: id.clone(),
            src,
            p,
        };
        if let Some(prev_src) = self.defined.get(&id).map(|p| p.src.clone()) {
            let keep_new = node.src < prev_src;
            let (a, b) = if keep_new {
                (node.src.clone(), prev_src)
            } else {
                (prev_src, node.src.clone())
            };
            self.error(
                &error!(ErrorCode::KgDuplicateId, id.to_string(), a.clone(), b),
                &a,
            );
            if keep_new {
                self.defined.insert(id, node);
            }
            return;
        }
        self.defined.insert(id, node);
    }

    /// 2026-10-23: The nodes of `label` defined so far, in id order.
    pub fn defined_of(&self, label: Label) -> impl Iterator<Item = &Node> {
        self.defined.values().filter(move |n| n.id.label() == label)
    }

    /// 2026-10-23: Whether `id` is defined so far.
    pub fn is_defined(&self, id: &KgId) -> bool {
        self.defined.contains_key(id)
    }

    /// 2026-10-23: Note that `id` is mentioned at `at` (an implicit node, or a check later).
    pub fn mention(&mut self, id: &KgId, at: &str) {
        match self.mentioned.get_mut(id) {
            Some(prev) if prev.as_str() <= at => {}
            Some(prev) => *prev = at.to_string(),
            None => {
                self.mentioned.insert(id.clone(), at.to_string());
            }
        }
    }

    /// 2026-10-23: An edge `from -rel-> to`, recorded at `at`; `to` is checked at seal time.
    pub fn edge(&mut self, from: &KgId, rel: Rel, to: &KgId, w: f32, at: &str) {
        self.mention(to, at);
        let key = (from.clone(), rel, to.clone());
        let keep = self
            .edges
            .get(&key)
            .is_none_or(|(_, prev)| at < prev.as_str());
        if keep {
            self.edges.insert(key, (w, at.to_string()));
        }
    }

    /// 2026-10-23: An embeddable text of `node.field`; empty and `-` texts are skipped.
    pub fn text(&mut self, node: &KgId, field: &str, body: &str) {
        let body = body.trim();
        if body.is_empty() || body == "-" {
            return;
        }
        let text = format!("{} {} {field}: {body}", node.label().name(), node.key());
        self.texts.insert((node.clone(), field.to_string()), text);
    }

    /// 2026-10-23: Record a finding at `at`.
    pub fn error(&mut self, e: &metrale_error::Error, at: &str) {
        let msg = e.to_string();
        let msg = match secret_pattern(&msg) {
            Some(p) => format!("[message dropped: {p}]"),
            None => msg,
        };
        self.errors.insert(Finding {
            code: e.code().name(),
            at: at.to_string(),
            msg,
        });
    }

    /// 2026-10-23: Seal: resolve mentions, drop secrets, check labels and point ids, sort.
    pub fn seal(mut self, mut sources: Vec<SourceDigest>) -> Bundle {
        self.resolve_mentions();
        self.redact();
        let mut nodes: Vec<Node> = core::mem::take(&mut self.defined).into_values().collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        self.check_points(&nodes);
        let used: BTreeSet<Label> = nodes.iter().map(|n| n.id.label()).collect();
        if used.len() > LABEL_LIMIT {
            let e = error!(ErrorCode::KgTooManyLabels, used.len(), LABEL_LIMIT);
            self.error(&e, "ontology");
        }
        let mut edges: Vec<Edge> = core::mem::take(&mut self.edges)
            .into_iter()
            .map(|((from, rel, to), (w, _))| Edge { from, rel, to, w })
            .collect();
        edges.sort_by(|a, b| (&a.from, a.rel.name(), &a.to).cmp(&(&b.from, b.rel.name(), &b.to)));
        let texts = core::mem::take(&mut self.texts)
            .into_iter()
            .map(|((node, field), text)| Text {
                sha: record_sha(&text),
                node,
                field,
                text,
            })
            .collect();
        sources.sort();
        Bundle {
            nodes,
            edges,
            texts,
            sources,
            errors: self.errors.into_iter().collect(),
        }
    }

    fn resolve_mentions(&mut self) {
        let mentioned = core::mem::take(&mut self.mentioned);
        let mut redirect: BTreeMap<KgId, KgId> = BTreeMap::new();
        for (id, at) in mentioned {
            if self.defined.contains_key(&id) {
                continue;
            }
            if implicit(id.label()) {
                let mut p = BTreeMap::new();
                p.insert("implicit".to_string(), Value::Bool(true));
                let canon = id.to_string();
                self.define(id, at, &canon, p);
                continue;
            }
            let u = KgId::unresolved(&id);
            let forward = is_forward_link(id.label());
            let e = if forward {
                error!(ErrorCode::KgForwardLink, at.clone(), id.to_string())
            } else {
                error!(ErrorCode::KgDanglingRef, at.clone(), id.to_string())
            };
            self.error(&e, &at);
            let mut p = BTreeMap::new();
            p.insert("target".to_string(), Value::String(id.to_string()));
            p.insert("forward".to_string(), Value::Bool(forward));
            let canon = u.to_string();
            if !self.defined.contains_key(&u) {
                self.define(u.clone(), at, &canon, p);
            }
            redirect.insert(id, u);
        }
        if redirect.is_empty() {
            return;
        }
        let edges = core::mem::take(&mut self.edges);
        for ((from, rel, to), v) in edges {
            let to = redirect.get(&to).cloned().unwrap_or(to);
            self.edges.insert((from, rel, to), v);
        }
    }

    fn redact(&mut self) {
        let mut drop_ids = BTreeSet::new();
        let mut found = Vec::new();
        for (id, node) in &mut self.defined {
            if let Some(pat) = secret_pattern(&id.to_string()) {
                drop_ids.insert(id.clone());
                found.push((node.src.clone(), pat));
                continue;
            }
            let bad: Vec<(String, &'static str)> = node
                .p
                .iter()
                .filter_map(|(k, v)| value_secret(v).map(|pat| (k.clone(), pat)))
                .collect();
            for (k, pat) in bad {
                node.p.remove(&k);
                found.push((format!("{} field {k}", node.src), pat));
            }
        }
        for id in &drop_ids {
            self.defined.remove(id);
        }
        self.edges
            .retain(|(f, _, t), _| !drop_ids.contains(f) && !drop_ids.contains(t));
        let texts = core::mem::take(&mut self.texts);
        for ((node, field), text) in texts {
            match secret_pattern(&text) {
                Some(pat) => found.push((format!("{node} text {field}"), pat)),
                None if !drop_ids.contains(&node) => {
                    self.texts.insert((node, field), text);
                }
                None => {}
            }
        }
        for (at, pat) in found {
            let e = error!(ErrorCode::KgSecretRedacted, at.clone(), pat);
            self.error(&e, &at);
        }
    }

    fn check_points(&mut self, nodes: &[Node]) {
        let mut seen: BTreeMap<u64, &KgId> = BTreeMap::new();
        let mut hits = Vec::new();
        for n in nodes {
            let p = point_id(&n.id);
            if let Some(prev) = seen.insert(p, &n.id) {
                hits.push(error!(
                    ErrorCode::KgIdCollision,
                    prev.to_string(),
                    n.id.to_string(),
                    p
                ));
            }
        }
        for e in hits {
            self.error(&e, "point ids");
        }
    }
}

fn value_secret(v: &Value) -> Option<&'static str> {
    match v {
        Value::String(s) => secret_pattern(s),
        Value::Array(a) => a.iter().find_map(value_secret),
        Value::Object(o) => o
            .iter()
            .find_map(|(k, v)| secret_pattern(k).or_else(|| value_secret(v))),
        _ => None,
    }
}

/// 2026-10-23: The finding codes that make `metrale-kg check` fail (queries part §4).
pub const FATAL: &[ErrorCode] = &[
    ErrorCode::KgDuplicateId,
    ErrorCode::KgIdCollision,
    ErrorCode::KgBadLedgerRow,
    ErrorCode::KgTooManyLabels,
    ErrorCode::KgLkbFileRejected,
];

impl Bundle {
    /// 2026-10-23: The findings whose code is in [`FATAL`].
    pub fn fatal(&self) -> Vec<&Finding> {
        self.errors
            .iter()
            .filter(|f| FATAL.iter().any(|c| c.name() == f.code))
            .collect()
    }
}
