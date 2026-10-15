// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the `metrale.kkb/1` record model: the kinds (one label per
//! point), the relation table (u16 ids, order fixed, append-only) and the
//! record envelope common to every kind.
//!
//! A record is either complete (`body` present, typed per kind, unknown keys
//! refused) or a reference stub (`body` absent): an entity a run points at but
//! does not describe (its hardware class, its campaign, a kernel family). A
//! stub is replaced by the complete record when its source of truth is
//! ingested, and never overwrites one.

mod context;
mod graph;
mod journal;

pub use context::*;
pub use graph::*;
pub use journal::*;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{Error, Result, bail};

pub const SCHEMA: &str = "metrale.kkb/1";
/// The deepest kernel-tree level materialised as `KernelSample` points
/// (0-based: top-level nodes are depth 0).
pub const MAX_SAMPLE_DEPTH: usize = 2;
/// How many previous `content_sha256` values a point keeps in `supersedes`.
pub const SUPERSEDES_KEPT: usize = 16;
/// The natural-key part separator.
pub const KEY_SEP: &str = "‖";

macro_rules! kinds {
    ($($k:ident),* $(,)?) => {
        /// The point labels, in label-bit order (append-only).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub enum Kind { $($k),* }

        impl Kind {
            pub const ALL: &'static [Kind] = &[$(Kind::$k),*];

            pub fn label(self) -> &'static str {
                match self { $(Kind::$k => stringify!($k)),* }
            }
        }

        /// A complete record's kind-specific fields.
        #[derive(Debug, Clone, PartialEq, Serialize)]
        #[serde(untagged)]
        pub enum Body { $($k(::std::boxed::Box<$k>)),* }

        impl Body {
            pub fn kind(&self) -> Kind {
                match self { $(Body::$k(_) => Kind::$k),* }
            }

            /// The body of a `kind` record from its JSON object; unknown keys fail.
            pub fn from_value(kind: Kind, v: Value) -> Result<Body> {
                let r = match kind {
                    $(Kind::$k => serde_json::from_value::<$k>(v).map(|b| Body::$k(::std::boxed::Box::new(b)))),*
                };
                r.map_err(|e| Error(format!("{} body: {e}", kind.label())))
            }
        }
    };
}

kinds!(
    Kernel,
    KernelSymbol,
    Instantiation,
    ProfileRun,
    KernelSample,
    FusionRule,
    CircuitPlan,
    Bound,
    Campaign,
    HardwareClass,
    Model,
    Pattern,
    Lever,
    Candidate,
    Prediction,
    Build,
    Measurement,
    ABRecord,
    Gate,
    Box,
    ProcessMetric,
    PlaybookRevision,
    Evidence,
    Doc,
);

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        Kind::ALL.iter().copied().find(|k| k.label() == s)
    }

    /// The point's label bit (LatticeDB allows 64 labels per collection).
    pub fn bit(self) -> u64 {
        1u64 << (self as u32)
    }
}

macro_rules! relations {
    ($($r:ident),* $(,)?) => {
        /// The edge relations; the id is the position (order fixed, append-only).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        pub enum Relation { $($r),* }

        impl Relation {
            pub const ALL: &'static [Relation] = &[$(Relation::$r),*];

            pub fn name(self) -> &'static str {
                match self { $(Relation::$r => stringify!($r)),* }
            }
        }
    };
}

relations!(
    INSTANTIATES,
    ON_HARDWARE,
    OF_MODEL,
    IN_CAMPAIGN,
    HAS_SAMPLE,
    SAMPLE_OF,
    PARENT_OF,
    FUSES,
    RULE_PRODUCES,
    RULE_CONSUMES,
    PLANNED_BY,
    SYMBOL_OF,
    BOUND_OF,
    ASSESSED_IN,
    BUILT_FROM,
    MEASURED,
    ON_BOX,
    TARGETS,
    PROFILE_BEFORE,
    PROFILE_AFTER,
    AB_OF,
    PREDICTS,
    CHOSEN_AS,
    EXEMPLIFIES,
    CONTRADICTS,
    SUPERSEDES,
    CAUGHT_BY,
    WOULD_CATCH,
    EVIDENCE,
    DERIVED_FROM,
    REVISES,
    ABOUT,
    SOURCED_FROM,
);

impl Relation {
    pub fn id(self) -> u16 {
        self as u16
    }

    pub fn parse(s: &str) -> Option<Relation> {
        Relation::ALL.iter().copied().find(|r| r.name() == s)
    }
}

/// A point by kind and natural key (its id is `ids::point_id` of the two).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeRef {
    pub kind: Kind,
    pub key: String,
}

impl NodeRef {
    pub fn new(kind: Kind, key: impl Into<String>) -> NodeRef {
        NodeRef {
            kind,
            key: key.into(),
        }
    }
}

/// One record. `key` is the natural key (DESIGN's `id`; stored under `key`
/// because `id` is LatticeDB's virtual point-id property).
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub kind: Kind,
    pub key: String,
    pub summary: String,
    pub recorded_at: String,
    pub private: bool,
    pub body: Option<Body>,
}

impl Record {
    pub fn complete(
        key: impl Into<String>,
        summary: impl Into<String>,
        recorded_at: &str,
        private: bool,
        body: Body,
    ) -> Record {
        Record {
            kind: body.kind(),
            key: key.into(),
            summary: summary.into(),
            recorded_at: recorded_at.to_string(),
            private,
            body: Some(body),
        }
    }

    /// A reference stub: known only by its key; its summary is `<Kind> <key>`.
    pub fn stub(node: &NodeRef, recorded_at: &str, private: bool) -> Record {
        Record {
            kind: node.kind,
            key: node.key.clone(),
            summary: format!("{} {}", node.kind.label(), node.key),
            recorded_at: recorded_at.to_string(),
            private,
            body: None,
        }
    }

    pub fn node(&self) -> NodeRef {
        NodeRef::new(self.kind, self.key.clone())
    }

    pub fn is_stub(&self) -> bool {
        self.body.is_none()
    }

    /// What `content_sha256` covers: everything but `recorded_at` (the same
    /// content recorded twice is one version) and the store's own history.
    pub fn content(&self) -> Result<Value> {
        let body = match &self.body {
            Some(b) => serde_json::to_value(b).map_err(|e| Error(e.to_string()))?,
            None => Value::Null,
        };
        Ok(json!({
            "schema": SCHEMA,
            "kind": self.kind.label(),
            "key": self.key,
            "summary": self.summary,
            "private": self.private,
            "body": body,
        }))
    }
}

/// The payload keys of a point, in one place (the store adds `_labels`).
pub const PAYLOAD_KEYS: [&str; 10] = [
    "schema",
    "kind",
    "key",
    "summary",
    "content_sha256",
    "recorded_at",
    "private",
    "supersedes",
    "space",
    "body",
];

/// The stored form of a record: its payload fields as JSON values.
pub fn to_payload(
    r: &Record,
    content_sha256: &str,
    supersedes: &[String],
    space: &str,
) -> Result<Map<String, Value>> {
    let content = r.content()?;
    let mut m = Map::new();
    m.insert("schema".into(), json!(SCHEMA));
    m.insert("kind".into(), json!(r.kind.label()));
    m.insert("key".into(), json!(r.key));
    m.insert("summary".into(), json!(r.summary));
    m.insert("content_sha256".into(), json!(content_sha256));
    m.insert("recorded_at".into(), json!(r.recorded_at));
    m.insert("private".into(), json!(r.private));
    m.insert("supersedes".into(), json!(supersedes));
    m.insert("space".into(), json!(space));
    m.insert("body".into(), content["body"].clone());
    Ok(m)
}

/// A point payload read back: the record, its `content_sha256`, its
/// `supersedes` history and its embedding space. Every key is required and
/// no other key (but `_labels`) is accepted.
pub struct Stored {
    pub record: Record,
    pub content_sha256: String,
    pub supersedes: Vec<String>,
    pub space: String,
}

fn field<'m>(m: &'m Map<String, Value>, k: &str) -> Result<&'m Value> {
    m.get(k)
        .ok_or_else(|| Error(format!("kkb payload: missing key {k:?}")))
}

fn text(m: &Map<String, Value>, k: &str) -> Result<String> {
    field(m, k)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error(format!("kkb payload: {k:?} must be a string")))
}

pub fn from_payload(m: &Map<String, Value>) -> Result<Stored> {
    for k in m.keys() {
        if k != "_labels" && !PAYLOAD_KEYS.contains(&k.as_str()) {
            bail!("kkb payload: unknown key {k:?}");
        }
    }
    if text(m, "schema")? != SCHEMA {
        bail!("kkb payload: schema is not {SCHEMA:?}");
    }
    let label = text(m, "kind")?;
    let kind = Kind::parse(&label).ok_or_else(|| Error(format!("kkb payload: kind {label:?}")))?;
    let body = match field(m, "body")? {
        Value::Null => None,
        v => Some(Body::from_value(kind, v.clone())?),
    };
    let private = field(m, "private")?
        .as_bool()
        .ok_or_else(|| Error("kkb payload: \"private\" must be a bool".into()))?;
    let supersedes = serde_json::from_value(field(m, "supersedes")?.clone())
        .map_err(|e| Error(format!("kkb payload: supersedes: {e}")))?;
    Ok(Stored {
        record: Record {
            kind,
            key: text(m, "key")?,
            summary: text(m, "summary")?,
            recorded_at: text(m, "recorded_at")?,
            private,
            body,
        },
        content_sha256: text(m, "content_sha256")?,
        supersedes,
        space: text(m, "space")?,
    })
}
