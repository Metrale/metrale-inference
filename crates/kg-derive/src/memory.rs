// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The memory directory → `Rule` (feedback) and `Memory` nodes with `LINKS` edges
//! (ontology §2; owner decision 7: every file, with the secrets denylist). Only frontmatter
//! (`name`, `description`, `metadata.type`, `metadata.modified`) and link targets are read; bodies
//! are never stored or embedded.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - A credential-named file (withheld by the host) yields a `redacted` node with no description,
//!   no links and no text: its contents never reach the bundle.
//! - `LINKS` weight is 1/outdegree, so long link lists do not dominate ranking.

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::prelude::*;
use crate::secrets::name_is_secret;
use crate::text::{md_links, wikilinks};
use serde_json::Value;

/// 2026-10-23: The frontmatter fields read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Front {
    pub name: Option<String>,
    pub description: Option<String>,
    pub kind: Option<String>,
    pub modified: Option<String>,
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    let q = (v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\''));
    if q && v.len() >= 2 {
        v[1..v.len() - 1].replace("\\\"", "\"")
    } else {
        v.to_string()
    }
}

/// 2026-10-23: The YAML frontmatter between the first two `---` lines (flat keys and the
/// `metadata:` block's indented keys), and the body after it.
pub fn frontmatter(text: &str) -> (Front, &str) {
    let mut f = Front::default();
    let Some(rest) = text.strip_prefix("---\n") else {
        return (f, text);
    };
    let Some(end) = rest.find("\n---") else {
        return (f, text);
    };
    let body = rest[end + 4..].trim_start_matches(['\n', '-']);
    for line in rest[..end].lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let v = unquote(v);
        match k.trim() {
            "name" => f.name = Some(v),
            "description" => f.description = Some(v),
            "type" => f.kind = Some(v),
            "modified" => f.modified = Some(v),
            _ => {}
        }
    }
    (f, body)
}

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.strip_suffix(".md").unwrap_or(name).to_string()
}

fn label_for(kind: Option<&str>, stem: &str) -> Label {
    if kind == Some("feedback") || stem.starts_with("feedback_") {
        Label::Rule
    } else {
        Label::Memory
    }
}

/// 2026-10-23: Derive every memory file under `dir`.
pub fn derive<'a>(b: &mut Builder, files: impl Iterator<Item = (&'a str, Option<&'a str>)>) {
    let mut ids: BTreeMap<String, KgId> = BTreeMap::new();
    let mut links: Vec<(KgId, String, Vec<String>)> = Vec::new();
    for (path, text) in files {
        if !path.ends_with(".md") {
            continue;
        }
        let at = format!("{path}:1");
        let st = stem(path);
        let (front, body) = match text {
            Some(t) if !name_is_secret(path) => frontmatter(t),
            _ => (Front::default(), ""),
        };
        let redacted = text.is_none() || name_is_secret(path);
        let name = front
            .name
            .clone()
            .filter(|n| !redacted && !n.is_empty())
            .unwrap_or_else(|| st.clone());
        let label = label_for(front.kind.as_deref(), &name);
        let Ok(id) = KgId::new(label, name.clone()) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("file".into(), Value::String(path.to_string()));
        if redacted {
            p.insert("redacted".into(), Value::Bool(true));
        } else {
            let opt = |v: &Option<String>| v.clone().map_or(Value::Null, Value::String);
            p.insert("description".into(), opt(&front.description));
            p.insert("type".into(), opt(&front.kind));
            p.insert("modified".into(), opt(&front.modified));
            let stars = front
                .description
                .as_deref()
                .unwrap_or("")
                .matches('★')
                .count();
            p.insert("stars".into(), Value::from(stars));
        }
        let canon = format!("{name}\n{}", front.description.as_deref().unwrap_or(""));
        b.define(id.clone(), at.clone(), &canon, p);
        ids.insert(st.clone(), id.clone());
        ids.insert(name, id.clone());
        if redacted {
            continue;
        }
        if let Some(d) = &front.description {
            b.text(&id, "description", d);
        }
        let mut targets = wikilinks(body);
        for t in md_links(body) {
            if !targets.contains(&t) {
                targets.push(t);
            }
        }
        links.push((id, at, targets));
    }
    for (from, at, targets) in links {
        let resolved: Vec<KgId> = targets
            .iter()
            .filter_map(|t| {
                ids.get(t)
                    .cloned()
                    .or_else(|| KgId::new(label_for(None, t), t.clone()).ok())
            })
            .filter(|t| *t != from)
            .collect();
        let n = resolved.len();
        for to in resolved {
            let w = 1.0 / f32::from(u16::try_from(n).unwrap_or(u16::MAX));
            b.edge(&from, Rel::Links, &to, w, &at);
        }
    }
}
