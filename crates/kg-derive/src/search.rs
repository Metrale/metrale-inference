// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_search` ranking (design §3.2 tool 2, queries Q5): seeds from text similarity
//! (cosine of embeddings in `hybrid` mode, a lexical TF-IDF cosine in `graph_only` mode) and from
//! ids or lever names the query names verbatim, expanded `hops` steps through `TOGGLES`,
//! `PREDICTED_BY`, `LANDS_AS`, `DEPENDS_ON`, `LINKS` and `CITES` in both directions.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - `score = w_cosine * sim * decay + w_hops * decay + w_kind * [kind requested]`, with
//!   `decay = 2^-hop`, the weights from config (`[search]`); no weight is a constant here.
//! - The lexical channel runs in both modes, so the `graph_only` result is a strict subset of the
//!   inputs `hybrid` sees, never a different algorithm.
//! - Snippets are the indexed text bodies (memory files index only their `description`).

use crate::bundle::Bundle;
use crate::config::Search;
use crate::id::{KgId, Label, Rel};
use crate::prelude::*;

/// 2026-10-23: One hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: KgId,
    pub score: f64,
    pub snippet: String,
    pub src: String,
    pub via: Vec<String>,
}

const EXPAND: [Rel; 6] = [
    Rel::Toggles,
    Rel::PredictedBy,
    Rel::LandsAs,
    Rel::DependsOn,
    Rel::Links,
    Rel::Cites,
];
const SNIPPET: usize = 200;

#[allow(clippy::cast_precision_loss)]
fn ratio(n: usize, d: usize) -> f64 {
    // 2026-10-23: document counts stay far below 2^52, so the cast is exact.
    1.0 + n as f64 / d as f64
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

fn body(text: &str) -> &str {
    text.split_once(": ").map_or(text, |(_, b)| b)
}

/// 2026-10-23: Lexical similarity of `query` to every node's texts: the max over a node's texts
/// of the TF-IDF cosine over lower-cased words longer than two characters.
pub fn lexical(b: &Bundle, query: &str) -> BTreeMap<KgId, f64> {
    let docs: Vec<(&KgId, Vec<String>)> = b
        .texts
        .iter()
        .map(|t| (&t.node, words(body(&t.text))))
        .collect();
    let n = docs.len().max(1);
    let mut df: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, ws) in &docs {
        let uniq: BTreeSet<&str> = ws.iter().map(String::as_str).collect();
        for w in uniq {
            *df.entry(w).or_insert(0) += 1;
        }
    }
    let idf = |w: &str| libm::log(ratio(n, df.get(w).copied().unwrap_or(0) + 1));
    fn vec_of<'w>(ws: &'w [String], idf: &impl Fn(&str) -> f64) -> BTreeMap<&'w str, f64> {
        let mut v: BTreeMap<&str, f64> = BTreeMap::new();
        for w in ws {
            *v.entry(w.as_str()).or_insert(0.0) += 1.0;
        }
        for (w, x) in v.iter_mut() {
            *x *= idf(w);
        }
        v
    }
    let qw = words(query);
    let q = vec_of(&qw, &idf);
    let qn: f64 = libm::sqrt(q.values().map(|x| x * x).sum());
    let mut out: BTreeMap<KgId, f64> = BTreeMap::new();
    if qn == 0.0 {
        return out;
    }
    for (id, ws) in &docs {
        let d = vec_of(ws, &idf);
        let dot: f64 = q.iter().filter_map(|(w, x)| d.get(w).map(|y| x * y)).sum();
        if dot <= 0.0 {
            continue;
        }
        let dn: f64 = libm::sqrt(d.values().map(|x| x * x).sum());
        let s = dot / (qn * dn);
        let e = out.entry((*id).clone()).or_insert(0.0);
        if s > *e {
            *e = s;
        }
    }
    out
}

/// 2026-10-23: Nodes the query names verbatim (`ledger:x`, a ledger id, a `METRALE_*` lever).
pub fn named(b: &Bundle, query: &str) -> BTreeSet<KgId> {
    let mut out = BTreeSet::new();
    for tok in
        query.split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '`' | '"'))
    {
        let tok = tok.trim_end_matches(['.', ':']);
        if tok.is_empty() {
            continue;
        }
        let cands = [
            KgId::parse(tok).ok(),
            KgId::new(Label::LedgerEntry, tok).ok(),
            KgId::new(Label::Lever, tok).ok(),
        ];
        out.extend(
            cands
                .into_iter()
                .flatten()
                .filter(|id| b.node(id).is_some()),
        );
    }
    out
}

/// 2026-10-23: Rank. `sims` is the per-node similarity of the active mode (already the max of
/// the cosine and lexical channels in `hybrid`); `kinds` empty means every label.
pub fn rank(
    b: &Bundle,
    sims: &BTreeMap<KgId, f64>,
    named: &BTreeSet<KgId>,
    kinds: &[Label],
    k: usize,
    hops: u32,
    w: &Search,
) -> Vec<Hit> {
    struct Best {
        score: f64,
        via: Vec<String>,
    }
    let mut best: BTreeMap<KgId, Best> = BTreeMap::new();
    let wanted = |id: &KgId| kinds.is_empty() || kinds.contains(&id.label());
    let mut frontier: Vec<(KgId, f64, Vec<String>)> = sims
        .iter()
        .map(|(id, s)| (id.clone(), *s, Vec::new()))
        .collect();
    frontier.extend(named.iter().map(|id| (id.clone(), 1.0, Vec::new())));
    for hop in 0..=hops {
        let decay = 1.0 / f64::from(1u32 << hop.min(16));
        let mut next = Vec::new();
        for (id, sim, via) in frontier {
            let score = w.w_cosine * sim * decay
                + w.w_hops * decay
                + if wanted(&id) { w.w_kind } else { 0.0 };
            let better = best.get(&id).is_none_or(|b| score > b.score);
            if better {
                best.insert(
                    id.clone(),
                    Best {
                        score,
                        via: via.clone(),
                    },
                );
            }
            if hop == hops || !better {
                continue;
            }
            let mut hop_via = via.clone();
            hop_via.push(id.to_string());
            for rel in EXPAND {
                let outs = b.out(&id, rel).map(|e| e.to.clone());
                let ins = b.into(&id, rel).map(|e| e.from.clone());
                for n in outs.chain(ins).filter(|n| n.label() != Label::Unresolved) {
                    next.push((n, sim, hop_via.clone()));
                }
            }
        }
        frontier = next;
    }
    let mut hits: Vec<Hit> = best
        .into_iter()
        .filter(|(id, _)| wanted(id) && b.node(id).is_some())
        .map(|(id, x)| {
            let snippet: String = b
                .texts
                .iter()
                .find(|t| t.node == id)
                .map_or("", |t| body(&t.text))
                .chars()
                .take(SNIPPET)
                .collect();
            let src = b.node(&id).map_or_else(String::new, |n| n.src.clone());
            Hit {
                id,
                score: x.score,
                snippet,
                src,
                via: x.via,
            }
        })
        .collect();
    hits.sort_by(|a, c| c.score.total_cmp(&a.score).then_with(|| a.id.cmp(&c.id)));
    hits.truncate(k);
    hits
}
