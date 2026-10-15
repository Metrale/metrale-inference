// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the LatticeDB store (DESIGN §3.1 step 3): one durable
//! collection per embedding space, every relation declared up front (a
//! relation the engine creates on the fly is not persisted), points keyed by
//! `ids::node_id`, payload fields JSON-encoded with the kind in `_labels`.
//!
//! Upsert rules: a stub is written only where nothing is; a complete record
//! whose `content_sha256` is unchanged is left alone (first `recorded_at`
//! kept); a changed one replaces the point, its previous digest appended to
//! `supersedes` (the last `SUPERSEDES_KEPT`). An edge already present with the
//! same weight is left alone; with another weight it is replaced. Edges a
//! changed record no longer derives are not removed.

use std::collections::HashMap;
use std::io::Write;

use lattice_core::{
    CollectionConfig, CollectionEngine, Distance, DurabilityMode, ExportOptions, HnswConfig,
    LatticeStorage, Point, VectorConfig, export_jsonl,
};
use serde_json::{Map, Value, json};

use crate::derive::Graph;
use crate::embed::{Embedder, HASHED_COLLECTION, Space};
use crate::ids::node_id;
use crate::schema::{NodeRef, Record, Relation, SUPERSEDES_KEPT, Stored, from_payload, to_payload};
use crate::{Error, Result, bail, canon};

/// The HNSW parameters of every KB collection (explicit, never a default).
pub const HNSW_M: usize = 16;
pub const HNSW_M0: usize = 32;
pub const HNSW_EF: usize = 64;
pub const HNSW_EF_CONSTRUCTION: usize = 200;

/// The payload key carrying a point's labels.
pub const LABELS_KEY: &str = "_labels";

fn lattice(what: &str, e: impl std::fmt::Display) -> Error {
    Error(format!("lattice: {what}: {e}"))
}

/// The collection a space's points live in.
pub fn collection_name(space: &Space) -> String {
    if *space == Space::hashed() {
        HASHED_COLLECTION.to_string()
    } else {
        format!("kkb@{}", space.id())
    }
}

/// The durable collection configuration of `space`.
pub fn collection_config(space: &Space) -> CollectionConfig {
    let hnsw = HnswConfig {
        m: HNSW_M,
        m0: HNSW_M0,
        ml: HnswConfig::recommended_ml(HNSW_M),
        ef: HNSW_EF,
        ef_construction: HNSW_EF_CONSTRUCTION,
    };
    let vectors = VectorConfig::new(space.dims(), Distance::Cosine);
    Relation::ALL
        .iter()
        .fold(
            CollectionConfig::new(collection_name(space), vectors, hnsw),
            |c, r| c.with_relation(r.name(), r.id()),
        )
        .with_durability(DurabilityMode::Durable)
}

/// What one `apply` changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Applied {
    pub inserted: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub edges_added: usize,
    pub edges_updated: usize,
}

impl Applied {
    pub fn add(&mut self, o: Applied) {
        self.inserted += o.inserted;
        self.updated += o.updated;
        self.unchanged += o.unchanged;
        self.edges_added += o.edges_added;
        self.edges_updated += o.edges_updated;
    }

    /// True when nothing was written.
    pub fn is_noop(&self) -> bool {
        self.inserted + self.updated + self.edges_added + self.edges_updated == 0
    }
}

/// A durable KB collection.
pub struct Store<W: LatticeStorage, D: LatticeStorage> {
    engine: CollectionEngine<W, D>,
    space: Space,
}

fn encode_payload(m: &Map<String, Value>, label: &str) -> Result<HashMap<String, Vec<u8>>> {
    let mut p = HashMap::new();
    for (k, v) in m {
        p.insert(k.clone(), canon::canonical(v)?);
    }
    p.insert(LABELS_KEY.to_string(), canon::canonical(&json!([label]))?);
    Ok(p)
}

fn decode_payload(p: &HashMap<String, Vec<u8>>) -> Result<Stored> {
    let mut m = Map::new();
    for (k, v) in p {
        let v: Value = serde_json::from_slice(v).map_err(|e| lattice("payload", e))?;
        m.insert(k.clone(), v);
    }
    from_payload(&m)
}

impl<W: LatticeStorage, D: LatticeStorage> Store<W, D> {
    /// Open (replaying the WAL) the collection of `space`; every point
    /// already there must be of that space.
    pub async fn open(space: &Space, wal: W, data: D) -> Result<Self> {
        let engine = CollectionEngine::open_durable(collection_config(space), wal, data)
            .await
            .map_err(|e| lattice("open", e))?;
        let store = Store {
            engine,
            space: space.clone(),
        };
        for id in store.engine.point_ids().map_err(|e| lattice("ids", e))? {
            let stored = store.stored(id)?;
            if let Some(s) = stored
                && s.space != space.id()
            {
                bail!(
                    "lattice: point {id:016x} is of space {:?}, not {:?}; vectors of two spaces never mix",
                    s.space,
                    space.id()
                );
            }
        }
        Ok(store)
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    fn stored(&self, id: u64) -> Result<Option<Stored>> {
        match self.engine.get_point(id).map_err(|e| lattice("get", e))? {
            Some(p) => decode_payload(&p.payload).map(Some),
            None => Ok(None),
        }
    }

    /// A record as stored, by kind and natural key.
    pub fn get(&self, node: &NodeRef) -> Result<Option<Stored>> {
        self.stored(node_id(node))
    }

    /// The new payload for `r`, or None when the stored point stays.
    fn next_payload(&self, r: &Record) -> Result<Option<(Map<String, Value>, bool)>> {
        let old = self.get(&r.node())?;
        let sha = canon::content_sha256(r)?;
        let supersedes = match &old {
            None => Vec::new(),
            Some(_) if r.is_stub() => return Ok(None),
            Some(o) if o.content_sha256 == sha => return Ok(None),
            Some(o) if o.record.is_stub() => o.supersedes.clone(),
            Some(o) => {
                let mut s = o.supersedes.clone();
                s.push(o.content_sha256.clone());
                let skip = s.len().saturating_sub(SUPERSEDES_KEPT);
                s.split_off(skip)
            }
        };
        let payload = to_payload(r, &sha, &supersedes, self.space.id())?;
        Ok(Some((payload, old.is_some())))
    }

    /// Write a derived graph (one batched upsert, then the edges).
    pub async fn apply(&mut self, graph: &Graph, embedder: &dyn Embedder) -> Result<Applied> {
        self.space.require(embedder.space())?;
        let mut done = Applied::default();
        let mut changed = Vec::new();
        for r in &graph.records {
            match self.next_payload(r)? {
                None => done.unchanged += 1,
                Some((payload, existed)) => {
                    if existed {
                        done.updated += 1;
                    } else {
                        done.inserted += 1;
                    }
                    changed.push((r, payload));
                }
            }
        }
        let texts: Vec<&str> = changed.iter().map(|(r, _)| r.summary.as_str()).collect();
        let vectors = embedder.embed(&texts)?;
        if vectors.len() != changed.len() {
            bail!(
                "lattice: the embedder returned {} vectors for {} texts",
                vectors.len(),
                texts.len()
            );
        }
        let mut points = Vec::with_capacity(changed.len());
        for ((r, payload), v) in changed.iter().zip(vectors) {
            self.space.require(&v.space)?;
            let mut p = Point::new_vector(node_id(&r.node()), v.vector);
            p.payload = encode_payload(payload, r.kind.label())?;
            p.label_bitmap = r.kind.bit();
            points.push(p);
        }
        if !points.is_empty() {
            self.engine
                .upsert_points_async(points)
                .await
                .map_err(|e| lattice("upsert", e))?;
        }
        for e in &graph.edges {
            let (from, to, rel) = (node_id(&e.from), node_id(&e.to), e.rel.name());
            let present = self
                .engine
                .get_edges(from)
                .map_err(|x| lattice("edges", x))?
                .into_iter()
                .find(|x| x.target_id == to && x.relation == rel);
            match present {
                Some(x) if x.weight == e.weight => continue,
                Some(_) => {
                    self.engine
                        .remove_edge_async(from, to, Some(rel))
                        .await
                        .map_err(|x| lattice("remove edge", x))?;
                    done.edges_updated += 1;
                }
                None => done.edges_added += 1,
            }
            self.engine
                .add_edge_async(from, to, rel, e.weight)
                .await
                .map_err(|x| lattice("add edge", x))?;
        }
        Ok(done)
    }

    /// The collection as deterministic JSONL, vectors omitted.
    pub fn export(&self, out: &mut dyn Write) -> Result<()> {
        export_jsonl(
            &self.engine,
            self.engine.config(),
            &mut WriteRef(out),
            &ExportOptions::omit_vectors(),
        )
        .map(|_| ())
        .map_err(|e| lattice("export", e))
    }
}

/// `export_jsonl` takes a sized writer.
struct WriteRef<'a>(&'a mut dyn Write);

impl Write for WriteRef<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}
