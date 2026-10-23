// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The derived LatticeDB store (design §2.2): collection `kg` holds every node (vector
//! size 1, never searched) with its edges; a `meta` point carries the bundle digest. Loading is a
//! replace: when the stored digest differs from the bundle's, every point is deleted and the bundle
//! is written again, so the store never holds a fact the bundle does not.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - The store is rebuildable: delete `data_dir`, load the same bundle, and [`Store::export`] is
//!   byte-identical (the P0 gate).
//! - Only [`Store::load`] writes the `kg` collection.

use anyhow::{Context, Result, anyhow, bail};
use lattice_core::cypher::handler::{CypherHandler, DefaultCypherHandler};
use lattice_core::{
    CollectionConfig, CollectionEngine, CypherValue, Distance, DurabilityMode, Edge, ExportOptions,
    HnswConfig, Point, VectorConfig, export_jsonl,
};
use lattice_storage::DiskStorage;
use metrale_kg_derive::hash::point_id;
use metrale_kg_derive::jsonl::digest;
use metrale_kg_derive::{Bundle, KgId, Rel};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// 2026-10-23: The `meta` point's id: not the point id of any typed id (those carry a `:`).
pub const META_ID: u64 = 0;

/// 2026-10-23: The point layout; part of the stored digest so a layout change reloads the store
/// even when the bundle is unchanged (v2: the typed id moved from `id` to `name`).
pub const STORE_FORMAT: &str = "kg-store/2";

/// 2026-10-23: The `kg` collection's config: size-1 vectors, the ontology's relations by id.
pub fn graph_config() -> CollectionConfig {
    let hnsw = HnswConfig {
        m: 4,
        m0: 8,
        ml: HnswConfig::recommended_ml(4),
        ef: 8,
        ef_construction: 8,
    };
    let mut c = CollectionConfig::new("kg", VectorConfig::new(1, Distance::Euclid), hnsw)
        .with_durability(DurabilityMode::Durable);
    for r in Rel::ALL {
        c = c.with_relation(r.name(), r.id());
    }
    c
}

type Engine = CollectionEngine<DiskStorage, DiskStorage>;

/// 2026-10-23: The open store.
pub struct Store {
    engine: Engine,
    /// 2026-10-23: point id → typed id, for rendering `NodeRef`s.
    names: HashMap<u64, KgId>,
}

fn json_bytes(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

fn node_point(n: &metrale_kg_derive::Node, edges: &[Edge]) -> Point {
    let mut payload: HashMap<String, Vec<u8>> = HashMap::new();
    payload.insert(
        "_labels".into(),
        json_bytes(&Value::from(vec![n.id.label().name()])),
    );
    // 2026-10-23: `id` is LatticeDB's reserved point-id property, so the typed id is `name`
    // (queries filter and sort on it) and `.id` result columns are mapped back in `query`.
    payload.insert("name".into(), json_bytes(&Value::String(n.id.to_string())));
    payload.insert("src".into(), json_bytes(&Value::String(n.src.clone())));
    payload.insert("sha".into(), json_bytes(&Value::String(n.sha.hex())));
    for (k, v) in &n.p {
        if !matches!(k.as_str(), "id" | "name" | "src" | "sha") {
            payload.insert(k.clone(), json_bytes(v));
        }
    }
    Point::new_graph(point_id(&n.id), vec![0.0], edges.iter().cloned().collect())
        .with_payload(payload)
}

/// 2026-10-23: A WAL and a data `DiskStorage` under `dir` (`DiskStorage` is not `Clone`, so the
/// unified opener does not apply).
pub async fn disk_pair(dir: &Path) -> Result<(DiskStorage, DiskStorage)> {
    let mut out = Vec::new();
    for part in ["wal", "data"] {
        let d = dir.join(part);
        std::fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
        let s = DiskStorage::with_defaults(d);
        s.init()
            .await
            .map_err(|e| anyhow!("store init {part}: {e:?}"))?;
        out.push(s);
    }
    let data = out.pop().ok_or_else(|| anyhow!("no data storage"))?;
    let wal = out.pop().ok_or_else(|| anyhow!("no wal storage"))?;
    Ok((wal, data))
}

impl Store {
    /// 2026-10-23: Open (or create) the durable store under `dir`.
    pub async fn open(dir: &Path) -> Result<Store> {
        let (wal, data) = disk_pair(&dir.join("kg")).await?;
        let engine = CollectionEngine::open_durable(graph_config(), wal, data)
            .await
            .map_err(|e| anyhow!("kg store open: {e:?}"))?;
        Ok(Store {
            engine,
            names: HashMap::new(),
        })
    }

    /// 2026-10-23: The digest the store was last loaded with.
    pub fn stored_digest(&self) -> Option<String> {
        let p = self.engine.get_point(META_ID).ok().flatten()?;
        let v: Value = serde_json::from_slice(p.payload.get("digest")?).ok()?;
        v.as_str().map(str::to_string)
    }

    /// 2026-10-23: Make the store hold exactly `b`; returns whether anything was written.
    pub async fn load(&mut self, b: &Bundle) -> Result<bool> {
        self.names = b
            .nodes
            .iter()
            .map(|n| (point_id(&n.id), n.id.clone()))
            .collect();
        let d = format!("{STORE_FORMAT}/{}", digest(b).hex());
        if self.stored_digest().as_deref() == Some(d.as_str()) {
            return Ok(false);
        }
        let old = self
            .engine
            .point_ids()
            .map_err(|e| anyhow!("kg store ids: {e:?}"))?;
        if !old.is_empty() {
            self.engine
                .delete_points_async(&old)
                .await
                .map_err(|e| anyhow!("kg store clear: {e:?}"))?;
        }
        let mut out: BTreeMap<u64, Vec<Edge>> = BTreeMap::new();
        for e in &b.edges {
            out.entry(point_id(&e.from)).or_default().push(Edge::new(
                point_id(&e.to),
                e.w,
                e.rel.id(),
            ));
        }
        let mut points: Vec<Point> = b
            .nodes
            .iter()
            .map(|n| node_point(n, out.get(&point_id(&n.id)).map_or(&[][..], Vec::as_slice)))
            .collect();
        let mut meta = HashMap::new();
        meta.insert(
            "_labels".to_string(),
            json_bytes(&Value::from(vec!["Meta"])),
        );
        meta.insert("digest".to_string(), json_bytes(&Value::String(d)));
        points.push(Point::new_vector(META_ID, vec![0.0]).with_payload(meta));
        self.engine
            .upsert_points_async(points)
            .await
            .map_err(|e| anyhow!("kg store upsert: {e:?}"))?;
        Ok(true)
    }

    /// 2026-10-23: The deterministic JSONL export of `kg` (vectors omitted: they are constant).
    pub fn export(&self) -> Result<String> {
        let mut out = Vec::new();
        export_jsonl(
            &self.engine,
            self.engine.config(),
            &mut out,
            &ExportOptions::omit_vectors(),
        )
        .map_err(|e| anyhow!("kg store export: {e:?}"))?;
        String::from_utf8(out).context("export is not utf-8")
    }

    /// 2026-10-23: Run a query that [`crate::cypher::read_only`] accepted; rows as JSON.
    pub fn query(
        &mut self,
        cypher: &str,
        params: HashMap<String, CypherValue>,
    ) -> Result<(Vec<String>, Vec<Vec<Value>>)> {
        let h = DefaultCypherHandler::new();
        let r = h
            .query(cypher, &mut self.engine, params)
            .map_err(|e| anyhow!("{e}"))?;
        let is_id: Vec<bool> = r
            .columns
            .iter()
            .map(|c| c == "id" || c.ends_with(".id"))
            .collect();
        let rows = r
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&is_id)
                    .map(|(v, id)| match v {
                        CypherValue::Int(i) if *id => {
                            self.name_of(u64::from_ne_bytes(i.to_ne_bytes()))
                        }
                        other => self.json(other),
                    })
                    .collect()
            })
            .collect();
        Ok((r.columns, rows))
    }

    fn name_of(&self, point: u64) -> Value {
        Value::String(
            self.names
                .get(&point)
                .map_or_else(|| format!("point:{point}"), ToString::to_string),
        )
    }

    fn json(&self, v: &CypherValue) -> Value {
        match v {
            CypherValue::Null => Value::Null,
            CypherValue::Bool(b) => Value::Bool(*b),
            CypherValue::Int(i) => Value::from(*i),
            CypherValue::Float(f) => Value::from(*f),
            CypherValue::String(s) => Value::String(s.to_string()),
            CypherValue::List(l) => Value::Array(l.iter().map(|x| self.json(x)).collect()),
            CypherValue::Map(m) => {
                Value::Object(m.iter().map(|(k, x)| (k.clone(), self.json(x))).collect())
            }
            CypherValue::NodeRef(id) => self.name_of(*id),
            other => Value::String(format!("{other:?}")),
        }
    }
}

/// 2026-10-23: Refuse a store directory that is not absolute (the config says where it lives).
pub fn data_dir(raw: &str) -> Result<PathBuf> {
    let p = PathBuf::from(raw);
    if !p.is_absolute() {
        bail!("store.data_dir {raw} is not absolute");
    }
    Ok(p)
}
