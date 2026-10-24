// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The embedding cache (design §2.2, §4.1 invariant 4's one exception): LatticeDB
//! collection `kg_text`, one point per embedded text keyed by the text's sha, so a re-derive
//! embeds only texts whose sha is new. A meta point records the vector space (model, dimension and
//! document `input_type`); a change of any empties the cache rather than mixing vector spaces.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - A cached vector is only ever looked up by the sha of the text it embeds.
//! - The cache holds vectors and shas, never text bodies (bodies stay in the SSOT files).

use anyhow::{Result, anyhow, bail};
use lattice_core::{
    CollectionConfig, CollectionEngine, Distance, DurabilityMode, HnswConfig, Point, VectorConfig,
};
use lattice_storage::DiskStorage;
use metrale_kg_derive::hash::Sha256;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

const META_ID: u64 = 0;

type Engine = CollectionEngine<DiskStorage, DiskStorage>;

/// 2026-10-23: The open cache.
pub struct TextStore {
    engine: Engine,
    dim: usize,
}

fn point_of(sha: &Sha256) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&sha.0[..8]);
    u64::from_le_bytes(b)
}

fn bytes(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

fn str_of(p: &Point, k: &str) -> Option<String> {
    serde_json::from_slice::<Value>(p.payload.get(k)?)
        .ok()?
        .as_str()
        .map(str::to_string)
}

impl TextStore {
    /// 2026-10-23: Open `<dir>/kg_text` for vector `space` (`EmbedSettings::space`) at `dim`,
    /// emptying it if it holds another space.
    pub async fn open(dir: &Path, space: &str, dim: u32) -> Result<TextStore> {
        let d = usize::try_from(dim)?;
        let (wal, data) = crate::store::disk_pair(&dir.join("kg_text")).await?;
        let hnsw = HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 64,
            ef_construction: 100,
        };
        let cfg = CollectionConfig::new("kg_text", VectorConfig::new(d, Distance::Cosine), hnsw)
            .with_durability(DurabilityMode::Durable);
        let engine = Engine::open_durable(cfg, wal, data)
            .await
            .map_err(|e| anyhow!("kg_text open: {e:?}"))?;
        let mut s = TextStore { engine, dim: d };
        let space = space.to_string();
        let held = s
            .engine
            .get_point(META_ID)
            .ok()
            .flatten()
            .and_then(|p| str_of(&p, "space"));
        if held.as_deref() != Some(space.as_str()) {
            let ids = s
                .engine
                .point_ids()
                .map_err(|e| anyhow!("kg_text ids: {e:?}"))?;
            if !ids.is_empty() {
                s.engine
                    .delete_points_async(&ids)
                    .await
                    .map_err(|e| anyhow!("kg_text clear: {e:?}"))?;
            }
            let mut meta = HashMap::new();
            meta.insert("space".to_string(), bytes(&Value::String(space)));
            let p = Point::new_vector(META_ID, vec![1.0; d]).with_payload(meta);
            s.engine
                .upsert_points_async(vec![p])
                .await
                .map_err(|e| anyhow!("kg_text meta: {e:?}"))?;
        }
        Ok(s)
    }

    /// 2026-10-23: Every sha with a cached vector.
    pub fn cached(&self) -> Result<BTreeSet<Sha256>> {
        Ok(self.vectors()?.into_keys().collect())
    }

    /// 2026-10-23: Every cached `(sha, vector)`.
    pub fn vectors(&self) -> Result<BTreeMap<Sha256, Vec<f32>>> {
        let mut out = BTreeMap::new();
        for id in self
            .engine
            .point_ids()
            .map_err(|e| anyhow!("kg_text ids: {e:?}"))?
        {
            if id == META_ID {
                continue;
            }
            let Some(p) = self
                .engine
                .get_point(id)
                .map_err(|e| anyhow!("kg_text get: {e:?}"))?
            else {
                continue;
            };
            let Some(hex) = str_of(&p, "sha") else {
                continue;
            };
            let Some(sha) = parse_hex(&hex) else { continue };
            out.insert(sha, p.vector.clone());
        }
        Ok(out)
    }

    /// 2026-10-23: Cache vectors.
    pub async fn put(&mut self, v: Vec<(Sha256, Vec<f32>)>) -> Result<usize> {
        let mut pts = Vec::with_capacity(v.len());
        for (sha, vec) in v {
            if vec.len() != self.dim {
                bail!(
                    "kg_text: a vector of {} components for a {}-dimension cache",
                    vec.len(),
                    self.dim
                );
            }
            let id = point_of(&sha);
            if id == META_ID {
                bail!("kg_text: sha {} maps to the meta point id", sha.hex());
            }
            let mut payload = HashMap::new();
            payload.insert("sha".to_string(), bytes(&Value::String(sha.hex())));
            pts.push(Point::new_vector(id, vec).with_payload(payload));
        }
        let n = pts.len();
        if n > 0 {
            self.engine
                .upsert_points_async(pts)
                .await
                .map_err(|e| anyhow!("kg_text upsert: {e:?}"))?;
        }
        Ok(n)
    }
}

fn parse_hex(h: &str) -> Option<Sha256> {
    if h.len() != 64 {
        return None;
    }
    let mut a = [0u8; 32];
    for (i, b) in a.iter_mut().enumerate() {
        *b = u8::from_str_radix(h.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(Sha256(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-23: The pinned LatticeDB (latticedb-core 0.3.4 = lattice-db main e94ccad) must
    /// carry 95b9ade: a search straight after a delete never returns a deleted point, even while
    /// its HNSW removal is still queued. Run through this layer's own durable disk config.
    #[tokio::test(flavor = "current_thread")]
    async fn a_search_after_a_delete_never_returns_the_deleted_point() {
        let dir = std::env::temp_dir().join(format!("kg-text-del-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = TextStore::open(&dir, "m", 4).await.unwrap();
        let shas: Vec<Sha256> = (0..32u8)
            .map(|i| metrale_kg_derive::hash::sha256(&[i]))
            .collect();
        let v = shas
            .iter()
            .enumerate()
            .map(|(i, h)| (*h, vec![1.0, i as f32 * 0.01, 0.5, 0.25]))
            .collect();
        s.put(v).await.unwrap();
        s.engine.flush_pending().unwrap();
        let gone: Vec<u64> = shas.iter().step_by(2).map(point_of).collect();
        s.engine.delete_points_async(&gone).await.unwrap();
        for _ in 0..50 {
            let q = lattice_core::SearchQuery::new(vec![1.0, 0.0, 0.5, 0.25], 33).with_ef(64);
            for r in s.engine.search(q).unwrap() {
                assert!(
                    !gone.contains(&r.id),
                    "search returned deleted point {}",
                    r.id
                );
                assert!(
                    s.engine.get_point(r.id).unwrap().is_some(),
                    "point {} not in the store",
                    r.id
                );
            }
        }
        assert_eq!(s.cached().unwrap().len(), 16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_sha_round_trips_through_its_hex_form() {
        let s = metrale_kg_derive::hash::sha256(b"abc");
        assert_eq!(parse_hex(&s.hex()), Some(s));
        assert_eq!(parse_hex("zz"), None);
    }
}
