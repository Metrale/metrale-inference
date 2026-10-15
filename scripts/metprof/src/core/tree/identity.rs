// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: kernel identity and per-launch time distribution (pure).
//!
//! A kernel's stable identity is its demangled name plus its launch
//! configuration as CUPTI records it: grid, block, registers per thread and
//! static and dynamic shared memory. `config_sha256` is the SHA-256 of the
//! configuration's canonical text ([`Identity::config_text`]); `kernel_id` is
//! the SHA-256 of the demangled name, a newline and that text. A leaf (and an
//! unattributed name) lists one variant per identity it saw.
//!
//! Quantiles are per launch, over every launch under a node, by linear
//! interpolation between closest ranks (`stats::quantile_sorted`).

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::core::model::{Grid, Kernel, Launch};
use crate::core::provenance::sha256_hex;
use crate::core::stats::quantile_sorted;
use crate::core::tree::us;

/// The quantile keys of a node, a variant and an unattributed name, with
/// their probabilities, in output order.
pub const QUANTILE_KEYS: [(&str, f64); 3] = [("p10_us", 0.1), ("median_us", 0.5), ("p90_us", 0.9)];

/// p10, median and p90 of the launch durations, in us.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantiles(pub [f64; 3]);

/// The quantiles of `durs` (ns), `None` when there are none.
pub fn quantiles(durs: &[i64]) -> Option<Quantiles> {
    let mut v = durs.to_vec();
    v.sort_unstable();
    let mut out = [0.0; 3];
    for (o, (_, p)) in out.iter_mut().zip(QUANTILE_KEYS) {
        *o = quantile_sorted(&v, p)? / 1000.0;
    }
    Some(Quantiles(out))
}

/// The quantile keys of `q`, as JSON (`null` when absent).
pub fn quantiles_json(q: Option<Quantiles>, m: &mut Map<String, Value>) {
    for (i, (k, _)) in QUANTILE_KEYS.iter().enumerate() {
        m.insert((*k).into(), q.map_or(Value::Null, |q| json!(q.0[i])));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identity {
    pub demangled: String,
    pub grid: Grid,
    pub launch: Launch,
}

impl Identity {
    pub fn of(k: &Kernel) -> Identity {
        Identity {
            demangled: k.demangled.clone(),
            grid: k.grid,
            launch: k.launch,
        }
    }

    pub fn config_text(&self) -> String {
        let (g, b, l) = (self.grid, self.launch.block, &self.launch);
        format!(
            "grid={},{},{};block={},{},{};registers_per_thread={};static_shared_bytes={};dynamic_shared_bytes={}",
            g.0,
            g.1,
            g.2,
            b.0,
            b.1,
            b.2,
            l.registers_per_thread,
            l.static_shared_bytes,
            l.dynamic_shared_bytes
        )
    }

    pub fn config_sha256(&self) -> String {
        sha256_hex(self.config_text().as_bytes())
    }

    pub fn kernel_id(&self) -> String {
        sha256_hex(format!("{}\n{}", self.demangled, self.config_text()).as_bytes())
    }
}

/// The launches of one identity. `durs` (ns) is filled while a tree is built
/// and consumed by `Tree::seal`, which computes `quantiles` from it; a tree
/// read back from JSON has no durations and its quantiles as stated.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Variant {
    pub n: u64,
    pub sum: i64,
    pub durs: Vec<i64>,
    pub quantiles: Option<Quantiles>,
}

pub type Variants = IndexMap<Identity, Variant>;

pub fn record(variants: &mut Variants, k: &Kernel) {
    let v = variants.entry(Identity::of(k)).or_default();
    v.n += 1;
    v.sum += k.dur();
    v.durs.push(k.dur());
}

pub fn absorb(into: &mut Variants, from: Variants) {
    for (id, o) in from {
        let v = into.entry(id).or_default();
        v.n += o.n;
        v.sum += o.sum;
        v.durs.extend(o.durs);
    }
}

/// Fills each variant's quantiles from its durations, which it consumes;
/// returns them all.
pub fn seal(variants: &mut Variants) -> Vec<i64> {
    let mut all = Vec::new();
    for v in variants.values_mut() {
        let durs = std::mem::take(&mut v.durs);
        v.quantiles = quantiles(&durs);
        all.extend(durs);
    }
    all
}

pub fn variants_json(variants: &Variants) -> Value {
    let items = variants.iter().map(|(id, v)| {
        let (g, b) = (id.grid, id.launch.block);
        let mut m = Map::new();
        m.insert("demangled".into(), json!(id.demangled));
        m.insert("grid".into(), json!([g.0, g.1, g.2]));
        m.insert("block".into(), json!([b.0, b.1, b.2]));
        m.insert(
            "registers_per_thread".into(),
            json!(id.launch.registers_per_thread),
        );
        m.insert(
            "static_shared_bytes".into(),
            json!(id.launch.static_shared_bytes),
        );
        m.insert(
            "dynamic_shared_bytes".into(),
            json!(id.launch.dynamic_shared_bytes),
        );
        m.insert("config_sha256".into(), json!(id.config_sha256()));
        m.insert("kernel_id".into(), json!(id.kernel_id()));
        m.insert("launches".into(), json!(v.n));
        m.insert("total_time".into(), json!(us(v.sum)));
        quantiles_json(v.quantiles, &mut m);
        Value::Object(m)
    });
    Value::Array(items.collect())
}
