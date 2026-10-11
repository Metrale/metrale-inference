// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Value statistics: per tensor class, the empirical distribution of a checkpoint's
//! stored element bit patterns (E2M1 nibbles, FP8 bytes, BF16 halves, F32 words), and sampling
//! from it. A mock whose bytes follow these distributions toggles the same bits as the real
//! model in every GEMM and GEMV, which is what GPU power, and so J/token, depends on (the GB10
//! MMA power scales with the operand bit patterns).
//!
//! A class is a tensor name with the layer prefix written `L` and every all-digit segment `*`,
//! plus its dtype: `L.mlp.experts.*.down_proj.weight|F8_E4M3`.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Pure: [`plan_reads`] says which byte ranges to read, [`collect`] reads them through
//!   `CheckpointSource::read_range`, [`Accumulator`] counts them.
//! - Sampling uses only integer arithmetic (a cumulative table and a binary search), so a
//!   sampled tensor's bytes are the same on every platform and thread count.
//! - A class the statistics do not hold is refused, never approximated by another class.

use std::collections::BTreeMap;

use metrale_circuit::LayerSchedule;
use serde_json::{Value, json};

use crate::error::{MlError, Result};
use crate::index::{Dtype, TensorIndex, hex};
use crate::io::CheckpointSource;
use crate::rng::Stream;

/// 2026-10-04: The element a class counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 2026-10-04: Packed 4-bit codes, two per byte (low nibble first).
    Nibbles,
    /// 2026-10-04: One-byte elements (FP8).
    Bytes,
    /// 2026-10-04: Two-byte elements (BF16, F16).
    Halves,
    /// 2026-10-04: Four-byte elements (F32).
    Words,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Nibbles => "nibbles",
            Kind::Bytes => "bytes",
            Kind::Halves => "halves",
            Kind::Words => "words",
        }
    }

    fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "nibbles" => Kind::Nibbles,
            "bytes" => Kind::Bytes,
            "halves" => Kind::Halves,
            "words" => Kind::Words,
            _ => return None,
        })
    }

    /// 2026-10-04: The kind a dtype is counted as; `None` for integer index tensors.
    pub fn of(d: Dtype) -> Option<Kind> {
        match d {
            Dtype::U8 | Dtype::I8 => Some(Kind::Nibbles),
            Dtype::F8E4m3 => Some(Kind::Bytes),
            Dtype::Bf16 | Dtype::F16 => Some(Kind::Halves),
            Dtype::F32 => Some(Kind::Words),
            Dtype::I32 | Dtype::I64 => None,
        }
    }

    /// 2026-10-04: Bytes per counted element (a nibble counts per byte, two at a time).
    fn width(self) -> usize {
        match self {
            Kind::Nibbles | Kind::Bytes => 1,
            Kind::Halves => 2,
            Kind::Words => 4,
        }
    }
}

/// 2026-10-04: One class's distribution: bit pattern -> count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassStats {
    /// 2026-10-04: What a pattern is.
    pub kind: Kind,
    /// 2026-10-04: Pattern -> occurrences.
    pub counts: BTreeMap<u32, u64>,
}

impl ClassStats {
    /// 2026-10-04: Root mean square of the values a BF16 class holds; `None` for other kinds.
    pub fn bf16_rms(&self) -> Option<f32> {
        if self.kind != Kind::Halves {
            return None;
        }
        let (mut s, mut n) = (0.0f64, 0u64);
        for (&p, &c) in &self.counts {
            let v = f32::from_bits(p << 16) as f64;
            if v.is_finite() {
                s += v * v * c as f64;
                n += c;
            }
        }
        (n > 0).then(|| (s / n as f64).sqrt() as f32)
    }
}

/// 2026-10-04: A checkpoint's value statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueStats {
    /// 2026-10-04: The checkpoint they were read from.
    pub source: String,
    /// 2026-10-04: Class -> distribution.
    pub classes: BTreeMap<String, ClassStats>,
    /// 2026-10-04: sha256 of the serialized text.
    pub digest: String,
}

/// 2026-10-04: The class of a tensor.
pub fn class_key(s: &LayerSchedule, name: &str, dtype: Dtype) -> String {
    let rel = match s.layer_of(name) {
        Some(l) => format!("L{}", &name[s.module_of(l).len()..]),
        None => name.to_string(),
    };
    let starred: Vec<&str> = rel
        .split('.')
        .map(|seg| {
            if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
                "*"
            } else {
                seg
            }
        })
        .collect();
    format!("{}|{}", starred.join("."), dtype.name())
}

/// 2026-10-04: One byte range to read for the statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    /// 2026-10-04: The class it counts toward.
    pub class: String,
    /// 2026-10-04: What it holds.
    pub kind: Kind,
    /// 2026-10-04: The shard file.
    pub shard: String,
    /// 2026-10-04: Absolute offset.
    pub offset: u64,
    /// 2026-10-04: Bytes.
    pub len: u64,
}

/// 2026-10-04: Tensors sampled per class, spread over the class's tensors.
const TENSORS_PER_CLASS: usize = 6;
/// 2026-10-04: A tensor up to this size is read whole; a larger one in [`CHUNKS`] chunks.
const WHOLE_BYTES: u64 = 2 << 20;
/// 2026-10-04: Chunks read from a large tensor, evenly spaced.
const CHUNKS: u64 = 16;
/// 2026-10-04: Bytes per chunk.
const CHUNK_BYTES: u64 = 128 << 10;

/// 2026-10-04: The byte ranges the statistics of `index` read.
pub fn plan_reads(s: &LayerSchedule, index: &TensorIndex) -> Vec<Read> {
    let mut by_class: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for e in index.iter() {
        if Kind::of(e.dtype).is_some() {
            by_class
                .entry(class_key(s, &e.name, e.dtype))
                .or_default()
                .push(e);
        }
    }
    let mut out = Vec::new();
    for (class, list) in by_class {
        let n = list.len();
        let picks = TENSORS_PER_CLASS.min(n);
        for k in 0..picks {
            let e = list[k * n / picks];
            let kind = Kind::of(e.dtype).expect("filtered above");
            let bytes = e.bytes();
            let w = kind.width() as u64;
            if bytes <= WHOLE_BYTES {
                out.push(Read {
                    class: class.clone(),
                    kind,
                    shard: e.shard.clone(),
                    offset: e.offset,
                    len: bytes,
                });
                continue;
            }
            for c in 0..CHUNKS {
                let start = (bytes - CHUNK_BYTES) * c / (CHUNKS - 1) / w * w;
                out.push(Read {
                    class: class.clone(),
                    kind,
                    shard: e.shard.clone(),
                    offset: e.offset + start,
                    len: CHUNK_BYTES,
                });
            }
        }
    }
    out
}

/// 2026-10-04: The statistics of `src`: every [`plan_reads`] range read and counted.
pub fn collect(
    src: &dyn CheckpointSource,
    s: &LayerSchedule,
    index: &TensorIndex,
    source: &str,
) -> Result<(ValueStats, String)> {
    let mut acc = Accumulator::default();
    for r in plan_reads(s, index) {
        let bytes = src.read_range(&r.shard, r.offset, r.len)?;
        if bytes.len() as u64 != r.len {
            return Err(MlError::Checkpoint(format!(
                "{}: {} bytes at {} read as {}",
                r.shard,
                r.len,
                r.offset,
                bytes.len()
            )));
        }
        acc.add(&r, &bytes);
    }
    Ok(acc.finish(source))
}

/// 2026-10-04: Counts patterns as reads arrive.
#[derive(Debug, Default)]
pub struct Accumulator {
    classes: BTreeMap<String, ClassStats>,
}

impl Accumulator {
    /// 2026-10-04: Count the bytes of `read`.
    pub fn add(&mut self, read: &Read, bytes: &[u8]) {
        let c = self
            .classes
            .entry(read.class.clone())
            .or_insert_with(|| ClassStats {
                kind: read.kind,
                counts: BTreeMap::new(),
            });
        let mut bump = |p: u32| *c.counts.entry(p).or_insert(0) += 1;
        match read.kind {
            Kind::Nibbles => bytes.iter().for_each(|&b| {
                bump(u32::from(b & 0xF));
                bump(u32::from(b >> 4));
            }),
            Kind::Bytes => bytes.iter().for_each(|&b| bump(u32::from(b))),
            Kind::Halves => bytes
                .chunks_exact(2)
                .for_each(|h| bump(u32::from(u16::from_le_bytes([h[0], h[1]])))),
            Kind::Words => bytes
                .chunks_exact(4)
                .for_each(|w| bump(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))),
        }
    }

    /// 2026-10-04: The statistics, serialized and digested.
    pub fn finish(self, source: &str) -> (ValueStats, String) {
        let text = to_text(source, &self.classes);
        let stats = ValueStats {
            source: source.to_string(),
            classes: self.classes,
            digest: hex(&<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes())),
        };
        (stats, text)
    }
}

fn to_text(source: &str, classes: &BTreeMap<String, ClassStats>) -> String {
    let c: serde_json::Map<String, Value> = classes
        .iter()
        .map(|(k, v)| {
            let counts: Vec<[u64; 2]> = v.counts.iter().map(|(&p, &n)| [u64::from(p), n]).collect();
            (
                k.clone(),
                json!({ "kind": v.kind.name(), "counts": counts }),
            )
        })
        .collect();
    serde_json::to_string(&json!({ "schema": 1, "source": source, "classes": c }))
        .expect("a JSON value serializes")
        + "\n"
}

impl ValueStats {
    /// 2026-10-04: Parse the text [`Accumulator::finish`] writes.
    pub fn parse(text: &str) -> Result<Self> {
        let bad = |w: String| MlError::Spec(format!("value statistics: {w}"));
        let v: Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        if v["schema"] != 1 {
            return Err(bad(format!("schema {} (this build reads 1)", v["schema"])));
        }
        let mut classes = BTreeMap::new();
        for (k, c) in v["classes"]
            .as_object()
            .ok_or_else(|| bad("no classes".into()))?
        {
            let kind = c["kind"]
                .as_str()
                .and_then(Kind::parse)
                .ok_or_else(|| bad(format!("{k}: kind {}", c["kind"])))?;
            let mut counts = BTreeMap::new();
            for pc in c["counts"]
                .as_array()
                .ok_or_else(|| bad(format!("{k}: no counts")))?
            {
                let (p, n) = match pc.as_array().map(Vec::as_slice) {
                    Some([p, n]) => (p.as_u64(), n.as_u64()),
                    _ => (None, None),
                };
                match (p.and_then(|p| u32::try_from(p).ok()), n) {
                    (Some(p), Some(n)) if n > 0 => {
                        counts.insert(p, n);
                    }
                    _ => return Err(bad(format!("{k}: a malformed count {pc}"))),
                }
            }
            if counts.is_empty() {
                return Err(bad(format!("{k}: no counts")));
            }
            classes.insert(k.clone(), ClassStats { kind, counts });
        }
        Ok(ValueStats {
            source: v["source"].as_str().unwrap_or_default().to_string(),
            classes,
            digest: hex(&<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes())),
        })
    }

    /// 2026-10-04: The sampler of `class`, refused when the statistics do not hold it.
    pub fn sampler(&self, class: &str) -> Result<Sampler> {
        let c = self
            .classes
            .get(class)
            .ok_or_else(|| MlError::Spec(format!("value statistics have no class `{class}`")))?;
        let mut cum = Vec::with_capacity(c.counts.len());
        let mut patterns = Vec::with_capacity(c.counts.len());
        let mut total = 0u64;
        for (&p, &n) in &c.counts {
            total += n;
            cum.push(total);
            patterns.push(p);
        }
        Ok(Sampler {
            kind: c.kind,
            patterns,
            cum,
            total,
        })
    }
}

/// 2026-10-04: Draws patterns in proportion to their counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sampler {
    kind: Kind,
    patterns: Vec<u32>,
    cum: Vec<u64>,
    total: u64,
}

impl Sampler {
    #[inline]
    fn draw(&self, s: Stream, i: u64) -> u32 {
        let r = s.below(i, self.total);
        self.patterns[self.cum.partition_point(|&c| c <= r)]
    }

    /// 2026-10-04: The kind of element it draws.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// 2026-10-04: `n` bytes of sampled elements; byte `b`'s value depends only on `(s, b)`.
    pub fn bytes(&self, s: Stream, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        let per = n.div_ceil(threads).max(1 << 20).next_multiple_of(4);
        std::thread::scope(|sc| {
            for (k, part) in out.chunks_mut(per).enumerate() {
                sc.spawn(move || self.fill(s, k * per, part));
            }
        });
        out
    }

    fn fill(&self, s: Stream, start: usize, out: &mut [u8]) {
        match self.kind {
            Kind::Nibbles => out.iter_mut().enumerate().for_each(|(j, b)| {
                let i = 2 * (start + j) as u64;
                *b = (self.draw(s, i) | (self.draw(s, i + 1) << 4)) as u8;
            }),
            Kind::Bytes => out
                .iter_mut()
                .enumerate()
                .for_each(|(j, b)| *b = self.draw(s, (start + j) as u64) as u8),
            Kind::Halves => out.chunks_exact_mut(2).enumerate().for_each(|(j, h)| {
                let v = self.draw(s, (start / 2 + j) as u64) as u16;
                h.copy_from_slice(&v.to_le_bytes());
            }),
            Kind::Words => out.chunks_exact_mut(4).enumerate().for_each(|(j, w)| {
                let v = self.draw(s, (start / 4 + j) as u64);
                w.copy_from_slice(&v.to_le_bytes());
            }),
        }
    }
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod stats_tests;
