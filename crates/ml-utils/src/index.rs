// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A checkpoint's tensor index: every tensor's name, dtype and shape, read from the
//! safetensors headers alone (no tensor bytes), plus the shard each lives in.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A header is untrusted input: an unknown dtype, a non-integer shape, a reversed or
//!   overflowing `data_offsets` pair, or a byte length that does not match dtype × shape is
//!   refused, naming the tensor.
//! - A tensor name appears once across all shards; a duplicate is refused.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::{MlError, Result};

/// 2026-10-03: A safetensors element type this crate reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dtype {
    /// 2026-10-03: bfloat16.
    Bf16,
    /// 2026-10-03: IEEE half.
    F16,
    /// 2026-10-03: IEEE single.
    F32,
    /// 2026-10-03: FP8 E4M3 (OCP E4M3FN).
    F8E4m3,
    /// 2026-10-03: Raw bytes (packed NVFP4 pairs).
    U8,
    /// 2026-10-03: Signed bytes (packed NVFP4 in some exports).
    I8,
    /// 2026-10-03: 32-bit signed integers.
    I32,
    /// 2026-10-03: 64-bit signed integers.
    I64,
}

impl Dtype {
    /// 2026-10-03: The safetensors header spelling.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "BF16",
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::F8E4m3 => "F8_E4M3",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
        }
    }

    /// 2026-10-03: Parse the header spelling.
    pub fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "F8_E4M3" => Dtype::F8E4m3,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            _ => return None,
        })
    }

    /// 2026-10-03: Bytes per element.
    pub fn bytes(self) -> u64 {
        match self {
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F8E4m3 | Dtype::U8 | Dtype::I8 => 1,
            Dtype::I64 => 8,
        }
    }
}

/// 2026-10-03: One tensor of a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// 2026-10-03: The tensor's full name.
    pub name: String,
    /// 2026-10-03: Element type.
    pub dtype: Dtype,
    /// 2026-10-03: Shape; empty for a scalar.
    pub shape: Vec<u64>,
    /// 2026-10-03: The shard file it is stored in.
    pub shard: String,
    /// 2026-10-04: Absolute byte offset of its data in the shard (8 + header length +
    /// `data_offsets[0]`); 0 for an entry built without a header.
    pub offset: u64,
}

impl TensorEntry {
    /// 2026-10-03: Element count (1 for a scalar).
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// 2026-10-03: Stored bytes.
    pub fn bytes(&self) -> u64 {
        self.numel() * self.dtype.bytes()
    }
}

/// 2026-10-03: Every tensor of a checkpoint, by name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorIndex {
    entries: BTreeMap<String, TensorEntry>,
}

impl TensorIndex {
    /// 2026-10-03: The index of the shards whose header JSON (the bytes after the 8-byte length)
    /// is given, as `(shard file name, header bytes)`.
    pub fn from_headers(headers: &[(String, Vec<u8>)]) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for (shard, bytes) in headers {
            for e in parse_header(shard, bytes)? {
                if let Some(prev) = entries.get(&e.name) {
                    let prev: &TensorEntry = prev;
                    return Err(MlError::Index(format!(
                        "tensor `{}` is in both {} and {shard}",
                        e.name, prev.shard
                    )));
                }
                entries.insert(e.name.clone(), e);
            }
        }
        if entries.is_empty() {
            return Err(MlError::Index("the checkpoint holds no tensors".into()));
        }
        Ok(TensorIndex { entries })
    }

    /// 2026-10-03: An index of the given entries (a duplicate name is refused).
    pub fn from_entries(list: Vec<TensorEntry>) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for e in list {
            let name = e.name.clone();
            if entries.insert(name.clone(), e).is_some() {
                return Err(MlError::Index(format!("tensor `{name}` appears twice")));
            }
        }
        Ok(TensorIndex { entries })
    }

    /// 2026-10-03: The entry named `name`.
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }

    /// 2026-10-03: Every entry, by name.
    pub fn iter(&self) -> impl Iterator<Item = &TensorEntry> {
        self.entries.values()
    }

    /// 2026-10-03: The number of tensors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 2026-10-03: True when the index holds no tensor.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 2026-10-03: Total stored bytes.
    pub fn bytes(&self) -> u64 {
        self.entries.values().map(TensorEntry::bytes).sum()
    }

    /// 2026-10-03: A digest of the index (names, dtypes, shapes; not the shard layout), so two
    /// mocks of different checkpoints never share a spec digest.
    pub fn digest(&self) -> String {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        for e in self.entries.values() {
            h.update(e.name.as_bytes());
            h.update([0]);
            h.update(e.dtype.name().as_bytes());
            for d in &e.shape {
                h.update(d.to_le_bytes());
            }
            h.update([0xFF]);
        }
        hex(&h.finalize())
    }
}

/// 2026-10-03: Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// 2026-10-03: The tensors one shard's header JSON declares. `__metadata__` is skipped.
pub fn parse_header(shard: &str, bytes: &[u8]) -> Result<Vec<TensorEntry>> {
    let bad = |what: String| MlError::Index(format!("{shard}: {what}"));
    let v: Value = serde_json::from_slice(bytes).map_err(|e| bad(e.to_string()))?;
    let obj = v
        .as_object()
        .ok_or_else(|| bad("the header is not an object".into()))?;
    let mut out = Vec::with_capacity(obj.len());
    let data_start = 8 + bytes.len() as u64;
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        let tbad = |what: &str| bad(format!("tensor `{name}`: {what}"));
        let dtype_s = t
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| tbad("no dtype"))?;
        let dtype = Dtype::parse(dtype_s).ok_or_else(|| tbad(&format!("dtype {dtype_s}")))?;
        let shape = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| tbad("no shape"))?
            .iter()
            .map(|d| d.as_u64().ok_or_else(|| tbad("a non-integer dimension")))
            .collect::<Result<Vec<u64>>>()?;
        let offs = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| tbad("data_offsets is not a pair"))?;
        let (start, end) = match (offs[0].as_u64(), offs[1].as_u64()) {
            (Some(s), Some(e)) if e >= s => (s, e),
            _ => return Err(tbad("data_offsets is reversed or not integers")),
        };
        let numel = shape
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d))
            .and_then(|n| n.checked_mul(dtype.bytes()))
            .ok_or_else(|| tbad("the shape overflows"))?;
        if end - start != numel {
            return Err(tbad(&format!(
                "{} bytes for {dtype_s} {shape:?} ({numel} expected)",
                end - start
            )));
        }
        out.push(TensorEntry {
            name: name.clone(),
            dtype,
            shape,
            shard: shard.to_string(),
            offset: data_start
                .checked_add(start)
                .ok_or_else(|| tbad("the offset overflows"))?,
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "index_tests.rs"]
mod index_tests;
