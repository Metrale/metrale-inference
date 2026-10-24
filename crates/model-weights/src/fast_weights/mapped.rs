// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: Every tensor of a safetensors checkpoint, memory-mapped and borrowed by name,
//! for CPU readers that consume weights in their stored precision without staging them.
//!
//! Owner: model-weights.
//! Invariants:
//! - A name maps to exactly one tensor across all shards; a duplicate is an error at open.
//! - The borrowed bytes are the header's `data_offsets` span, already bounds-checked against
//!   the file length by the header parser.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;

use super::header::{TensorMeta, parse_header, resolve_shards};

/// 2026-10-19: One tensor as stored: the header's dtype string, its shape and its bytes.
pub struct MappedTensor<'a> {
    pub stored_dtype: &'a str,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

pub struct MappedSafetensors {
    maps: Vec<Mmap>,
    tensors: HashMap<String, (usize, TensorMeta)>,
}

impl MappedSafetensors {
    /// 2026-10-19: Map every shard of the checkpoint in `model_dir` (the index's shards, or
    /// the shard files found there) and index their tensors by name.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let (files, _) = resolve_shards(model_dir)?;
        let mut maps = Vec::with_capacity(files.len());
        let mut tensors = HashMap::new();
        for (shard, path) in files.iter().enumerate() {
            let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            for meta in parse_header(&mut file).with_context(|| path.display().to_string())? {
                let name = meta.name.clone();
                ensure!(
                    tensors.insert(name.clone(), (shard, meta)).is_none(),
                    "tensor {name} appears in more than one shard"
                );
            }
            // SAFETY: a read-only mapping of a checkpoint shard; the files are not written
            // while the loader holds them, and every span read from it was bounds-checked by
            // the header parser.
            let map =
                unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
            maps.push(map);
        }
        Ok(Self { maps, tensors })
    }

    /// 2026-10-19: The tensor `name`, or `None` when no shard holds it.
    pub fn get(&self, name: &str) -> Option<MappedTensor<'_>> {
        let (shard, m) = self.tensors.get(name)?;
        let start = usize::try_from(m.abs_offset).expect("a span inside a mapped file fits usize");
        Some(MappedTensor {
            stored_dtype: &m.stored_dtype,
            shape: &m.shape,
            bytes: &self.maps[*shard][start..start + m.len],
        })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

#[cfg(test)]
#[path = "mapped_tests.rs"]
mod tests;
