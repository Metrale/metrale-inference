// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Reading what a plan needs from a [`CheckpointSource`], and writing a planned mock
//! to a [`CheckpointSink`] as a Hugging Face checkpoint: config.json, the quantization sidecar,
//! the tokenizer and template files, the resolved spec, the shards and their index.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A unit's tensors never straddle two shards.
//! - Units are synthesized in parallel but appended in plan order, so the files do not depend
//!   on the thread count.
//! - Every shard's `__metadata__` carries `metrale_mock = <digest>`, so a mock is recognisable
//!   from any one of its shards.

use std::collections::BTreeMap;

use crate::error::{MlError, Result};
use crate::index::TensorIndex;
use crate::io::{CheckpointSink, CheckpointSource, is_aux_file};
use crate::plan::{MockPlan, synthesize};
use crate::st_format::{HeaderTensor, index_json, shard_file, shard_header, split_shards};

/// 2026-10-03: Most bytes per output shard.
pub const SHARD_BYTES: u64 = 4 << 30;
/// 2026-10-03: The resolved spec's file name in a mock checkpoint.
pub const RESOLVED_FILE: &str = "mock.resolved.toml";
/// 2026-10-03: The shard metadata key that names a mock's digest.
pub const MOCK_METADATA_KEY: &str = "metrale_mock";
/// 2026-10-03: Bytes of synthesized units held at once while writing.
const BATCH_BYTES: u64 = 1 << 30;

/// 2026-10-03: What a mock is planned from, as read from a source.
#[derive(Debug, Clone)]
pub struct SourceTexts {
    /// 2026-10-03: config.json.
    pub config: String,
    /// 2026-10-03: hf_quant_config.json, when present.
    pub hf_quant: Option<String>,
    /// 2026-10-03: The tensor index.
    pub index: TensorIndex,
}

fn text(src: &dyn CheckpointSource, rel: &str) -> Result<String> {
    String::from_utf8(src.read(rel)?).map_err(|e| MlError::Io(format!("{rel}: {e}")))
}

/// 2026-10-03: Read config.json, the sidecar (if listed) and every shard header.
pub fn read_source(src: &dyn CheckpointSource) -> Result<SourceTexts> {
    let files = src.files()?;
    let hf_quant = if files.iter().any(|f| f == "hf_quant_config.json") {
        Some(text(src, "hf_quant_config.json")?)
    } else {
        None
    };
    Ok(SourceTexts {
        config: text(src, "config.json")?,
        hf_quant,
        index: TensorIndex::from_headers(&src.shard_headers()?)?,
    })
}

/// 2026-10-03: What was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteReport {
    /// 2026-10-03: Shard files.
    pub shards: usize,
    /// 2026-10-03: Tensor bytes.
    pub bytes: u64,
    /// 2026-10-03: Files copied unchanged from the source.
    pub copied: Vec<String>,
}

/// 2026-10-03: Write `plan` as a checkpoint, copying the aux files from `src`, synthesizing on up
/// to `threads` threads.
pub fn write_mock(
    plan: &MockPlan,
    src: &dyn CheckpointSource,
    sink: &mut dyn CheckpointSink,
    threads: usize,
) -> Result<WriteReport> {
    emit(plan, src, sink, Some(threads))
}

/// 2026-10-03: Write `plan`'s *skeleton*: every file of [`write_mock`] with the same shard
/// headers, but no tensor data (the sink skips it). Every reader of config, metadata and headers
/// sees the mock; the `--mock` loader synthesizes the weights instead of reading them.
pub fn write_skeleton(
    plan: &MockPlan,
    src: &dyn CheckpointSource,
    sink: &mut dyn CheckpointSink,
) -> Result<WriteReport> {
    emit(plan, src, sink, None)
}

/// 2026-10-03: The shared writer: `threads` = `None` writes a skeleton.
fn emit(
    plan: &MockPlan,
    src: &dyn CheckpointSource,
    sink: &mut dyn CheckpointSink,
    threads: Option<usize>,
) -> Result<WriteReport> {
    sink.write_file("config.json", plan.config_json.as_bytes())?;
    if let Some(h) = &plan.hf_quant_config {
        sink.write_file("hf_quant_config.json", h.as_bytes())?;
    }
    let mut copied = Vec::new();
    for f in src.files()? {
        if is_aux_file(&f) {
            sink.write_file(&f, &src.read(&f)?)?;
            copied.push(f);
        }
    }
    sink.write_file(RESOLVED_FILE, plan.resolved.as_bytes())?;

    let unit_tensors: Vec<Vec<usize>> = plan.units.iter().map(unit_tensor_ids).collect();
    let unit_bytes: Vec<u64> = unit_tensors
        .iter()
        .map(|ts| ts.iter().map(|&t| plan.tensors[t].bytes()).sum())
        .collect();
    let shards = split_shards(&unit_bytes, SHARD_BYTES);
    let meta = BTreeMap::from([
        ("format".to_string(), "pt".to_string()),
        (MOCK_METADATA_KEY.to_string(), plan.digest.clone()),
    ]);
    let mut weight_map = BTreeMap::new();
    for (si, units) in shards.iter().enumerate() {
        let file = shard_file(si, shards.len());
        let header: Vec<HeaderTensor<'_>> = units
            .iter()
            .flat_map(|&u| unit_tensors[u].iter())
            .map(|&t| {
                let o = &plan.tensors[t];
                weight_map.insert(o.name.clone(), file.clone());
                HeaderTensor {
                    name: &o.name,
                    dtype: o.dtype,
                    shape: &o.shape,
                }
            })
            .collect();
        sink.begin_shard(&file, &shard_header(&header, &meta))?;
        match threads {
            Some(threads) => {
                for batch in batches(plan, units, threads) {
                    for unit_out in synthesize_batch(plan, &batch)? {
                        for (_, bytes) in unit_out {
                            sink.append(&bytes)?;
                        }
                    }
                }
            }
            None => sink.skip(units.iter().map(|&u| unit_bytes[u]).sum())?,
        }
        sink.end_shard()?;
    }
    sink.write_file(
        "model.safetensors.index.json",
        index_json(&weight_map, plan.bytes()).as_bytes(),
    )?;
    Ok(WriteReport {
        shards: shards.len(),
        bytes: plan.bytes(),
        copied,
    })
}

/// 2026-10-03: The tensor indices of a unit, in its byte order.
pub fn unit_tensor_ids(u: &crate::plan::Unit) -> Vec<usize> {
    match u {
        crate::plan::Unit::Plain { tensor, .. } => vec![*tensor],
        crate::plan::Unit::Group { tensors, .. } => tensors.clone(),
    }
}

/// 2026-10-03: Consecutive runs of `units` of at most `threads` units and about
/// [`BATCH_BYTES`] bytes (a larger unit is a run of its own): the batches [`synthesize_batch`]
/// takes, for the writer and the `--mock` loader alike.
pub fn batches(plan: &MockPlan, units: &[usize], threads: usize) -> Vec<Vec<usize>> {
    let threads = threads.max(1);
    let bytes = |u: usize| -> u64 {
        unit_tensor_ids(&plan.units[u])
            .iter()
            .map(|&t| plan.tensors[t].bytes())
            .sum()
    };
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_bytes = 0;
    for &u in units {
        let b = bytes(u);
        if !cur.is_empty() && (cur.len() == threads || cur_bytes + b > BATCH_BYTES) {
            out.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur.push(u);
        cur_bytes += b;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// 2026-10-03: Synthesize `batch` concurrently; results in `batch` order.
pub fn synthesize_batch(plan: &MockPlan, batch: &[usize]) -> Result<Vec<Vec<(usize, Vec<u8>)>>> {
    if batch.len() == 1 {
        return Ok(vec![synthesize(plan, batch[0])?]);
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = batch
            .iter()
            .map(|&u| s.spawn(move || synthesize(plan, u)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a synthesis thread panicked"))
            .collect()
    })
}

#[cfg(test)]
#[path = "write_tests.rs"]
mod write_tests;
