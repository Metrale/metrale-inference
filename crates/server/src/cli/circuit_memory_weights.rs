// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: File reads behind `met circuit memory`: the kernel target's MODEL.toml
//! `[behavior]` defaults, and the bytes of the checkpoint tensors no weight node of the circuit
//! binds (norm, conv and gate parameters, a vision tower), from the safetensors headers when the
//! weights are local.
//!
//! Owner: server CLI.
//! Invariants:
//! - A tensor belongs to the circuit when its module (the name without its last segment) is a
//!   weight node's binding (`*` matching one segment), lies under one, or holds one (a fused
//!   experts tensor). Every other tensor is outside, sized from its header offsets.
//! - Only headers are read (8 bytes plus the JSON), never tensor data.

use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result};
use metrale_circuit::ir::Circuit;
use metrale_circuit::venn::Repo;

use super::circuit_hw::{CheckpointTexts, FsTree};
use super::circuit_memory_serve::Behavior;

/// 2026-10-02: The `[behavior]` of `kernels/<class>/<target>/MODEL.toml`, or the engine defaults
/// (all zero) when the class has no such target.
pub(crate) fn behavior(tree: &FsTree, class: &str, target: &str) -> Result<Behavior> {
    let rel = format!("kernels/{class}/{target}/MODEL.toml");
    match tree.read(&rel) {
        Ok(text) => Behavior::parse(&text, rel),
        Err(_) => Ok(Behavior {
            source: format!("{rel} absent: engine defaults"),
            ..Behavior::default()
        }),
    }
}

/// 2026-10-02: `module` belongs to a weight node binding `pattern` (dot-separated, `*` one
/// segment): one is a segment prefix of the other, so the module is the binding, lies under it,
/// or holds it (a fused experts tensor under `<prefix>.experts`).
fn bound(pattern: &str, module: &str) -> bool {
    pattern
        .split('.')
        .zip(module.split('.'))
        .all(|(p, m)| p == "*" || p == m)
}

/// 2026-10-02: The bindings of `c`'s weight-reading and embedding nodes.
fn bindings(c: &Circuit) -> Vec<&str> {
    c.nodes
        .iter()
        .filter(|n| n.weight.is_some() || n.op == metrale_circuit::OpKind::Embed)
        .flat_map(|n| n.binding.iter().map(String::as_str))
        .collect()
}

/// 2026-10-02: `(name, bytes)` of every tensor in one safetensors file's header.
fn header_tensors(path: &Path) -> Result<Vec<(String, u64)>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut json = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut json)?;
    let h: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&json)?;
    Ok(h.into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .filter_map(|(k, v)| {
            let o = v.get("data_offsets")?.as_array()?;
            Some((k, o.get(1)?.as_u64()? - o.first()?.as_u64()?))
        })
        .collect())
}

/// 2026-10-02: Bytes of the checkpoint's tensors no weight node of `c` binds; `None` when the
/// checkpoint has no local safetensors.
pub(crate) fn outside_bytes(texts: &CheckpointTexts, c: &Circuit) -> Result<Option<u64>> {
    let Some(dir) = &texts.dir else {
        return Ok(None);
    };
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    if files.is_empty() {
        return Ok(None);
    }
    files.sort();
    let pats = bindings(c);
    let mut out = 0;
    for f in files {
        for (name, bytes) in header_tensors(&f)? {
            let module = name.rsplit_once('.').map_or(name.as_str(), |(m, _)| m);
            if !pats.iter().any(|p| bound(p, module)) {
                out += bytes;
            }
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
#[path = "circuit_memory_weights_tests.rs"]
mod circuit_memory_weights_tests;
