// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `met circuit precision --checkpoint <id|dir> [--node <glob>]`: the I/O side of the
//! "what we need" view. It resolves the checkpoint as `met circuit plan` does (a directory, the
//! local Hugging Face cache, or with `--allow-network` huggingface.co), plans it on the device's
//! class from the working tree, and prints each matching node's required pipeline
//! (`metrale_circuit::pipeline::query`), which the plan has already checked against the
//! kernels' declarations.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here derives a requirement or matches a kernel; the crate does.
//! - A glob that matches no planned node is an error, never an empty listing.

use std::path::Path;

use anyhow::{Result, bail};
use metrale_circuit::hardware::{self, CircuitSource, ModelSpec};
use metrale_circuit::venn::Run;

use super::CircuitPrecisionArgs;
use super::circuit_hw::{FsTree, checkpoint_texts, mode_of, precision_of, registry, source};
use super::circuit_venn::find_root;

/// 2026-10-02: The listing `a` asks for, from the repository at `root`.
pub(crate) fn listing(root: &Path, a: &CircuitPrecisionArgs) -> Result<String> {
    let tree = FsTree::new(root.to_path_buf());
    let reg = registry(&tree)?;
    let texts = checkpoint_texts(&a.checkpoint, a.allow_network)?;
    let model = source(&tree).model(&ModelSpec {
        checkpoint: &texts.id,
        config_json: texts.config.as_deref(),
        hf_quant: texts.hf_quant.as_deref(),
        precision: precision_of(a.precision),
    })?;
    let mode = mode_of(a.mode);
    let rows = match (a.rows, mode) {
        (Some(0), _) => bail!("--rows must be at least 1"),
        (Some(r), _) => r,
        (None, metrale_circuit::Mode::Decode | metrale_circuit::Mode::Draft) => 1,
        (None, _) => bail!("--mode {} needs --rows", mode.name()),
    };
    let one = hardware::plan_one(&reg, &a.hardware, &tree, &model, Run { mode, rows })?;
    metrale_circuit::pipeline::query::precision_text(&model, &one, a.node.as_deref())
        .map_err(anyhow::Error::msg)
}

/// 2026-10-02: Run `met circuit precision`.
pub(crate) fn run(a: CircuitPrecisionArgs) -> Result<()> {
    let root = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let text = listing(&root, &a)?;
    use std::io::Write;
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

#[cfg(test)]
#[path = "circuit_precision_tests.rs"]
mod circuit_precision_tests;
