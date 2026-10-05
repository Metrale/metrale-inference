// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `met circuit lkb`: the Latent Kernel Blueprint of a checkpoint on a device's
//! class (generators and relations used, LKB coverage and its measured part, the LKB residual),
//! as Markdown or as campaign-ledger TOML. Definitions: book/src/architecture/lkb.md.
//!
//! Owner: server CLI.
//! Invariants:
//! - It builds the same report `met circuit plan --format report` builds
//!   ([`metrale_circuit::hardware::build_report`]) and reads its numbers from it; nothing is
//!   planned or estimated twice.

use anyhow::Result;
use metrale_circuit::hardware::{self, CircuitSource, ModelSpec};

use super::circuit_hw::{FsTree, checkpoint_texts, precision_of, registry, source};
use super::circuit_venn::find_root;

#[path = "circuit_lkb_args.rs"]
mod args;

pub use args::{CircuitLkbArgs, CircuitLkbFormat};

/// 2026-10-05: The text `met circuit lkb` prints for `a` on the tree at `root`.
pub(crate) fn lkb_text(a: &CircuitLkbArgs, root: &std::path::Path) -> Result<String> {
    let tree = FsTree::new(root.to_path_buf());
    let reg = registry(&tree)?;
    let texts = checkpoint_texts(&a.checkpoint, a.allow_network)?;
    let precision = precision_of(a.precision);
    let model = source(&tree).model(&ModelSpec {
        checkpoint: &texts.id,
        config_json: texts.config.as_deref(),
        hf_quant: texts.hf_quant.as_deref(),
        precision,
    })?;
    let command = format!(
        "met circuit lkb --checkpoint {} --hardware {} --precision {}",
        a.checkpoint,
        a.hardware,
        precision.name()
    );
    let report = hardware::build_report(&reg, &a.hardware, &tree, model, command.clone())?;
    let lkb = metrale_circuit::lkb::lkb(&report, &tree, command)?;
    Ok(match a.format {
        CircuitLkbFormat::Report => metrale_circuit::lkb::render_markdown(&lkb),
        CircuitLkbFormat::Toml => metrale_circuit::lkb::render_toml(&lkb),
    })
}

/// 2026-10-05: Run `met circuit lkb`.
pub(crate) fn run(a: CircuitLkbArgs) -> Result<()> {
    let root = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let text = lkb_text(&a, &root)?;
    super::circuit_hw::print(&text)
}

#[cfg(test)]
#[path = "circuit_lkb_tests.rs"]
mod circuit_lkb_tests;
