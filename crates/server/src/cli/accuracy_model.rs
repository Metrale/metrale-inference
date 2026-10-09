// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `met accuracy model`: the I/O side of `metrale_accuracy::model_check`. It reads two
//! logits files with their pins, prints the per-token summary and judges it against the limits
//! the caller states (each measured from a good arm by the model check's owner).
//!
//! Owner: server CLI.
//! Invariants:
//! - Logits of another corpus or reference are refused by the pins, never compared.
//! - Every limit is an explicit flag; there is no default limit.

use anyhow::{Context, Result};
use metrale_accuracy::model_check::{Pins, compare};
use serde::Deserialize;

use super::AccuracyModelArgs;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Side {
    corpus_sha256: String,
    reference_sha256: String,
    vocab: usize,
}

fn read(prefix: &std::path::Path) -> Result<(Pins, usize, Vec<f64>)> {
    let meta = prefix.with_extension("toml");
    let side: Side = toml::from_str(
        &std::fs::read_to_string(&meta).with_context(|| meta.display().to_string())?,
    )
    .with_context(|| meta.display().to_string())?;
    let data = prefix.with_extension("f32");
    let bytes = std::fs::read(&data).with_context(|| data.display().to_string())?;
    anyhow::ensure!(
        bytes.len() % 4 == 0,
        "{}: not a whole number of f32",
        data.display()
    );
    let logits = bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    Ok((
        Pins {
            corpus_sha256: side.corpus_sha256,
            reference_sha256: side.reference_sha256,
        },
        side.vocab,
        logits,
    ))
}

/// 2026-10-09: Run `met accuracy model`; the exit status.
pub(crate) fn run(a: &AccuracyModelArgs) -> Result<i32> {
    let (rp, rv, run) = read(&a.run)?;
    let (pp, pv, reference) = read(&a.reference)?;
    anyhow::ensure!(rv == pv, "vocab {rv} (run) and {pv} (reference) differ");
    let (_, s) = compare(&run, &reference, rv, &rp, &pp)?;
    println!(
        "tokens {} | mean KL {:.3e} max KL {:.3e} | top-1 {:.4} | max |dlogit| {:.3e}",
        s.tokens, s.mean_kl, s.max_kl, s.top1, s.max_dlogit
    );
    let mut failed = Vec::new();
    if s.mean_kl > a.max_mean_kl {
        failed.push(format!("mean KL {:.3e} > {:.3e}", s.mean_kl, a.max_mean_kl));
    }
    if s.top1 < a.min_top1 {
        failed.push(format!("top-1 {:.4} < {:.4}", s.top1, a.min_top1));
    }
    if s.max_dlogit > a.max_dlogit {
        failed.push(format!(
            "max |dlogit| {:.3e} > {:.3e}",
            s.max_dlogit, a.max_dlogit
        ));
    }
    for f in &failed {
        println!("FAIL {f}");
    }
    Ok(i32::from(!failed.is_empty()))
}
