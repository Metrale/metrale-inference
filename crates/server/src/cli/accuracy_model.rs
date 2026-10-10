// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `met accuracy model`: the I/O side of `metrale_accuracy::model_logprobs`. It reads
//! the pinned reference dump and the dump under test, and prints the exact or numerics verdict
//! per leg. The dumps come from an engine-neutral producer over the OpenAI-compatible API, so
//! one judge serves this engine and any comparison engine.
//!
//! Owner: server CLI.
//! Invariants:
//! - A reference whose bytes are not the pinned SHA-256, or dumps of different corpora, are
//!   refused, never compared.
//! - Numerics mode needs every limit on the command line; there is no default limit.

use anyhow::{Context, Result, bail};
use metrale_accuracy::model_logprobs::{LEGS, Limits, exact, judge, leg_metrics, load};

use super::AccuracyModelArgs;

fn limits(a: &AccuracyModelArgs) -> Result<Limits> {
    let need = |v: Option<f64>, flag: &str| {
        v.with_context(|| {
            format!("numerics mode needs --{flag} (set from measured good and bad arms)")
        })
    };
    Ok(Limits {
        tf_min_top1: need(a.tf_min_top1, "tf-min-top1")?,
        tf_max_kl: need(a.tf_max_kl, "tf-max-kl")?,
        tf_max_dlp_p99: need(a.tf_max_dlp_p99, "tf-max-dlp-p99")?,
        dec_max_kl: need(a.dec_max_kl, "dec-max-kl")?,
        dec_max_dlp_p99: need(a.dec_max_dlp_p99, "dec-max-dlp-p99")?,
        max_divergence_margin: need(a.max_divergence_margin, "max-divergence-margin")?,
        max_unmeasured_divergences: need(
            a.max_unmeasured_divergences,
            "max-unmeasured-divergences",
        )?,
    })
}

/// 2026-10-09: Run `met accuracy model`; the exit status.
pub(crate) fn run(a: &AccuracyModelArgs) -> Result<i32> {
    let reference =
        std::fs::read(&a.reference).with_context(|| a.reference.display().to_string())?;
    let test = std::fs::read(&a.test).with_context(|| a.test.display().to_string())?;
    let (r, t) = load(&reference, &a.reference_sha256, &test)?;
    match a.mode.as_str() {
        "exact" => {
            let diffs = exact(&r, &t);
            for (leg, i, at) in &diffs {
                println!("  {leg}[{i}] differs (first token/logprob difference at {at:?})");
            }
            println!("EXACT {}", if diffs.is_empty() { "PASS" } else { "FAIL" });
            Ok(i32::from(!diffs.is_empty()))
        }
        "numerics" => {
            let l = limits(a)?;
            let mut ok = true;
            for leg in LEGS {
                let m = leg_metrics(&r, &t, leg);
                let pass = judge(leg, &m, &l);
                ok &= pass;
                println!(
                    "{leg:5} {} positions {} top1 {:.5} kl_mean {:.5} dlp_p99 {:.5} dlp_max {:.5} diverged {} unmeasured {} max_margin {:?}",
                    if pass { "PASS" } else { "FAIL" },
                    m.positions,
                    m.top1,
                    m.kl_mean,
                    m.dlp_p99,
                    m.dlp_max,
                    m.diverged,
                    m.unmeasured_divergences,
                    m.max_margin_at_divergence
                );
            }
            println!("NUMERICS {}", if ok { "PASS" } else { "FAIL" });
            Ok(i32::from(!ok))
        }
        other => bail!("--mode {other} (exact | numerics)"),
    }
}
