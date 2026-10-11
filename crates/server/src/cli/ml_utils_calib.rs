// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `met ml-utils value-stats` and `met ml-utils calibrate-routing`: the two inputs a
//! value-faithful, histogram-routed mock needs beyond its spec. `value-stats` reads sampled
//! byte ranges of a checkpoint and writes the class bit-pattern statistics; `calibrate-routing`
//! reads the expert loads a mock produced and writes per-layer router gains.
//!
//! Owner: server CLI.
//! Invariants:
//! - Outputs are written to new files only (an existing file is refused), and their digests
//!   are printed; a spec names them, and the mock's resolved spec records the digests.
//! - `value-stats` reads only the ranges `metrale_ml_utils::stats::plan_reads` names.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_ml_utils::calibrate::{calibrate, gains};
use metrale_ml_utils::routing::RoutingCalibration;
use metrale_ml_utils::{RoutingProfile, read_source};

use super::ml_utils::{plan_of, read_spec};
use super::ml_utils_io::open_source;
use super::{CalibrateRoutingArgs, ValueStatsArgs};

fn write_new(path: &Path, text: &str) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {} (it must not exist)", path.display()))?;
    f.write_all(text.as_bytes())
        .with_context(|| format!("writing {}", path.display()))
}

/// 2026-10-04: `met ml-utils value-stats`.
pub(crate) fn value_stats(a: ValueStatsArgs) -> Result<()> {
    let src = open_source(&a.checkpoint, None, a.allow_network)?;
    let texts = read_source(src.source.as_ref()).map_err(|e| anyhow::anyhow!("{}: {e}", src.id))?;
    let schedule = metrale_circuit::layer_schedule(&texts.config)
        .map_err(|e| anyhow::anyhow!("{}: {e}", src.id))?;
    let (stats, text) =
        metrale_ml_utils::stats::collect(src.source.as_ref(), &schedule, &texts.index, &src.id)
            .map_err(|e| anyhow::anyhow!("{}: {e}", src.id))?;
    write_new(&a.out, &text)?;
    for (class, c) in &stats.classes {
        let n: u64 = c.counts.values().sum();
        let rms = c
            .bf16_rms()
            .map_or(String::new(), |r| format!(", rms {r:.4e}"));
        println!("  {class}: {n} elements, {} patterns{rms}", c.counts.len());
    }
    eprintln!(
        "value-stats: {} classes -> {} (sha256 {})",
        stats.classes.len(),
        a.out.display(),
        stats.digest
    );
    Ok(())
}

/// 2026-10-04: `met ml-utils calibrate-routing`.
pub(crate) fn calibrate_routing(a: CalibrateRoutingArgs) -> Result<()> {
    let files = read_spec(&a.spec)?;
    let Some(profile) = files.profile.as_ref() else {
        bail!(
            "{}: calibration needs a histogram-routed spec",
            a.spec.display()
        );
    };
    let src = open_source(&a.checkpoint, None, a.allow_network)?;
    let plan = plan_of(&src, &files)?;
    let text = std::fs::read_to_string(&a.measured)
        .with_context(|| format!("reading {}", a.measured.display()))?;
    let recorded = RoutingProfile::parse(&text)
        .map_err(|e| anyhow::anyhow!("{}: {e}", a.measured.display()))?;
    let layers = calibrate(&plan, profile, &recorded).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "mock  source  TV(recorded)  lambda  TV(model)  gain before -> after  never picked (mock/profile)"
    );
    for l in &layers {
        println!(
            "{:4}  {:6}  {:12.4}  {:6.3}  {:9.4}  {:11.4} -> {:<6.4}  {}/{}",
            l.mock_layer,
            l.source_layer,
            l.tv_recorded,
            l.lambda,
            l.tv_model,
            l.gain_before,
            l.gain,
            l.never_picked.0,
            l.never_picked.1
        );
    }
    let out = RoutingCalibration::to_text(&gains(&layers));
    write_new(&a.out, &out)?;
    let digest = RoutingCalibration::parse(&out)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .digest;
    eprintln!(
        "calibrate-routing: {} layers -> {} (sha256 {digest})",
        layers.len(),
        a.out.display()
    );
    Ok(())
}
