// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `met ml-utils`: model utilities over the pure planner `metrale-ml-utils`.
//! `inspect` reports a checkpoint from its metadata and headers, `mockify` writes a mock
//! (rehearsal) checkpoint, `extrapolate` estimates the full model from mock measurements;
//! `value-stats` and `calibrate-routing` live in `ml_utils_calib`.
//!
//! Owner: server CLI.
//! Invariants:
//! - No tensor data is read here: only config, quantization metadata, aux files and headers.
//! - Every output that depends on a spec writes the resolved spec beside it and prints its digest.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_ml_utils::extrapolate::{Scaling, extrapolate, points_of, units_of_resolved};
use metrale_ml_utils::routing::RoutingCalibration;
use metrale_ml_utils::spec::{RoutingMode, ValuesMode};
use metrale_ml_utils::stats::ValueStats;
use metrale_ml_utils::{MockInputs, MockPlan, MockSpec, RoutingProfile, plan_mock, read_source};

use super::ml_utils_io::{FsSink, OpenedSource, open_source};
use super::{ExtrapolateArgs, InspectArgs, MlUtilsAction, MlUtilsArgs, MockifyArgs};

/// 2026-10-03: Run one `met ml-utils` command.
pub(crate) fn dispatch(args: MlUtilsArgs) -> Result<()> {
    match args.action {
        MlUtilsAction::Inspect(a) => inspect(a),
        MlUtilsAction::Mockify(a) => mockify(a),
        MlUtilsAction::Extrapolate(a) => extrapolate_cmd(a),
        MlUtilsAction::ValueStats(a) => super::ml_utils_calib::value_stats(a),
        MlUtilsAction::CalibrateRouting(a) => super::ml_utils_calib::calibrate_routing(a),
    }
}

/// 2026-10-04: A spec and the files it names, each read relative to the spec's directory: the
/// routing profile and calibration (histogram routing) and the value statistics (`stats`).
#[derive(Debug)]
pub(crate) struct SpecFiles {
    pub spec: MockSpec,
    pub profile: Option<RoutingProfile>,
    pub calibration: Option<RoutingCalibration>,
    pub stats: Option<ValueStats>,
}

fn read_named(spec_path: &Path, rel: &str, what: &str) -> Result<(std::path::PathBuf, String)> {
    let full = spec_path.parent().unwrap_or(Path::new(".")).join(rel);
    let text = std::fs::read_to_string(&full)
        .with_context(|| format!("reading the {what} {}", full.display()))?;
    Ok((full, text))
}

/// 2026-10-03: Read a spec file and every file it names. Shared by `mockify`, `inspect --spec`,
/// `calibrate-routing` and `met serve --mock`.
pub(crate) fn read_spec(path: &Path) -> Result<SpecFiles> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let spec = MockSpec::parse(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let (mut profile, mut calibration, mut stats) = (None, None, None);
    if let RoutingMode::Histogram {
        path: p,
        calibration: c,
    } = &spec.routing
    {
        let (full, t) = read_named(path, p, "routing profile")?;
        profile = Some(
            RoutingProfile::parse(&t).map_err(|e| anyhow::anyhow!("{}: {e}", full.display()))?,
        );
        if let Some(c) = c {
            let (full, t) = read_named(path, c, "routing calibration")?;
            calibration = Some(
                RoutingCalibration::parse(&t)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", full.display()))?,
            );
        }
    }
    if let ValuesMode::Stats { path: p } = &spec.values {
        let (full, t) = read_named(path, p, "value statistics")?;
        stats =
            Some(ValueStats::parse(&t).map_err(|e| anyhow::anyhow!("{}: {e}", full.display()))?);
    }
    Ok(SpecFiles {
        spec,
        profile,
        calibration,
        stats,
    })
}

/// 2026-10-03: Plan the mock of `src` under `files`.
pub(crate) fn plan_of(src: &OpenedSource, files: &SpecFiles) -> Result<MockPlan> {
    let texts = read_source(src.source.as_ref()).map_err(|e| anyhow::anyhow!("{}: {e}", src.id))?;
    plan_mock(&MockInputs {
        source_id: &src.id,
        revision: src.revision.as_deref(),
        config_json: &texts.config,
        hf_quant_config: texts.hf_quant.as_deref(),
        index: &texts.index,
        spec: &files.spec,
        routing: files.profile.as_ref(),
        calibration: files.calibration.as_ref(),
        stats: files.stats.as_ref(),
    })
    .map_err(|e| anyhow::anyhow!("{}: {e}", src.id))
}

pub(crate) fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

fn inspect(a: InspectArgs) -> Result<()> {
    let src = open_source(&a.checkpoint, None, a.allow_network)?;
    let texts = read_source(src.source.as_ref()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let r =
        metrale_ml_utils::inspect::inspect(&texts.config, texts.hf_quant.as_deref(), &texts.index)
            .map_err(|e| anyhow::anyhow!("{}: {e}", src.id))?;
    let plan = match &a.spec {
        Some(p) => Some(plan_of(&src, &read_spec(p)?)?),
        None => None,
    };
    if a.json {
        let mut v = serde_json::to_value(&r)?;
        if let Some(p) = &plan {
            v["mock"] = serde_json::json!({
                "digest": p.digest, "layers_kept": p.selection.kept_layers(),
                "tensors": p.tensors.len(), "bytes": p.bytes(), "quant_pins": p.pinned,
                "resolved": p.resolved,
            });
        }
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!(
        "checkpoint  {} (revision {})",
        src.id,
        src.revision.as_deref().unwrap_or("not recorded")
    );
    println!(
        "arch        {} (model_type {}), {} layers, layout {}",
        r.arch, r.model_type, r.layers, r.layout
    );
    println!(
        "tensors     {} ({:.2} GB; {:.2} GB outside the layers)",
        r.tensors,
        gb(r.bytes),
        gb(r.fixed_bytes)
    );
    println!("signatures  (a mock keeps whole units of each)");
    for (i, s) in r.signatures.iter().enumerate() {
        println!(
            "  #{i}  {} unit(s) of [{}], {:.3} GB per unit, first unit layers {:?}",
            s.units,
            s.kinds.join(", "),
            gb(s.bytes_per_unit),
            s.first_unit
        );
        for p in &s.precision {
            println!("        {p}");
        }
    }
    println!("schemes");
    for (k, n) in &r.schemes {
        println!("  {n:6}  {k}");
    }
    if let Some(m) = &r.moe {
        println!(
            "experts     {} routed, top-{}: distinct experts per decode step (uniform routing)",
            m.experts, m.top_k
        );
        for (c, u) in &m.unique_experts {
            println!("  C{c:<4} {u:7.1}");
        }
        println!("  Keeping fewer experts lowers these counts, so mock specs keep all experts.");
    }
    if let Some(p) = &plan {
        println!("mock        digest {}", p.digest);
        println!(
            "  layers kept {:?}, {} tensors, {:.2} GB, {} quantization pin(s)",
            p.selection.kept_layers(),
            p.tensors.len(),
            gb(p.bytes()),
            p.pinned
        );
        for f in &p.routers {
            println!(
                "  router {} <- source layer {}: fit TV {:.4}, floored {}, gain {}",
                f.tensor, f.source_layer, f.tv, f.floored, f.gain
            );
        }
    }
    Ok(())
}

fn mockify(a: MockifyArgs) -> Result<()> {
    let files = read_spec(&a.spec)?;
    let src = open_source(&a.checkpoint, None, a.allow_network)?;
    let plan = plan_of(&src, &files)?;
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    print!("{}", plan.resolved);
    eprintln!(
        "mockify: {} tensors, {:.2} GB, layers {:?} -> {}",
        plan.tensors.len(),
        gb(plan.bytes()),
        plan.selection.kept_layers(),
        a.out.display()
    );
    let mut sink = FsSink::create(&a.out)?;
    let report = metrale_ml_utils::write_mock(&plan, src.source.as_ref(), &mut sink, threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let out = sink.commit()?;
    eprintln!(
        "mockify: wrote {} shard(s), {:.2} GB, copied {:?}; digest {} ({})",
        report.shards,
        gb(report.bytes),
        report.copied,
        plan.digest,
        out.join(metrale_ml_utils::RESOLVED_FILE).display()
    );
    Ok(())
}

fn extrapolate_cmd(a: ExtrapolateArgs) -> Result<()> {
    let scaling = Scaling::parse(&a.scaling).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut mocks = Vec::new();
    for p in &a.points {
        let (path, value) = p
            .rsplit_once('=')
            .with_context(|| format!("--point {p}: expected <resolved>=<value>"))?;
        let value: f64 = value
            .parse()
            .with_context(|| format!("--point {p}: {value} is not a number"))?;
        let mut file = PathBuf::from(path);
        if file.is_dir() {
            file = file.join(metrale_ml_utils::RESOLVED_FILE);
        }
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("reading {}", file.display()))?;
        mocks.push((
            units_of_resolved(&text).map_err(|e| anyhow::anyhow!("{}: {e}", file.display()))?,
            value,
        ));
    }
    let (points, full_units) = points_of(&mocks).map_err(|e| anyhow::anyhow!("{e}"))?;
    let e = extrapolate(&points, &full_units, scaling).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("full-model estimate  {:.6}", e.full);
    println!(
        "fit                  fixed {:.6e}, per unit {:?}, max residual {:.4}%",
        e.fixed,
        e.per_unit,
        e.max_residual * 100.0
    );
    println!("source units         {full_units:?}");
    if let Some(m) = a.full {
        if m == 0.0 {
            bail!("--full 0: an error relative to zero is undefined");
        }
        println!("measured             {m:.6}");
        println!("error                {:+.2}%", (e.full - m) / m * 100.0);
    }
    Ok(())
}

#[cfg(test)]
#[path = "ml_utils_tests.rs"]
mod ml_utils_tests;
