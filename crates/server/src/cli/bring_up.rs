// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `met ml-utils model-bring-up-bench` (also `met dev-tools …`): bring-up bench of
//! any OpenAI-compatible endpoint, vLLM or Metrale. It runs the suite's own TTFT gates
//! sequentially at one stream (cold, warm, high-ISL cold, high-ISL warm) with prompts cut to
//! exact token counts of the served model's tokenizer, reading the engine's prefix-cache
//! counters around each, then the suite's `concurrency-sweep` gate over the `--concs` ladder
//! with optional NVML energy readers (`--energy`); it prints the tables and writes a JSON
//! record. `--compare A B` renders two records side by side.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants:
//! - The I/O host: the plan, the rows, the verdicts and the text are `bring_up_core` and
//!   `bring_up_render`; the measurement is `metrale_bench::headless::run_blocking` of the gates.
//! - No gate run stores a baseline (`update_baseline = false`) or a history record; the record
//!   this writes holds every gate's run record instead.
//! - Without `--url` the local serve is discovered (`bring_up_discover`); without `--tokenizer`
//!   the tokenizer comes from that serve's `--model-from-path`; otherwise the run refuses.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use metrale_bench::headless::{HeadlessOptions, RunRequest, run_blocking};
use metrale_bench::serve_identity::{ServeIdentity, file_sha256};
use metrale_bench::{
    ArtifactStore, BenchmarkExecutor, ParamValue, ParamValues, TargetEndpoint, http,
};

use super::BringUpArgs;
use super::bring_up_conc::{self as conc, Instrument, Section};
use super::bring_up_core::{self as core, Bench, Engine, Record, Row, Step};
use super::bring_up_discover::{self as discover, Candidate};
use super::bring_up_render as render;

const HTTP: Duration = Duration::from_secs(10);

/// 2026-10-10: Measure, or with `--compare`, render two records.
pub(crate) async fn dispatch(a: BringUpArgs) -> Result<()> {
    if let Some(files) = &a.compare {
        let [x, y] = files.as_slice() else {
            bail!("--compare takes two records");
        };
        let (x, y) = (read_record(x)?, read_record(y)?);
        print!("{}", render::compare(&x, &y)?);
        return Ok(());
    }
    let out = a.out.clone().context("--out is required when measuring")?;
    let concs = conc::parse_concs(&a.concs)?;
    if a.skip_ttft && a.skip_concurrency {
        bail!("--skip-ttft and --skip-concurrency together leave nothing to measure");
    }
    let (url, discovered) = match &a.url {
        Some(u) => (u.clone(), None),
        None => {
            let (u, c) = discover::local_serve().await?;
            eprintln!("bring-up: local Metrale serve pid {} at {u}", c.pid);
            (u, Some(c))
        }
    };
    let probe = TargetEndpoint::new(url.clone(), "");
    let models = http::get_json(&probe, "/v1/models", HTTP)
        .await
        .with_context(|| format!("{url} did not answer /v1/models"))?;
    let model = core::resolve_model(&models, a.model.as_deref())?;
    let target = TargetEndpoint::new(url.clone(), model.clone());
    let tokenizer = if a.skip_ttft {
        None
    } else {
        Some(tokenizer(a.tokenizer.as_deref(), discovered.as_ref())?)
    };
    let engine = identify(&target, &models, &model, a.label.clone()).await;
    let steps = if a.skip_ttft {
        Vec::new()
    } else {
        core::plan(&isl(&a)?, high_isl(&a)?)
    };
    print_plan(&engine, &model, &steps, a.reps)?;
    if let Some(concern) = coherence(&target).await {
        eprintln!("bring-up: endpoint check: {concern}");
    }

    let store = ArtifactStore::discover().context("locating METRALE_HOME")?;
    let executor = BenchmarkExecutor::new(tokio::runtime::Handle::current(), store);
    crate::tui::shutdown::disarm_startup_escape();
    crate::tui::shutdown::install_signal_listeners();
    let started_at = now();
    let mut ttft = Vec::new();
    for step in &steps {
        if crate::tui::shutdown::requested() {
            bail!("interrupted before {} @{}", step.bench.label(), step.tokens);
        }
        let tok = tokenizer
            .as_deref()
            .context("the TTFT benches need a tokenizer")?;
        ttft.push(run_step(&executor, &target, step, tok, a.reps).await?);
    }
    let concurrency = if a.skip_concurrency {
        None
    } else {
        Some(run_ladder(&executor, &target, &a, concs).await?)
    };
    let record = Record {
        schema: core::SCHEMA.to_string(),
        url,
        model,
        tokenizer: tokenizer.as_ref().map(|t| t.display().to_string()),
        tokenizer_sha256: tokenizer.as_deref().map(file_sha256).transpose()?,
        engine,
        started_at,
        metrale_version: super::METRALE_VERSION.to_string(),
        ttft,
        concurrency,
    };
    let path = write_record(&out, &record)?;
    print!("{}", render::table(&record));
    println!("record: {}", path.display());
    let failed = record
        .ttft
        .iter()
        .filter(|r| r.status != "completed")
        .count()
        + record
            .concurrency
            .as_ref()
            .map_or(0, |c| usize::from(c.status != "completed"));
    if failed > 0 {
        bail!("{failed} bench(es) did not complete; see the table");
    }
    Ok(())
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn read_record(p: &Path) -> Result<Record> {
    let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    Record::parse(&text).with_context(|| format!("{}", p.display()))
}

fn write_record(dir: &Path, r: &Record) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let name: String = r
        .engine
        .label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("bring-up-{name}-{}.json", r.started_at));
    if path.exists() {
        bail!("{} exists; nothing is overwritten", path.display());
    }
    std::fs::write(&path, serde_json::to_string_pretty(r)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// 2026-10-10: The tokenizer.json to cut prompts with: `--tokenizer` (a file, a checkpoint
/// directory or a Hub id in the local cache), else the discovered serve's `--model-from-path`.
fn tokenizer(arg: Option<&str>, discovered: Option<&Candidate>) -> Result<PathBuf> {
    let spec = match (arg, discovered.and_then(|c| c.model_from_path.as_ref())) {
        (Some(t), _) => t.to_string(),
        (None, Some(dir)) => dir.display().to_string(),
        (None, None) => bail!(
            "pass --tokenizer: the served model's tokenizer.json, its checkpoint directory or its \
             Hub id, so the prompts are cut to that model's token counts"
        ),
    };
    let local = if Path::new(&spec).exists() {
        PathBuf::from(&spec)
    } else {
        crate::model_resolver::resolve_model_dir(&spec, None)
            .with_context(|| format!("--tokenizer {spec}: not a path and not in the local cache"))?
    };
    metrale_bench::benchmarks::ttft::tokenizer_file(&local.display().to_string())
}

fn gate_defaults(bench: Bench) -> Result<ParamValues> {
    let d = super::bench_run::find(bench.gate())?;
    Ok(ParamValues::defaults(&d.build().parameters()))
}

/// 2026-10-10: `--isl`, else the synthetic gates' own `prompt_lengths` default.
fn isl(a: &BringUpArgs) -> Result<Vec<usize>> {
    match &a.isl {
        Some(v) => Ok(v.clone()),
        None => Ok(gate_defaults(Bench::Cold)?
            .int_list("prompt_lengths")?
            .iter()
            .map(|v| *v as usize)
            .collect()),
    }
}

/// 2026-10-10: `--high-isl`, else the high-ISL gates' own `min_prompt_tokens` default.
fn high_isl(a: &BringUpArgs) -> Result<usize> {
    match a.high_isl {
        Some(v) => Ok(v),
        None => gate_defaults(Bench::HighIslCold)?.usize("min_prompt_tokens"),
    }
}

/// 2026-10-10: A step's gate parameters: the gate's defaults, the exact size, the tokenizer,
/// no baseline, and `--reps` when given.
fn values(step: &Step, tokenizer: &Path, reps: Option<usize>) -> Result<ParamValues> {
    let d = super::bench_run::find(step.bench.gate())?;
    let specs = d.build().parameters();
    let mut v = ParamValues::defaults(&specs);
    v.set(
        "tokenizer",
        ParamValue::Text(tokenizer.display().to_string()),
    );
    v.set("update_baseline", ParamValue::Bool(false));
    let size = match step.bench {
        Bench::Cold | Bench::Warm => ParamValue::IntList(vec![step.tokens as i64]),
        Bench::HighIslCold | Bench::HighIslWarm => ParamValue::Int(step.tokens as i64),
    };
    v.set(step.bench.size_param(), size);
    if let Some(r) = reps {
        v.set("repeats", ParamValue::Int(r as i64));
    }
    v.validate_against(&specs)?;
    Ok(v)
}

fn print_plan(e: &Engine, model: &str, steps: &[Step], reps: Option<usize>) -> Result<()> {
    eprintln!("bring-up: {} ({:?}) model {model}", e.label, e.kind);
    for s in steps {
        let r = match reps {
            Some(r) => r,
            None => gate_defaults(s.bench)?.usize("repeats")?,
        };
        eprintln!(
            "  {} @{} tokens x {r} ({})",
            s.bench.label(),
            s.tokens,
            s.bench.gate()
        );
    }
    Ok(())
}

/// 2026-10-10: The suite's coherence probe, once before the plan; the gate runs skip it so
/// its requests stay out of every bench's cache-counter window.
async fn coherence(target: &TargetEndpoint) -> Option<String> {
    let report = metrale_bench::coherence::probe_for(target, None, Duration::from_secs(60)).await;
    report.concern(target)
}

async fn identify(
    t: &TargetEndpoint,
    models: &serde_json::Value,
    model: &str,
    label: Option<String>,
) -> Engine {
    let entry = models["data"]
        .as_array()
        .and_then(|a| a.iter().find(|m| m["id"].as_str() == Some(model)))
        .cloned()
        .unwrap_or_default();
    let serve_identity = http::get_json(t, "/serve-config", HTTP)
        .await
        .ok()
        .filter(|v| serde_json::from_value::<ServeIdentity>(v.clone()).is_ok());
    let version = http::get_json(t, "/version", HTTP)
        .await
        .ok()
        .and_then(|v| v["version"].as_str().map(str::to_string));
    let metrics = http::get_text(t, "/metrics", HTTP)
        .await
        .unwrap_or_default();
    let owned_by = entry["owned_by"].as_str().map(str::to_string);
    let kind = core::engine_kind(owned_by.as_deref(), serve_identity.is_some());
    Engine {
        kind,
        label: label.unwrap_or_else(|| format!("{kind:?}").to_lowercase()),
        owned_by,
        version,
        max_model_len: entry["max_model_len"].as_u64(),
        serve_identity,
        cache_config: core::cache_config(&metrics),
    }
}

async fn hit_tokens(t: &TargetEndpoint) -> Option<(&'static str, f64)> {
    let text = http::get_text(t, "/metrics", HTTP).await.ok()?;
    core::hit_tokens(&text)
}

/// 2026-10-10: Run one gate to its end, off the runtime (`run_blocking` sleeps its thread).
async fn run_gate(
    executor: &BenchmarkExecutor,
    target: &TargetEndpoint,
    gate: &str,
    values: ParamValues,
) -> Result<metrale_bench::headless::RunOutcome> {
    let request = RunRequest {
        descriptor: super::bench_run::find(gate)?,
        values,
        target: target.clone(),
        options: HeadlessOptions {
            save: false,
            coherence: metrale_bench::CoherencePolicy::Skip,
            ..HeadlessOptions::cli(super::METRALE_VERSION)
        },
    };
    let executor = executor.clone();
    tokio::task::spawn_blocking(move || {
        let mut reporter = super::bench_print::StdoutReporter::new(true);
        run_blocking(
            &executor,
            request,
            &mut reporter,
            &crate::tui::shutdown::requested,
        )
    })
    .await?
}

/// 2026-10-10: `completed`, or the terminal frame's status, phase and last log line.
fn status_of(frame: &metrale_bench::BenchmarkResult) -> String {
    match frame.status {
        metrale_bench::RunStatus::Completed => "completed".to_string(),
        _ => format!(
            "{:?} at {}: {}",
            frame.status,
            frame.phase,
            frame.log.last().map_or("", |l| l.text.as_str())
        ),
    }
}

async fn run_step(
    executor: &BenchmarkExecutor,
    target: &TargetEndpoint,
    step: &Step,
    tokenizer: &Path,
    reps: Option<usize>,
) -> Result<Row> {
    let values = values(step, tokenizer, reps)?;
    let repeats = values.usize("repeats")?;
    eprintln!("bring-up: {} @{} tokens", step.bench.label(), step.tokens);
    let before = hit_tokens(target).await;
    let outcome = run_gate(executor, target, step.bench.gate(), values).await?;
    let after = hit_tokens(target).await;
    let frame = &outcome.record.frame;
    Ok(core::row(
        step,
        repeats,
        &frame.metrics,
        status_of(frame),
        before,
        after,
        serde_json::to_value(&outcome.record)?,
    ))
}

/// 2026-10-10: The sweep's `prompt_mode` for the ladder: the published ladder's long-essay ask,
/// so every request runs to its output budget.
const LADDER_PROMPT_MODE: &str = "essay";

/// 2026-10-10: The concurrency ladder: one `concurrency-sweep` run over `concs` at
/// `--conc-isl`/`--conc-osl`, with `--energy` readers around it.
async fn run_ladder(
    executor: &BenchmarkExecutor,
    target: &TargetEndpoint,
    a: &BringUpArgs,
    concs: Vec<usize>,
) -> Result<Section> {
    const GATE: &str = "concurrency-sweep";
    let specs = super::bench_run::find(GATE)?.build().parameters();
    let mut v = ParamValues::defaults(&specs);
    v.set(
        "concurrencies",
        ParamValue::IntList(concs.iter().map(|c| *c as i64).collect()),
    );
    v.set("isls", ParamValue::IntList(vec![a.conc_isl as i64]));
    v.set("osl", ParamValue::Int(a.conc_osl as i64));
    v.set(
        "prompt_mode",
        ParamValue::Text(LADDER_PROMPT_MODE.to_string()),
    );
    v.validate_against(&specs)?;
    let instrument = Instrument {
        concs: concs.clone(),
        isl: a.conc_isl,
        osl: a.conc_osl,
        prompt_mode: LADDER_PROMPT_MODE.to_string(),
        warmup: v.usize("warmup")?,
    };
    eprintln!(
        "bring-up: concurrency ladder {concs:?} (isl {} osl {})",
        a.conc_isl, a.conc_osl
    );
    let meter = if a.energy.is_empty() {
        None
    } else {
        let hosts = a.energy.clone();
        Some(
            tokio::task::spawn_blocking(move || super::bring_up_energy::Meter::start(&hosts))
                .await??,
        )
    };
    let outcome = run_gate(executor, target, GATE, v).await?;
    let series = meter.as_ref().map(|m| m.series()).unwrap_or_default();
    drop(meter);
    let frame = &outcome.record.frame;
    Ok(Section {
        instrument,
        energy_hosts: a.energy.clone(),
        rungs: conc::rungs(&concs, &frame.metrics, &series),
        status: status_of(frame),
        run: serde_json::to_value(&outcome.record)?,
    })
}
