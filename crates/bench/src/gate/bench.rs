// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `kernels/<hw>/<model>/BENCH.toml`: a model's benchmark entries
//! and thresholds, beside its `MODEL.toml`, and [`baseline_for`], which
//! assembles one gate's [`GateBaseline`] from every such file.
//!
//! One file per model, one `[[benchmarks]]` entry per (quant, checkpoint,
//! gate). Hardware and model come from the file's path.
//!
//! [`super::taxon::configs`] does not list `BENCH.toml`, so editing a
//! threshold does not change any target's closure hash.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`load_all`] refuses an entry whose `status` is neither `measured` nor
//!   `unmeasured`, an `unmeasured` entry with metrics, a `measured` entry
//!   without them, a `port` serve override, an incomplete hermetic pin set,
//!   an undeclared `serve_env` lever, invalid `noise`, and a file whose
//!   top-level `owns` list `owns::from_bench_toml` refuses.
//! - [`baseline_for`] contains only `measured` entries, and exactly one
//!   default checkpoint per hardware.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::record::{GateBaseline, HardwareBaseline, ModelBaseline};
use super::taxon;

/// 2026-09-26: One `[[benchmarks]]` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchEntry {
    /// 2026-09-26: The quant, used as the `quant` of the entry's
    /// `taxon::Target`.
    pub quant: String,
    /// 2026-09-26: The served checkpoint; thresholds are keyed by it, not by
    /// the model directory.
    pub checkpoint: String,
    /// 2026-09-26: Benchmark id, e.g. `bfcl-subset`.
    pub gate: String,
    /// 2026-09-26: The recipe id `<family>/<stem>` that serves this entry,
    /// when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// 2026-09-26: Human name for this variant in the TUI's variant list;
    /// when empty the list shows the checkpoint id (`tui::bench_variants`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    /// 2026-09-26: Whether this is the checkpoint the gate runs when none is
    /// named.
    #[serde(default)]
    pub default: bool,
    /// 2026-09-26: `measured` or `unmeasured`; [`load_all`] refuses anything
    /// else.
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// 2026-09-26: Recipe keys self-start applies on every
    /// `--pull-request-gate` run, as strings matching
    /// `--serve-override KEY=VALUE` (e.g. `ssm_cache_slots = "256"` on the
    /// qwen3.6-35b-a3b `bfcl-subset-echolp` entry). Empty and omitted is the
    /// normal case.
    /// `check_record` requires the record's serve overrides to equal this map
    /// exactly. `port` is refused: self-start binds a free port itself.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_overrides: BTreeMap<String, String>,
    /// 2026-09-26: Benchmark parameter pins the gate applies on every
    /// `--pull-request-gate` run, for thresholds measured with non-default
    /// parameters. Empty and omitted is the normal case.
    ///
    /// Values are strings parsed by each parameter's own `ParamKind::parse`,
    /// like a typed `--param KEY=VALUE`, and an explicit `--param` wins
    /// (`bench_resolve::apply_param_overrides`). A key naming a
    /// `threshold_params`-coupled parameter, or no parameter at all, is
    /// refused there. `check_record` requires each pin on the record's
    /// `params`, compared ignoring spaces around commas.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub param_overrides: BTreeMap<String, String>,
    /// 2026-09-26: `METRALE_*` serve levers the gate pins for this entry,
    /// applied to the serve on top of the recipe's own `env:` block, for
    /// levers that are not recipe keys (e.g. `METRALE_FP8_ROWWISE` on the
    /// qwen3.8-27b `concurrency-sweep` entry). Empty and omitted is the normal
    /// case. Validated at parse by `serve_env::declared`; disclosed on the
    /// record as `GateRecord::serve_env`, never demanded by `check_record`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_env: BTreeMap<String, String>,
    /// 2026-09-26: Thresholds. Absent exactly when `status` is `unmeasured`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BTreeMap<String, super::record::Bound>>,
    /// 2026-10-15: A reference value per declared metric, e.g. another engine's number on the same
    /// cell (`targets = { median_ms_256 = 488.0 }`). Informational: no verdict reads it,
    /// so a bound can be a regression ratchet while a known loss against the reference is
    /// still on the page. [`load_all`] refuses a value that is not finite and, on a measured
    /// entry, a key that names no declared metric; an `unmeasured` entry has no metrics yet,
    /// so its keys are checked against its battle (`battle::problems`). Empty and omitted is
    /// the normal case.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub targets: BTreeMap<String, f64>,
    /// 2026-10-16: A short caveat per target, shown beside the target where it is published
    /// (`target_caveats = { c16_aggregate_tok_s = "likely vLLM stall" }`), for a reference figure
    /// that is not a clean bar. Prose only: no verdict and no battle reads it, and the full
    /// account belongs in `target_note`. [`load_all`] refuses a key that names no target, an
    /// empty text, and a caveat on an entry with no `target_note` for it to point to. Empty and
    /// omitted is the normal case.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub target_caveats: BTreeMap<String, String>,
    /// 2026-10-16: The account of the targets for a reader of the published page, which the
    /// caveat marks link to; `note` stays the engineering account and is never published. Prose
    /// only: no verdict, ratchet or battle reads it. [`load_all`] refuses it on an entry with
    /// no `targets`. Empty and omitted is the normal case.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub target_note: String,
    /// 2026-10-16: The context the reference run behind `targets` was served at
    /// (`target_max_model_len = 262144`), so a page can say where that run was not made. Prose
    /// only: no verdict, ratchet or battle reads it. [`load_all`] refuses it on an entry with
    /// no `targets` and refuses a value that is not positive. Omitted is the normal case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_max_model_len: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
struct BenchFile {
    #[serde(default)]
    benchmarks: Vec<BenchEntry>,
}

/// 2026-09-26: Every benchmark entry in the tree, validated, with the target
/// each belongs to.
///
/// Files are found by `bench_files`, not [`taxon::walk`]: the walk needs a
/// `MODEL.toml` and a quant directory, and a BENCH.toml may exist before its
/// model's kernels do.
pub fn load_all(root: &Path) -> Result<Vec<(taxon::Target, BenchEntry)>> {
    let mut out = Vec::new();
    for (hardware, model, path) in bench_files(root) {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let parsed: BenchFile =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        super::owns::from_bench_toml(&text)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        for entry in parsed.benchmarks {
            if entry.status != "measured" && entry.status != "unmeasured" {
                bail!(
                    "{}: status must be \"measured\" or \"unmeasured\", got {:?}",
                    path.display(),
                    entry.status
                );
            }
            if entry.status == "unmeasured" && entry.metrics.is_some() {
                bail!(
                    "{}: {} / {} is unmeasured but carries thresholds. A guessed \
                     number a run can clear is worse than no number — it reports \
                     PASS for something nobody measured.",
                    path.display(),
                    entry.gate,
                    entry.checkpoint
                );
            }
            if entry.status == "measured" && entry.metrics.as_ref().is_none_or(BTreeMap::is_empty) {
                bail!(
                    "{}: {} / {} claims to be measured but declares no metrics",
                    path.display(),
                    entry.gate,
                    entry.checkpoint
                );
            }
            if entry.serve_overrides.contains_key("port") {
                bail!(
                    "{}: {} / {} serve_overrides cannot set `port`: self-start binds \
                     a free port itself, so a pin here would name a listener that is not there",
                    path.display(),
                    entry.gate,
                    entry.checkpoint
                );
            }
            // 2026-09-26: A hermetic entry must also pin every key hermetic
            // closes (`hermetic::CLOSED_KEYS`) at its value. The record carries
            // those keys, and `scoring::check_record` compares serve overrides
            // in both directions, so a partial pin set could never pass;
            // refusing it here costs a parse, not a campaign.
            if super::hermetic::is_requested(&entry.serve_overrides) {
                let missing = super::hermetic::missing_pins(&entry.serve_overrides);
                if !missing.is_empty() {
                    bail!(
                        "{}: {} / {} pins hermetic=true but not {}: --hermetic expands into                          those keys, so the record will carry them and `check_record`                          compares the two sets in both directions. Pin them at these values                          or drop hermetic.",
                        path.display(),
                        entry.gate,
                        entry.checkpoint,
                        missing
                            .iter()
                            .map(|(k, v)| format!("{k}={v}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            // 2026-09-26: Lever pins are validated at parse for the same
            // reason, rather than by the serve after the first model load.
            crate::serve_env::declared(
                &format!(
                    "{} / {} [benchmarks.serve_env]",
                    entry.gate, entry.checkpoint
                ),
                &entry.serve_env,
            )
            .with_context(|| path.display().to_string())?;
            validate_noise(&path, &entry)?;
            validate_targets(&path, &entry)?;
            out.push((
                taxon::Target {
                    hardware: hardware.clone(),
                    model: model.clone(),
                    quant: entry.quant.clone(),
                },
                entry,
            ));
        }
    }
    Ok(out)
}

/// 2026-09-26: Every `kernels/<hw>/<model>/BENCH.toml`, with the hardware and
/// model directory names, sorted. A hardware directory is one holding
/// `HARDWARE.toml`; a model directory needs only the BENCH.toml.
fn bench_files(root: &Path) -> Vec<(String, String, std::path::PathBuf)> {
    let subdirs = |dir: &Path| -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        names.sort();
        names
    };
    let kernels = root.join("kernels");
    let mut out = Vec::new();
    for hardware in subdirs(&kernels) {
        let hw_dir = kernels.join(&hardware);
        if !hw_dir.join("HARDWARE.toml").exists() {
            continue;
        }
        for model in subdirs(&hw_dir) {
            let path = hw_dir.join(&model).join("BENCH.toml");
            if path.exists() {
                out.push((hardware.clone(), model, path));
            }
        }
    }
    out
}

/// 2026-09-26: Assemble the baseline for one gate from every `BENCH.toml` in
/// the tree.
///
/// `unmeasured` entries are left out, so `GateBaseline::resolve` on one fails
/// rather than returning an entry with no bounds: with "no baseline for
/// hardware …" when no `measured` entry is left on that hardware, and "no
/// baseline for model …" otherwise. Errors on two defaults on one hardware, a
/// checkpoint declared twice on one hardware, or a hardware with entries but no
/// default.
pub fn baseline_for(root: &Path, benchmark_id: &str) -> Result<GateBaseline> {
    let mut hardware: BTreeMap<String, HardwareBaseline> = BTreeMap::new();
    let mut defaults: BTreeMap<String, (String, String)> = BTreeMap::new();

    for (target, entry) in load_all(root)? {
        if entry.gate != benchmark_id || entry.status != "measured" {
            continue;
        }
        let Some(metrics) = entry.metrics.clone() else {
            continue;
        };
        if entry.default {
            if let Some((prior, prior_model)) = defaults.get(&target.hardware) {
                bail!(
                    "{benchmark_id}: both {prior} (in {prior_model}) and {} (in {}) \
                     claim to be the default on {}",
                    entry.checkpoint,
                    target.model,
                    target.hardware
                );
            }
            defaults.insert(
                target.hardware.clone(),
                (entry.checkpoint.clone(), target.model.clone()),
            );
        }
        let hw = hardware
            .entry(target.hardware.clone())
            .or_insert_with(|| HardwareBaseline {
                default: String::new(),
                models: BTreeMap::new(),
            });
        if hw
            .models
            .insert(
                entry.checkpoint.clone(),
                ModelBaseline {
                    recipe: entry.recipe.clone(),
                    label: entry.label.clone(),
                    note: entry.note.clone(),
                    metrics,
                    serve_overrides: entry.serve_overrides.clone(),
                    param_overrides: entry.param_overrides.clone(),
                    serve_env: entry.serve_env.clone(),
                },
            )
            .is_some()
        {
            bail!(
                "{benchmark_id}: {} is declared twice on {}",
                entry.checkpoint,
                target.hardware
            );
        }
    }

    for (hw_name, hw) in &mut hardware {
        match defaults.get(hw_name) {
            Some((checkpoint, _)) => hw.default = checkpoint.clone(),
            // 2026-09-26: No implicit default, even for a single checkpoint.
            None => bail!(
                "{benchmark_id}: no checkpoint on {hw_name} sets `default = true`; \
                 one must, or the gate has no defined subject"
            ),
        }
    }
    Ok(GateBaseline {
        schema: 2,
        hardware,
    })
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod bench_tests;

#[cfg(test)]
#[path = "bench_serve_pin_tests.rs"]
mod bench_serve_pin_tests;

#[cfg(test)]
#[path = "bench_target_tests.rs"]
mod bench_target_tests;

/// 2026-09-26: The largest `noise` an entry may declare, as a fraction of the
/// bound's magnitude. `noise` widens a threshold at compare time
/// (`scoring::compare`), so without a cap a large value would pass any record
/// while reading like a measurement annotation.
const MAX_NOISE_FRACTION: f64 = 0.05;

/// 2026-09-26: `noise` must be finite, non-negative, at most
/// [`MAX_NOISE_FRACTION`] of a non-zero bound, and absent on an exact pin.
///
/// `min == max` is how the BFCL draw size is pinned (`samples` = 995 in
/// qwen3.6-27b, 1004 in qwen3.6-35b-a3b), to catch a changed draw.
/// `scoring::compare` applies `noise` to the two-sided arm as well, so noise
/// on a pin would widen it into a range.
fn validate_noise(path: &std::path::Path, entry: &BenchEntry) -> Result<()> {
    let Some(metrics) = entry.metrics.as_ref() else {
        return Ok(());
    };
    for (name, bound) in metrics {
        let Some(noise) = bound.noise else { continue };
        if !noise.is_finite() || noise < 0.0 {
            bail!(
                "{}: {} / {} metric {name}: noise must be finite and non-negative, got {noise}",
                path.display(),
                entry.gate,
                entry.checkpoint,
            );
        }
        if bound.min.is_some() && bound.min == bound.max {
            bail!(
                "{}: {} / {} metric {name} is an EXACT pin (min == max == {:?}) and carries \
                 noise {noise}. Noise on a pin disables it — and a pin is used for things like \
                 the BFCL draw size, where a changed draw is undetectable after the fact.",
                path.display(),
                entry.gate,
                entry.checkpoint,
                bound.min,
            );
        }
        let magnitude = bound.min.or(bound.max).unwrap_or(0.0).abs();
        let cap = magnitude * MAX_NOISE_FRACTION;
        if magnitude > 0.0 && noise > cap {
            bail!(
                "{}: {} / {} metric {name}: noise {noise} exceeds {:.0}% of the bound \
                 ({magnitude}) — that is a threshold change wearing a measurement-noise \
                 label. Move the bound instead, so the ratchet is visible in review.",
                path.display(),
                entry.gate,
                entry.checkpoint,
                MAX_NOISE_FRACTION * 100.0,
            );
        }
    }
    Ok(())
}

/// 2026-10-15: Each `targets` key of a measured entry must name a declared metric and each value
/// must be finite: a reference on a metric the entry does not bound, or a NaN, reads like a bar
/// while meaning nothing. 2026-10-16: An `unmeasured` entry declares no metric yet, so its
/// keys are left to `battle::problems`, which names one that is no battle cell.
fn validate_targets(path: &std::path::Path, entry: &BenchEntry) -> Result<()> {
    for (name, value) in &entry.targets {
        let declared = entry.metrics.as_ref().is_none_or(|m| m.contains_key(name));
        if !declared {
            bail!(
                "{}: {} / {} [benchmarks.targets] names {name}, which is not a declared metric",
                path.display(),
                entry.gate,
                entry.checkpoint,
            );
        }
        if !value.is_finite() {
            bail!(
                "{}: {} / {} target {name} must be finite, got {value}",
                path.display(),
                entry.gate,
                entry.checkpoint,
            );
        }
    }
    validate_target_caveats(path, entry)
}

/// 2026-10-16: A caveat flags a published target, so it must name one; an empty caveat flags
/// nothing; and the marker points at the entry's `target_note`, so an entry with a caveat must
/// carry one, and that account describes targets, so it needs some. The reference run's context describes the targets, so it needs some, and a
/// context is positive.
fn validate_target_caveats(path: &std::path::Path, entry: &BenchEntry) -> Result<()> {
    for (name, text) in &entry.target_caveats {
        if !entry.targets.contains_key(name) {
            bail!(
                "{}: {} / {} [benchmarks.target_caveats] names {name}, which is not a target",
                path.display(),
                entry.gate,
                entry.checkpoint,
            );
        }
        if text.trim().is_empty() {
            bail!(
                "{}: {} / {} target caveat {name} is empty",
                path.display(),
                entry.gate,
                entry.checkpoint,
            );
        }
    }
    let (gate, ckpt) = (&entry.gate, &entry.checkpoint);
    if !entry.target_caveats.is_empty() && entry.target_note.trim().is_empty() {
        bail!(
            "{}: {gate} / {ckpt} has target caveats but no target_note for them to point to",
            path.display()
        );
    }
    if !entry.target_note.is_empty() && entry.targets.is_empty() {
        bail!(
            "{}: {gate} / {ckpt} declares target_note with no targets for it to describe",
            path.display()
        );
    }
    match entry.target_max_model_len {
        Some(_) if entry.targets.is_empty() => bail!(
            "{}: {} / {} declares target_max_model_len with no targets for it to describe",
            path.display(),
            entry.gate,
            entry.checkpoint,
        ),
        Some(len) if len <= 0 => bail!(
            "{}: {} / {} target_max_model_len must be positive, got {len}",
            path.display(),
            entry.gate,
            entry.checkpoint,
        ),
        _ => {}
    }
    Ok(())
}
