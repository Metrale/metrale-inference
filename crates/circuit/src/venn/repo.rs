// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: A whole `met circuit venn` run over a repository reached through [`Repo`]:
//! resolve the target and compared instances, load their circuits, check the family manifest
//! against the kernel sources, build and render the report. The CLI supplies a file-system
//! [`Repo`]; the tests supply the same one over the checked-out tree, so both run this code.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Business logic only: every byte arrives through [`Repo`] (SBIO), nothing touches the
//!   environment.
//! - Manifest drift against the sources is an error before any report is built.
//! - A target named by a checkpoint directory is checked against that checkpoint's own
//!   `config.json` (layer kinds, declared precision of every bound module); a disagreement is an
//!   error, never a note.

use super::discover::{KernelSources, drift, macro_files};
use super::report::{Side, VennInputs};
use super::{VennArgs, VennError, build, parse_families, parse_measurements, render};
use crate::instances::{Instance, PrecisionSpec, parse_instances};
use crate::ir::LayerKind;
use crate::{Loaded, Sources, includes_of, load};

/// 2026-09-29: Read access to a repository, by repo-relative path.
pub trait Repo {
    /// 2026-09-29: The text of `rel`.
    fn read(&self, rel: &str) -> Result<String, String>;
    /// 2026-09-29: Every file under the directory `rel`, recursively, repo-relative with `/`.
    fn list(&self, rel: &str) -> Result<Vec<String>, String>;
}

/// 2026-09-29: The instance `spec` names: a recipe id, or an arch that exactly one instance
/// serves, or a checkpoint id.
pub fn resolve(all: &[Instance], spec: &str) -> Result<Instance, VennError> {
    if let Some(i) = all
        .iter()
        .find(|i| i.recipe == spec || i.checkpoint == spec)
    {
        return Ok(i.clone());
    }
    let by_arch: Vec<&Instance> = all.iter().filter(|i| i.arch == spec).collect();
    match by_arch.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(VennError::Load(format!(
            "`{spec}` is no recipe, checkpoint or arch in kernels/circuits/INSTANCES.toml (recipes: {})",
            all.iter()
                .map(|i| i.recipe.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        many => Err(VennError::Load(format!(
            "arch `{spec}` has several instances; name one recipe: {}",
            many.iter()
                .map(|i| i.recipe.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// 2026-09-29: The checkpoint id a directory holds, from its path: the Hugging Face cache
/// layout (`.../models--<org>--<name>/snapshots/<rev>`) or, failing that, the last two path
/// components (`.../<org>/<name>`).
pub fn checkpoint_id_of(dir: &str) -> Option<String> {
    let parts: Vec<&str> = dir
        .trim_end_matches('/')
        .split('/')
        .filter(|p| !p.is_empty())
        .collect();
    if let Some(p) = parts.iter().find_map(|p| p.strip_prefix("models--")) {
        let (org, name) = p.split_once("--")?;
        return Some(format!("{org}/{name}"));
    }
    match parts.as_slice() {
        [.., org, name] => Some(format!("{org}/{name}")),
        _ => None,
    }
}

pub(crate) fn load_instance(repo: &dyn Repo, inst: &Instance) -> Result<Loaded, VennError> {
    let io = |e: String| VennError::Load(format!("{}: {e}", inst.recipe));
    let circuit = repo
        .read(&format!("kernels/circuits/{}.toml", inst.arch))
        .map_err(io)?;
    let names = includes_of(&circuit).map_err(|e| io(e.to_string()))?;
    let mut blocks = Vec::with_capacity(names.len());
    for n in names {
        let text = repo
            .read(&format!("kernels/circuits/blocks/{n}.toml"))
            .map_err(io)?;
        blocks.push((n, text));
    }
    let precision = match &inst.precision {
        PrecisionSpec::Table(n) => format!("kernels/circuits/precision/{n}.toml"),
        PrecisionSpec::Policy {
            checkpoint_plan, ..
        } => format!("kernels/circuits/checkpoints/{checkpoint_plan}.toml"),
    };
    let precision = repo.read(&precision).map_err(io)?;
    let rules = repo
        .read(&format!("kernels/{}/common/FUSIONS.toml", hardware(inst)?))
        .map_err(io)?;
    let blocks: Vec<(&str, &str)> = blocks
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    load(
        inst,
        Sources {
            circuit: &circuit,
            precision: &precision,
            rules: &rules,
            blocks: &blocks,
        },
    )
    .map_err(|e| io(e.to_string()))
}

fn hardware(inst: &Instance) -> Result<&str, VennError> {
    inst.target
        .split('/')
        .next()
        .filter(|h| !h.is_empty())
        .ok_or_else(|| {
            VennError::Load(format!(
                "{}: target `{}` names no hardware",
                inst.recipe, inst.target
            ))
        })
}

/// 2026-10-05: Everything a report is built from, read and checked: the target and compared
/// instances, loaded; the family manifest (checked against the kernel sources); the measurements.
pub struct Prepared {
    /// 2026-10-05: The target instance.
    pub target: Instance,
    /// 2026-10-05: Its circuit and rules.
    pub target_loaded: Loaded,
    /// 2026-10-05: The compared instances, in `--against` order.
    pub against: Vec<Instance>,
    /// 2026-10-05: Their circuits and rules, in the same order.
    pub against_loaded: Vec<Loaded>,
    /// 2026-10-05: The instances' hardware's kernel-family manifest.
    pub families: super::Families,
    /// 2026-10-05: measurements.toml.
    pub measurements: super::Measurements,
}

/// 2026-10-05: Resolve, load and check what `args` names. `checkpoint`, when the target was
/// given as a checkpoint directory, is that directory's `config.json` text and, if present, its
/// `hf_quant_config.json` text.
pub fn prepare(
    repo: &dyn Repo,
    args: &VennArgs,
    checkpoint: Option<(&str, Option<&str>)>,
) -> Result<Prepared, VennError> {
    let io = |e: String| VennError::Load(e);
    let all = parse_instances(&repo.read("kernels/circuits/INSTANCES.toml").map_err(io)?)
        .map_err(|e| io(e.to_string()))?;
    let target = resolve(&all, &args.target)?;
    let against = args
        .against
        .iter()
        .map(|a| resolve(&all, a))
        .collect::<Result<Vec<_>, _>>()?;
    let hw = hardware(&target)?;
    if let Some(a) = against.iter().find(|a| hardware(a).ok() != Some(hw)) {
        return Err(io(format!(
            "`{}` runs on another hardware than the target",
            a.recipe
        )));
    }
    let lt = load_instance(repo, &target)?;
    if let Some((config, hfq)) = checkpoint {
        super::checkpoint::check(&target, &lt, config, hfq)?;
    }
    let loaded = against
        .iter()
        .map(|a| load_instance(repo, a))
        .collect::<Result<Vec<_>, _>>()?;
    let fams = parse_families(
        &repo
            .read(&format!("kernels/{hw}/common/KERNEL_FAMILIES.toml"))
            .map_err(io)?,
    )
    .map_err(|e| io(e.to_string()))?;
    let mut src = KernelSources {
        paths: repo
            .list("kernels")
            .map_err(io)?
            .into_iter()
            .chain(repo.list("crates").map_err(io)?)
            .collect(),
        ..KernelSources::default()
    };
    for f in macro_files(&fams) {
        let text = repo.read(&f).map_err(io)?;
        src.texts.insert(f, text);
    }
    let problems = drift(&fams, &src);
    if !problems.is_empty() {
        return Err(VennError::Drift(problems));
    }
    let meas = parse_measurements(
        &repo
            .read("docs/kernel-perf/measurements.toml")
            .map_err(io)?,
    )
    .map_err(io)?;
    Ok(Prepared {
        target,
        target_loaded: lt,
        against,
        against_loaded: loaded,
        families: fams,
        measurements: meas,
    })
}

/// 2026-09-29: Build and render the report `args` asks for, each side planned offline with its
/// own rules. A report on a device (`args.hardware`) is `hardware::venn_text`'s.
pub fn report_text(
    repo: &dyn Repo,
    args: &VennArgs,
    checkpoint: Option<(&str, Option<&str>)>,
) -> Result<String, VennError> {
    if let Some(h) = &args.hardware {
        return Err(VennError::Run(format!(
            "--hardware {h}: a report on a device is planned by hardware::venn_text"
        )));
    }
    let p = prepare(repo, args, checkpoint)?;
    let inputs = VennInputs {
        target: Side {
            instance: &p.target,
            loaded: &p.target_loaded,
            on_device: None,
        },
        against: p
            .against
            .iter()
            .zip(&p.against_loaded)
            .map(|(instance, loaded)| Side {
                instance,
                loaded,
                on_device: None,
            })
            .collect(),
        families: &p.families,
        measurements: &p.measurements,
        runs: args.runs()?,
        command: args.command(),
        device: None,
    };
    Ok(render(&build(&inputs)?))
}

/// 2026-09-29: The layer kinds a `config.json` declares, from `layers_block_type`
/// (`mamba` / `moe` / `attention`) or `layer_types` (`linear_attention` / `full_attention`).
pub fn config_layer_kinds(cfg: &serde_json::Value) -> Result<Vec<LayerKind>, VennError> {
    let bad = |d: String| VennError::Checkpoint(d);
    let (key, list) = ["layers_block_type", "layer_types"]
        .iter()
        .find_map(|k| cfg.get(*k).and_then(|v| v.as_array()).map(|a| (*k, a)))
        .ok_or_else(|| {
            bad("config.json has neither `layers_block_type` nor `layer_types`".into())
        })?;
    list.iter()
        .map(|v| {
            let s = v.as_str().unwrap_or_default();
            let kind = match s {
                "attention" => Some(LayerKind::FullAttention),
                other => LayerKind::parse(other),
            };
            kind.ok_or_else(|| {
                bad(format!(
                    "`{key}` entry `{s}` is no layer kind a circuit models"
                ))
            })
        })
        .collect()
}
