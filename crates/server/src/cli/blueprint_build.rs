// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met blueprint from-recipe`: the Phase-1 builder. A recipe, its circuit
//! instance (embedded INSTANCES.toml), the checkpoint directory and the kernel modules embedded
//! in this binary become an unsigned blueprint whose every module is `internal`.
//!
//! Owner: server CLI (blueprint).
//! Invariants:
//! - Every arm the instance lists is fused; one that does not fuse is an error, never skipped.
//! - Every `module::func` a plan names must be an entry of an embedded module of the
//!   instance's target; a missing one is `KernelMissing` at build, not at load.
//! - The checkpoint's `config.json` must reproduce the instance's shape.
//! - I/O (recipe, checkpoint, template) happens here; the blueprint crate stays pure.

use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_blueprint::frame::{KernelSource, Unsigned};
use metrale_blueprint::recipe_view::{self, RecipeParts};
use metrale_blueprint::types::{self as w, Blueprint};
use metrale_blueprint::{ArtifactId, LineageId, Sha256, Sha256Stream};
use metrale_circuit::{AvailableKernels, FusionPlan, Instance, KernelId};

use super::blueprint_args::BuildArgs;
use crate::recipe::Recipe;

/// 2026-10-22: HARDWARE.toml per class, embedded beside FUSIONS.toml so a blueprint pins the
/// class this binary was built against.
const HARDWARE: [(&str, &str); 1] = [(
    "gb10",
    include_str!(concat!(
        env!("METRALE_KERNEL_TREE_ROOT"),
        "/kernels/gb10/HARDWARE.toml"
    )),
)];

/// 2026-10-22: The recipe at `path`, with the id `<family>/<stem>` its path names.
pub(crate) fn recipe_at(path: &Path) -> Result<Recipe> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("recipe path has no stem")?;
    let family = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|s| s.to_str())
        .context("recipe path has no family directory")?;
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Recipe::parse(format!("{family}/{stem}"), &text)
}

/// 2026-10-22: The fields of `r` the projection reads.
pub(crate) fn parts_of(r: &Recipe) -> Result<RecipeParts> {
    Ok(RecipeParts {
        id: r.id.clone(),
        version: r.version.clone(),
        model: r.model.clone(),
        runtime: r.runtime.clone(),
        container: r.container.clone(),
        min_nodes: u32::try_from(r.min_nodes).context("min_nodes exceeds u32")?,
        defaults: r.defaults.clone(),
        env: r.env.clone(),
    })
}

/// 2026-10-22: A recipe from projected parts; metadata the blueprint does not carry is empty.
pub(crate) fn recipe_of(p: RecipeParts) -> Recipe {
    Recipe {
        id: p.id,
        version: p.version,
        model: p.model,
        runtime: p.runtime,
        container: p.container,
        min_nodes: p.min_nodes as usize,
        description: String::new(),
        maintainer: String::new(),
        category: String::new(),
        model_params: String::new(),
        quantization: String::new(),
        kv_dtype: String::new(),
        updated: String::new(),
        defaults: p.defaults,
        env: p.env,
        starting_point: None,
    }
}

/// 2026-10-22: The embedded modules of `target`.
pub(crate) fn embedded(
    target: &w::KernelTarget<String>,
) -> Result<Vec<(&'static str, &'static [u8])>> {
    metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target == target.as_strs())
        .map(|s| s.modules)
        .with_context(|| {
            format!(
                "this binary embeds no kernel target {}/{}/{}; build it with METRALE_TARGET_MODEL={}",
                target.arch, target.model, target.quant, target.model
            )
        })
}

/// 2026-10-22: The `.entry` names of a PTX module, sorted. 2026-10-24: A fatbin's are read
/// from the uncompressed PTX it carries, whose lines survive the lossy decode intact.
pub(crate) fn ptx_entries(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    let mut out: Vec<String> = text
        .lines()
        .filter_map(|l| {
            let rest = l.split(".entry ").nth(1)?;
            let name: String = rest
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '(')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

fn sha_file(path: &Path) -> Result<Sha256> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256Stream::default();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            return Ok(h.finish());
        }
        h.update(&buf[..n]);
    }
}

fn checkpoint(args: &BuildArgs, repo: &str, inst: &Instance) -> Result<w::CheckpointPin> {
    let dir = &args.checkpoint;
    let config = std::fs::read(dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", dir.display()))?;
    let mapped = metrale_circuit::map_checkpoint(std::str::from_utf8(&config)?)?;
    let drift = super::circuit::shape_drift(&inst.shape, &mapped.shape);
    if !drift.is_empty() {
        bail!(
            "{}/config.json disagrees with the instance:\n  {}",
            dir.display(),
            drift.join("\n  ")
        );
    }
    let quant = dir.join("hf_quant_config.json");
    let weights = match args.weights_digest.as_str() {
        "skip" => w::WeightsDigest::NotPinned,
        "full" => {
            let mut shards = Vec::new();
            for e in std::fs::read_dir(dir)? {
                let p = e?.path();
                if p.extension().is_some_and(|x| x == "safetensors") {
                    shards.push(w::ShardDigest {
                        name: p
                            .file_name()
                            .and_then(|n| n.to_str())
                            .context("shard name")?
                            .to_string(),
                        len: std::fs::metadata(&p)?.len(),
                        sha256: sha_file(&p)?,
                    });
                }
            }
            if shards.is_empty() {
                bail!("{} has no *.safetensors shard to pin", dir.display());
            }
            w::WeightsDigest::of_shards(shards)
        }
        other => bail!("--weights-digest {other} is not full or skip"),
    };
    Ok(w::CheckpointPin {
        repo: repo.to_string(),
        revision: args.checkpoint_revision.clone(),
        config_sha256: Sha256::of(&[&config]),
        quant_config_sha256: quant.exists().then(|| sha_file(&quant)).transpose()?,
        weights,
    })
}

fn hardware(
    class: &str,
    ptx_arch: &str,
    world_size: u32,
    min_visible_mib: u64,
) -> Result<w::HardwareSpec> {
    let text = HARDWARE
        .iter()
        .find(|(c, _)| *c == class)
        .map(|(_, t)| *t)
        .with_context(|| format!("no HARDWARE.toml embedded for class `{class}`"))?;
    let t: toml::Table = toml::from_str(text)?;
    let hw = t
        .get("hardware")
        .and_then(toml::Value::as_table)
        .context("HARDWARE.toml has no [hardware]")?;
    let s = |k: &str| {
        hw.get(k)
            .and_then(toml::Value::as_str)
            .with_context(|| format!("[hardware].{k}"))
    };
    if s("arch")? != ptx_arch {
        bail!(
            "HARDWARE.toml declares {} but the embedded target was compiled for {ptx_arch}",
            s("arch")?
        );
    }
    Ok(w::HardwareSpec {
        class: class.to_string(),
        ptx_arch: ptx_arch.to_string(),
        vendor: w::Vendor::parse(s("vendor")?).context("unknown [hardware].vendor")?,
        sm_count: u32::try_from(
            hw.get("sm_count")
                .and_then(toml::Value::as_integer)
                .context("[hardware].sm_count")?,
        )?,
        world_size,
        min_visible_mib,
        devices: vec![class.to_string()],
    })
}

/// 2026-10-22: The embedded kernel target the instance names (`hw/model/quant`).
fn target_set(inst: &Instance) -> Result<metrale_kernels::TargetPtxSet> {
    let mut seg = inst.target.split('/');
    let (Some(_class), Some(model), Some(quant)) = (seg.next(), seg.next(), seg.next()) else {
        bail!("instance target `{}` is not hw/model/quant", inst.target);
    };
    metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == model && s.target.quant == quant)
        .with_context(|| format!("this binary embeds no kernel target for {}; build it with METRALE_TARGET_MODEL={model}", inst.target))
}

fn kernels(
    inst: &Instance,
    set: &metrale_kernels::TargetPtxSet,
    plans: &[FusionPlan],
) -> Result<w::KernelSet> {
    let mut modules: Vec<w::ModuleSpec> = set
        .modules
        .iter()
        .map(|(name, bytes)| w::ModuleSpec {
            name: name.to_string(),
            kind: if metrale_gpu_runtime::fatbin::is_fatbin(bytes) {
                w::ArtifactKind::NvidiaFatbin
            } else {
                w::ArtifactKind::PtxText
            },
            arch: set.ptx_arch.to_string(),
            artifact: ArtifactId::of(bytes),
            len: bytes.len() as u64,
            entries: ptx_entries(bytes),
        })
        .collect();
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    let families = super::circuit::families_for(inst)?;
    let mut ids: Vec<KernelId> = plans
        .iter()
        .flat_map(|p| p.groups.iter().flat_map(|g| g.kernels.clone()))
        .collect();
    ids.sort();
    ids.dedup();
    let mut specs = Vec::with_capacity(ids.len());
    for id in ids {
        let (i, m) = modules
            .iter()
            .enumerate()
            .find(|(_, m)| m.name == id.module)
            .with_context(|| {
                format!(
                    "{}: a plan names {id}, but no embedded module is `{}`",
                    metrale_error::ErrorCode::KernelMissing.name(),
                    id.module
                )
            })?;
        if !m.entries.contains(&id.func) {
            bail!(
                "{}: a plan names {id}, but module `{}` has no entry `{}`",
                metrale_error::ErrorCode::KernelMissing.name(),
                m.name,
                id.func
            );
        }
        specs.push(w::KernelSpec {
            module: u32::try_from(i).context("a module index exceeds u32")?,
            family: families
                .families
                .iter()
                .find(|f| f.kernels.contains(&id))
                .map(|f| f.id.clone()),
            compute: families.compute_of(&id).cloned(),
            id,
        });
    }
    Ok(w::KernelSet {
        target: set.target.to_owned_strings(),
        modules,
        specs,
    })
}

/// 2026-10-22: Every `(mode, rows)` arm the instance lists, fused under `avail`. An arm that does not fuse is an
/// error for a golden instance (the executor runs the plan) and a recorded gap for a legacy one
/// (its engine runs the arm outside the plan; DESIGN.md §6).
pub(crate) fn fuse_plans(
    inst: &Instance,
    loaded: &metrale_circuit::Loaded,
    avail: &AvailableKernels,
) -> Result<(Vec<FusionPlan>, Vec<w::PlanGap>)> {
    let mut plans = Vec::new();
    let mut gaps = Vec::new();
    for (mode, rows) in &inst.plans {
        for &r in rows {
            match metrale_circuit::fuse(
                &loaded.circuit,
                &loaded.rules,
                avail,
                &inst.policy,
                *mode,
                r,
            ) {
                Ok(p) => plans.push(p),
                Err(e) if !inst.golden => gaps.push(w::PlanGap {
                    mode: *mode,
                    rows: r,
                    reason: e.to_string(),
                }),
                Err(e) => {
                    return Err(e).with_context(|| format!("fusing {} at {r} rows", mode.name()));
                }
            }
        }
    }
    Ok((plans, gaps))
}

/// 2026-10-22: Build the unsigned lite blueprint `args` describe.
pub(crate) fn build(args: &BuildArgs) -> Result<(Blueprint, Unsigned)> {
    let recipe = recipe_at(&args.recipe)?;
    let proj = recipe_view::project(&parts_of(&recipe)?)?;
    let inst = super::circuit::instance(&recipe.id)?;
    let src = super::circuit::sources(&inst)?;
    let loaded = metrale_circuit::load(&inst, src)?;
    let set = target_set(&inst)?;
    // 2026-10-22: The kernels the target's modules define, as the executor reads them at boot
    // (`circuit_exec::kernels::available_in`), so the plans are the ones this binary runs.
    let avail =
        metrale_model_layers::circuit_exec::kernels::available_in(&loaded.rules, &set.modules)?;
    let (plans, plan_gaps) = fuse_plans(&inst, &loaded, &avail)?;
    let kernels = kernels(&inst, &set, &plans)?;
    let class = inst.target.split('/').next().unwrap_or_default();
    let ptx_arch = kernels
        .modules
        .first()
        .map_or("", |m| m.arch.as_str())
        .to_string();
    let mut policy = proj.policy;
    policy.chat_template = match proj.resources.flags.get("chat_template") {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading the chat template {path}"))?;
            Some(w::TemplatePin {
                sha256: Sha256::of(&[text.as_bytes()]),
                text,
            })
        }
        None => None,
    };
    let name = recipe.id.rsplit('/').next().unwrap_or_default().to_string();
    let bp = Blueprint {
        schema: metrale_blueprint::BODY_SCHEMA,
        lineage: LineageId::of(&proj.model, class, &name),
        revision: args.blueprint_revision,
        name,
        engine: w::EnginePin {
            version: super::METRALE_VERSION.to_string(),
            digest_schema: metrale_circuit::digest::DIGEST_SCHEMA,
            rules_sha256: Sha256::parse_hex(&loaded.rules_digest).context("rules digest")?,
        },
        checkpoint: checkpoint(args, &proj.model, &inst)?,
        hardware: hardware(class, &ptx_arch, proj.world_size, args.min_visible_mib)?,
        circuit: loaded.circuit.clone(),
        precision: w::PrecisionPin {
            spec: inst.precision.clone(),
            table_sha256: Sha256::of(&[src.precision.as_bytes()]),
        },
        plans,
        plan_gaps,
        kernels,
        settings: proj.settings,
        policy,
        evidence: Vec::new(),
        execution: if inst.golden {
            w::Execution::Executor
        } else {
            w::Execution::Legacy {
                family: inst.arch.clone(),
            }
        },
    };
    let unsigned = Unsigned {
        sources: bp
            .kernels
            .modules
            .iter()
            .map(|_| KernelSource::Internal)
            .collect(),
        resources: proj.resources,
    };
    Ok((bp, unsigned))
}

#[cfg(test)]
#[path = "blueprint_tests.rs"]
pub(crate) mod blueprint_tests;
