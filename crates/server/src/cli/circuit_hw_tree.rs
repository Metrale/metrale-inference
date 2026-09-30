// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The kernel tree on disk for `met circuit plan`: a class's compiled modules,
//! resolved by the same layout the kernels build uses (`metrale_closure::layout`), with the
//! KERNEL.toml `[modules]` renames applied least specific first, and the MODEL.toml
//! `[expected_absent]` entries of the target.
//!
//! Owner: server CLI.
//! Invariants:
//! - A class with no target for the model resolves its common layer only, from one of its
//!   targets (the common layer is the same for every target of a class); the planner reports it.
//! - Nothing here decides availability: `metrale_circuit::hardware::avail` does.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use metrale_circuit::hardware::{ClassSources, KernelTree, Module};
use metrale_circuit::venn::Repo;
use metrale_closure::layout::{self, Target};
use metrale_kernels::ModelTypeMatch;
use metrale_kernels::resolve::{ResolveCandidate, resolve_target};

use super::circuit_venn::FsRepo;

/// 2026-09-30: The repository on disk, with kernel-class resolution.
pub(crate) struct FsTree {
    pub(crate) repo: FsRepo,
}

impl FsTree {
    /// 2026-09-30: The tree rooted at `root`.
    pub(crate) fn new(root: PathBuf) -> Self {
        FsTree {
            repo: FsRepo { root },
        }
    }

    fn rel(&self, p: &Path) -> Result<String, String> {
        p.strip_prefix(&self.repo.root)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .map_err(|_| format!("{} is outside the repository", p.display()))
    }
}

impl Repo for FsTree {
    fn read(&self, rel: &str) -> Result<String, String> {
        self.repo.read(rel)
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        self.repo.list(rel)
    }
}

fn renames(configs: &[PathBuf]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for manifest in configs {
        let text = std::fs::read_to_string(manifest)
            .map_err(|e| format!("{}: {e}", manifest.display()))?;
        let value: toml::Table =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
        if let Some(t) = value.get("modules").and_then(|m| m.as_table()) {
            for (stem, name) in t {
                let name = name.as_str().ok_or_else(|| {
                    format!("{}: [modules] {stem} is not a string", manifest.display())
                })?;
                out.insert(stem.clone(), name.to_string());
            }
        }
    }
    Ok(out)
}

fn expected_absent(
    model_toml: &Path,
) -> Result<BTreeMap<String, BTreeMap<String, String>>, String> {
    let Ok(text) = std::fs::read_to_string(model_toml) else {
        return Ok(BTreeMap::new());
    };
    let value: toml::Table =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", model_toml.display()))?;
    let mut out = BTreeMap::new();
    if let Some(t) = value.get("expected_absent").and_then(|v| v.as_table()) {
        for (module, funcs) in t {
            let funcs = funcs.as_table().ok_or_else(|| {
                format!(
                    "{}: [expected_absent.{module}] is not a table",
                    model_toml.display()
                )
            })?;
            let m = funcs
                .iter()
                .map(|(f, why)| {
                    why.as_str()
                        .map(|w| (f.clone(), w.trim().to_string()))
                        .ok_or_else(|| {
                            format!(
                                "{}: [expected_absent.{module}] {f} gives no reason",
                                model_toml.display()
                            )
                        })
                })
                .collect::<Result<BTreeMap<String, String>, String>>()?;
            out.insert(module.clone(), m);
        }
    }
    Ok(out)
}

impl KernelTree for FsTree {
    fn class_sources(&self, class: &str, model: &str, quant: &str) -> Result<ClassSources, String> {
        let root = &self.repo.root;
        let want = Target {
            hardware: class.to_string(),
            model: model.to_string(),
            quant: quant.to_string(),
        };
        // 2026-09-30: The target itself when the class resolves it (a model directory, or a
        // MODEL.toml `kernel_source` redirect); otherwise any target of the class, for its
        // common layer only.
        let (own, target, lay) = match layout::discover(root, &want) {
            Ok(lay) => (true, want, lay),
            Err(_) => {
                let any = layout::walk(root)
                    .map_err(|e| format!("{e:?}"))?
                    .into_iter()
                    .find(|t| t.hardware == class)
                    .ok_or_else(|| format!("kernels/{class} has no kernel target at all"))?;
                let lay = layout::discover(root, &any).map_err(|e| format!("{any}: {e:?}"))?;
                (false, any, lay)
            }
        };
        let configs: Vec<PathBuf> = lay
            .configs()
            .into_iter()
            .filter(|c| own || c.components().any(|x| x.as_os_str() == "common"))
            .collect();
        let names = renames(&configs)?;
        let mut modules = BTreeMap::new();
        for (stem, entry) in lay.modules() {
            if !own
                && lay
                    .common
                    .get(&entry.name)
                    .is_none_or(|e| e.source != entry.source)
            {
                continue;
            }
            let text = std::fs::read_to_string(&entry.source)
                .map_err(|e| format!("{}: {e}", entry.source.display()))?;
            let name = names.get(&stem).cloned().unwrap_or(stem);
            modules.insert(
                name,
                Module {
                    path: self.rel(&entry.source)?,
                    text,
                },
            );
        }
        let mut files = BTreeSet::new();
        for (entries, _) in [lay.role(layout::Role::Common), lay.role(layout::Role::Leaf)]
            .into_iter()
            .take(if own { 2 } else { 1 })
        {
            for e in entries.values() {
                files.insert(self.rel(&e.source)?);
            }
        }
        let absent = if own {
            expected_absent(&lay.model_dir.join("MODEL.toml"))?
        } else {
            BTreeMap::new()
        };
        Ok(ClassSources {
            class: class.to_string(),
            target: own.then(|| target.to_string()),
            modules,
            files,
            expected_absent: absent,
        })
    }

    fn kernel_target(&self, config_json: &str, refs: &[&str]) -> Result<Option<String>, String> {
        // 2026-09-30: The serve path's inputs (serve_load/model_setup.rs): the engine's config
        // parse, which rewrites some model_types (crates/config/src/dispatch.rs), not the raw
        // config.json field.
        let cfg = metrale_config::parse_config(config_json).map_err(|e| format!("{e:#}"))?;
        let owned = model_targets(&self.repo.root)?;
        let needles: Vec<Vec<&str>> = owned
            .iter()
            .map(|t| t.match_names.iter().map(String::as_str).collect())
            .collect();
        let candidates: Vec<ResolveCandidate<'_>> = owned
            .iter()
            .zip(&needles)
            .map(|(t, n)| ResolveCandidate {
                name: &t.name,
                type_matches: &t.types,
                match_names: n,
            })
            .collect();
        resolve_target(&candidates, &cfg.model_type, cfg.hidden_size, refs)
            .map(|i| i.map(|i| owned[i].name.clone()))
            .map_err(|e| e.to_string())
    }

    fn as_repo(&self) -> &dyn Repo {
        self
    }
}

/// 2026-09-30: One gb10 kernel target's resolution keys, from its MODEL.toml.
struct ModelTarget {
    name: String,
    types: Vec<ModelTypeMatch>,
    match_names: Vec<String>,
}

/// 2026-09-30: Every `kernels/gb10/<model>/MODEL.toml` (the reference class names the targets;
/// the other classes reuse the directory names). `ModelTypeMatch` holds `&'static str` because
/// the build bakes it; the CLI leaks the few names it parses, once per call.
fn model_targets(root: &Path) -> Result<Vec<ModelTarget>, String> {
    let dir = root.join("kernels/gb10");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .filter(|e| e.path().join("MODEL.toml").is_file())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let p = dir.join(&name).join("MODEL.toml");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let t: toml::Table = toml::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
        let types = t
            .get("model_types")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .map(|m| {
                let model_type = m
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        format!("{}: [[model_types]] without model_type", p.display())
                    })?;
                let hidden_size = match m.get("hidden_size") {
                    None => None,
                    Some(v) => Some(
                        v.as_integer()
                            .and_then(|h| usize::try_from(h).ok())
                            .ok_or_else(|| format!("{}: hidden_size is not a size", p.display()))?,
                    ),
                };
                Ok(ModelTypeMatch {
                    model_type: model_type.to_string().leak(),
                    hidden_size,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let match_names = t
            .get("model")
            .and_then(|m| m.get("match_names"))
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        out.push(ModelTarget {
            name,
            types,
            match_names,
        });
    }
    Ok(out)
}
