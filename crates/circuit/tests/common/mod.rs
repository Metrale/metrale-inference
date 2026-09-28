// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The file side of the golden-plan tests: reading the repo's circuit TOMLs, and
//! the kernels each instance's target compiles. The crate under test does no I/O; this module
//! is its caller.
//!
//! Owner: metrale-circuit tests.
//! Invariants:
//! - Paths are relative to the workspace root, found from `CARGO_MANIFEST_DIR`.
//! - A kernel is available to a target when its module (the source stem after the target's
//!   KERNEL.toml `[modules]` renames, applied least specific first as the kernels build does)
//!   is one of the target's sources and that source names the function as a whole word.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use metrale_circuit::{AvailableKernels, Instance, KernelId, Loaded, Rule, Sources};

/// 2026-09-28: The workspace root.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// 2026-09-28: Read a repo-relative file.
pub fn read(rel: &str) -> String {
    let p = root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// 2026-09-28: The plans directory.
pub fn plans_dir() -> PathBuf {
    root().join("kernels/circuits/plans")
}

/// 2026-09-28: FUSIONS.toml of the instance's hardware.
pub fn fusions_rel(instance: &Instance) -> String {
    let hw = instance.target.split('/').next().expect("target hw");
    format!("kernels/{hw}/common/FUSIONS.toml")
}

/// 2026-09-28: All instances.
pub fn instances() -> Vec<Instance> {
    metrale_circuit::parse_instances(&read("kernels/circuits/INSTANCES.toml")).expect("instances")
}

/// 2026-09-28: The instance's circuit, block library, precision and rule texts.
pub struct Texts {
    pub circuit: String,
    pub blocks: Vec<(String, String)>,
    pub precision: String,
    pub rules: String,
}

impl Texts {
    pub fn of(instance: &Instance) -> Self {
        let circuit = read(&format!("kernels/circuits/{}.toml", instance.arch));
        let blocks = metrale_circuit::includes_of(&circuit)
            .expect("circuit parses")
            .into_iter()
            .map(|n| {
                let text = read(&format!("kernels/circuits/blocks/{n}.toml"));
                (n, text)
            })
            .collect();
        Texts {
            circuit,
            blocks,
            precision: read(&format!(
                "kernels/circuits/precision/{}.toml",
                instance.precision
            )),
            rules: read(&fusions_rel(instance)),
        }
    }
}

/// 2026-09-28: Load `t` for `instance`.
pub fn load_texts(instance: &Instance, t: &Texts) -> Result<Loaded, metrale_circuit::LoadError> {
    let blocks: Vec<(&str, &str)> = t
        .blocks
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_str()))
        .collect();
    metrale_circuit::load(
        instance,
        Sources {
            circuit: &t.circuit,
            blocks: &blocks,
            precision: &t.precision,
            rules: &t.rules,
        },
    )
}

/// 2026-09-28: Load an instance from the repo.
pub fn load(instance: &Instance) -> Loaded {
    load_texts(instance, &Texts::of(instance))
        .unwrap_or_else(|e| panic!("{}: {e}", instance.recipe))
}

/// 2026-09-28: Module name -> source text, for the instance's target.
pub fn target_modules(instance: &Instance) -> BTreeMap<String, String> {
    let mut parts = instance.target.split('/');
    let target = metrale_closure::layout::Target {
        hardware: parts.next().expect("hw").to_string(),
        model: parts.next().expect("model").to_string(),
        quant: parts.next().expect("quant").to_string(),
    };
    let layout = metrale_closure::layout::discover(&root(), &target)
        .unwrap_or_else(|e| panic!("{}: {e:?}", instance.target));
    let mut renames: BTreeMap<String, String> = BTreeMap::new();
    for manifest in layout.configs() {
        let value: toml::Table =
            toml::from_str(&std::fs::read_to_string(&manifest).expect("KERNEL.toml"))
                .expect("KERNEL.toml parses");
        if let Some(t) = value.get("modules").and_then(|m| m.as_table()) {
            for (stem, name) in t {
                renames.insert(
                    stem.clone(),
                    name.as_str().expect("module name").to_string(),
                );
            }
        }
    }
    layout
        .modules()
        .into_iter()
        .map(|(stem, entry)| {
            let name = renames.get(&stem).cloned().unwrap_or(stem);
            let text = std::fs::read_to_string(&entry.source).expect("kernel source");
            (name, text)
        })
        .collect()
}

fn names_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let ok = |c: Option<char>| c.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
        ok(text[..at].chars().next_back()) && ok(text[at + word.len()..].chars().next())
    })
}

/// 2026-09-28: Whether `k` is compiled into a target whose modules are `modules`.
pub fn present(modules: &BTreeMap<String, String>, k: &KernelId) -> bool {
    modules
        .get(&k.module)
        .is_some_and(|t| names_word(t, &k.func))
}

/// 2026-09-28: The kernels of `rules` the instance's target compiles.
pub fn available(instance: &Instance, rules: &[Rule]) -> AvailableKernels {
    let modules = target_modules(instance);
    let mut out = AvailableKernels::default();
    for r in rules {
        for k in &r.kernels {
            if present(&modules, k) {
                out.kernels.insert(k.clone());
            }
        }
        out.caps.extend(r.requires.iter().cloned());
    }
    out
}

/// 2026-09-28: The display snapshots: decode n1 of every golden instance, Unicode at 80 and
/// 120 columns and ASCII at 80.
pub const DISPLAY_SNAPSHOTS: [(&str, metrale_circuit::display::Glyphs, usize); 3] = [
    ("display-w80", metrale_circuit::display::Glyphs::Unicode, 80),
    (
        "display-w120",
        metrale_circuit::display::Glyphs::Unicode,
        120,
    ),
    (
        "display-ascii-w80",
        metrale_circuit::display::Glyphs::Ascii,
        80,
    ),
];

/// 2026-09-28: Draw one plan of `inst` as plain text.
pub fn display_text(
    inst: &Instance,
    loaded: &Loaded,
    mode: metrale_circuit::Mode,
    rows: u64,
    opts: metrale_circuit::display::DisplayOpts,
) -> String {
    let avail = available(inst, &loaded.rules);
    metrale_circuit::display_plan(inst, loaded, &avail, mode, rows, &opts)
        .unwrap_or_else(|e| panic!("{} display: {e}", inst.recipe))
        .plain()
}

/// 2026-09-28: Every golden plan: (file name, rendered text).
pub fn golden_plans() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for inst in instances().iter().filter(|i| i.golden) {
        let loaded = load(inst);
        let avail = available(inst, &loaded.rules);
        for (&mode, rows) in &inst.plans {
            for &n in rows {
                let text = metrale_circuit::render_plan(inst, &loaded, &avail, mode, n)
                    .unwrap_or_else(|e| panic!("{} {} n={n}: {e}", inst.recipe, mode.name()));
                out.push((inst.plan_file(mode, n), text));
            }
        }
        for (tag, glyphs, width) in DISPLAY_SNAPSHOTS {
            let opts = metrale_circuit::display::DisplayOpts {
                width,
                glyphs,
                expand: metrale_circuit::display::Expand::Summary,
            };
            let text = display_text(inst, &loaded, metrale_circuit::Mode::Decode, 1, opts);
            out.push((format!("{}-decode-n1.{tag}.txt", inst.arch), text));
        }
    }
    out
}
