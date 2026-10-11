// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: What the `met circuit venn` tests share: the checked-out tree as a [`Repo`], the
//! Lightning report's arguments, the built report and lookups into it, and a compact Lightning
//! checkpoint declaration.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use metrale_circuit::Mode;
use metrale_circuit::venn::report::{Row, Side, VennInputs};
use metrale_circuit::venn::{self, Finding, ParamKind, Repo, VennArgs, VennReport};

use crate::common;

pub const LIGHTNING: &str = "nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4";
pub const MOE: &str = "qwen3.6/qwen3.6-35b-a3b-fp8-bf16head";
pub const DENSE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
pub const REPORT: &str = "kernels/circuits/venn/nemotron-3.5-lightning-vs-qwen3.6-35b-a3b.md";
pub const MANIFEST: &str = "kernels/gb10/common/KERNEL_FAMILIES.toml";

/// 2026-09-29: The checked-out tree, as the CLI's file-system repo reads it.
pub struct Tree;

impl Repo for Tree {
    fn read(&self, rel: &str) -> Result<String, String> {
        std::fs::read_to_string(common::root().join(rel)).map_err(|e| format!("{rel}: {e}"))
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        let root = common::root();
        let mut out = Vec::new();
        let mut stack = vec![root.join(rel)];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)
                .map_err(|e| e.to_string())?
                .flatten()
            {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(r) = path.strip_prefix(&root) {
                    // 2026-10-07: The manifest and the CLI list use `/`. A Windows
                    // `strip_prefix` yields `\`, which the drift check then calls missing.
                    out.push(r.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        Ok(out)
    }
}

pub fn args() -> VennArgs {
    VennArgs {
        target: LIGHTNING.into(),
        against: vec![MOE.into(), DENSE.into()],
        modes: Mode::ALL.to_vec(),
        rows: vec![1, 16, 128],
        verify_rows: vec![2],
        out: REPORT.into(),
    }
}

pub fn families() -> venn::Families {
    venn::parse_families(&common::read(MANIFEST)).expect("manifest")
}

pub fn report() -> VennReport {
    let a = args();
    let all = common::instances();
    let find = |r: &str| {
        all.iter()
            .find(|i| i.recipe == r)
            .cloned()
            .expect("instance")
    };
    let (t, m, d) = (find(LIGHTNING), find(MOE), find(DENSE));
    let (lt, lm, ld) = (common::load(&t), common::load(&m), common::load(&d));
    let fams = families();
    let meas = venn::parse_measurements(&common::read("docs/kernel-perf/measurements.toml"))
        .expect("measurements");
    venn::build(&VennInputs {
        target: Side {
            instance: &t,
            loaded: &lt,
        },
        against: vec![
            Side {
                instance: &m,
                loaded: &lm,
            },
            Side {
                instance: &d,
                loaded: &ld,
            },
        ],
        families: &fams,
        measurements: &meas,
        runs: a.runs().expect("runs"),
        command: a.command(),
    })
    .expect("report")
}

pub fn row<'r>(r: &'r VennReport, mode: Mode, rows: u64, site: &str) -> &'r Row {
    let t = r
        .tables
        .iter()
        .find(|t| t.run.mode == mode && t.run.rows == rows)
        .unwrap_or_else(|| panic!("no table {} n={rows}", mode.name()));
    t.rows
        .iter()
        .find(|x| x.site == site)
        .unwrap_or_else(|| panic!("no row {site} in {} n={rows}", mode.name()))
}

/// 2026-09-29: The finding of `family` in a row, primary or secondary.
pub fn finding<'r>(row: &'r Row, family: &str) -> &'r Finding {
    row.primary
        .iter()
        .chain(&row.also)
        .find(|f| f.family == family)
        .unwrap_or_else(|| panic!("{}: no {family} finding", row.site))
}

pub fn diff(f: &Finding, param: &str) -> (String, String, ParamKind) {
    let d = f
        .diffs
        .iter()
        .find(|d| d.param == param)
        .unwrap_or_else(|| panic!("{}: no {param} difference in {:?}", f.family, f.diffs));
    (d.target.clone(), d.other.clone(), d.kind)
}

/// 2026-09-29: A compact `config.json` and `hf_quant_config.json` declaring what the Lightning
/// checkpoint declares (the full files are 1.3 MB and 0.9 MB of per-expert entries; every
/// expert shares one declaration, and the check asks expert 0).
pub fn lightning_checkpoint() -> (serde_json::Value, serde_json::Value) {
    let all = common::instances();
    let inst = all
        .iter()
        .find(|i| i.recipe == LIGHTNING)
        .expect("instance");
    let mut kinds = Vec::new();
    let mut layers = serde_json::Map::new();
    let fp4 = serde_json::json!({ "quant_algo": "W4A16_NVFP4", "group_size": 16 });
    for (i, k) in inst.shape.layer_kinds.iter().enumerate() {
        let l = format!("backbone.layers.{i}.mixer");
        kinds.push(match k.name() {
            "full_attention" => "attention",
            other => other,
        });
        match k.name() {
            "mamba" => {
                for p in ["in_proj", "out_proj"] {
                    layers.insert(
                        format!("{l}.{p}"),
                        serde_json::json!({ "quant_algo": "FP8" }),
                    );
                }
            }
            "moe" => {
                for p in [
                    "experts.0.up_proj",
                    "experts.0.down_proj",
                    "shared_experts.up_proj",
                    "shared_experts.down_proj",
                ] {
                    layers.insert(format!("{l}.{p}"), fp4.clone());
                }
            }
            _ => {}
        }
    }
    layers.insert("lm_head".into(), fp4);
    let config = serde_json::json!({ "model_type": "nemotron_h", "layers_block_type": kinds });
    let hfq = serde_json::json!({ "quantization": {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": "FP8",
        "quantized_layers": layers,
        "exclude_modules": ["backbone.embeddings", "mtp*"],
    }});
    (config, hfq)
}
