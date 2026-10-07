// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The strix-hip (gfx1151) GPT-OSS-20B MXFP4 target, checked
//! through `metrale_closure::layout`, the resolver build.rs compiles from.
//!
//! Owner: metrale-kernels tests.
//! Invariants checked (host-only; nothing here compiles or launches a kernel):
//! - Every kernel lookup in the GPT-OSS runtime either resolves to an entry
//!   point the target compiles, under the module name the build gives it, or
//!   is declared in MODEL.toml `[expected_absent]`; no declaration is stale.
//! - Paged decode attention is the HDIM 64 wrapper, as a declared shadow.
//! - Only `mxfp4` resolves; other quants refuse the common-only fallback.
//! - No resolved source defines the strix-hip/common no-op stub entry points.
//! - The Qwen strix-hip targets resolve none of the GPT-only files.
//!
//! Usage: `METRALE_SKIP_BUILD=1 cargo test -p metrale-kernels --test strix_hip_gpt_oss`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use metrale_closure::layout::{Layout, Role, Target, discover, walk};

/// 2026-10-07: Entry points strix-hip/common defines as no-op compile stubs.
const STUBS: [&str; 4] = [
    "moe_w4a16_grouped_gemm",
    "moe_w4a16_grouped_gemm_ptrtable",
    "moe_w4a16_grouped_gemm_ptrtable_t",
    "moe_fp8_grouped_gemm_v2",
];

/// 2026-10-07: Sources only the GPT-OSS strix-hip leaf may bring in.
const GPT_ONLY: [&str; 6] = [
    "kernels/gb10/common/dense_gemv_bf16_batchm.cu",
    "kernels/gb10/common/gpt_oss_expert_ops.cu",
    "kernels/gb10/common/gpt_oss_mxfp4_gemv.cu",
    "kernels/gb10/common/gpt_oss_rope.cu",
    "kernels/gb10/common/projection_bias.cu",
    "kernels/gb10/gpt-oss-20b/mxfp4/paged_decode_attn.cu",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn target(model: &str, quant: &str) -> Target {
    Target {
        hardware: "strix-hip".into(),
        model: model.into(),
        quant: quant.into(),
    }
}

fn gpt() -> Layout {
    discover(&root(), &target("gpt-oss-20b", "mxfp4")).expect("strix-hip GPT-OSS resolves")
}

/// 2026-10-07: `extern "C" __global__` names a text defines, skipping
/// `__launch_bounds__(...)` between the return type and the name.
fn entry_points(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (at, _) in text.match_indices("__global__") {
        let mut rest = &text[at + "__global__".len()..];
        while let Some(open) = rest.find('(') {
            let name = rest[..open]
                .split_whitespace()
                .last()
                .unwrap_or_default()
                .trim_start_matches('*');
            if name == "__launch_bounds__" {
                let close = rest[open..].find(')').map_or(rest.len(), |c| open + c + 1);
                rest = &rest[close..];
                continue;
            }
            if !name.is_empty() {
                out.insert(name.to_string());
            }
            break;
        }
    }
    out
}

/// 2026-10-07: The text a module compiles: its source plus quoted includes,
/// resolved as the build stages them (`../../common/x` is the target's common
/// role, a bare name is the including file's role).
fn compiled_text(layout: &Layout, role: Role, source: &Path, depth: usize) -> String {
    let mut text = std::fs::read_to_string(source)
        .unwrap_or_else(|e| panic!("read {}: {e}", source.display()));
    assert!(depth < 8, "include chain too deep at {}", source.display());
    let includes: Vec<String> = text
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#include \""))
        .filter_map(|l| l.split('"').next())
        .map(str::to_string)
        .collect();
    for inc in includes {
        let (inc_role, name) = match inc.strip_prefix("../../common/") {
            Some(name) => (Role::Common, name.to_string()),
            None => (role, inc.clone()),
        };
        let (entries, _) = layout.role(inc_role);
        let path = entries
            .get(&name)
            .map(|e| e.source.clone())
            .unwrap_or_else(|| source.parent().unwrap().join(&inc));
        text.push_str(&compiled_text(layout, inc_role, &path, depth + 1));
    }
    text
}

/// 2026-10-07: Module name -> entry points, with `[modules]` renames merged
/// least specific first, as build.rs merges them.
fn compiled_entries(layout: &Layout) -> BTreeMap<String, BTreeSet<String>> {
    let mut renames = BTreeMap::new();
    for config in layout.configs() {
        let manifest: toml::Value =
            toml::from_str(&std::fs::read_to_string(config).unwrap()).expect("KERNEL.toml parses");
        if let Some(table) = manifest.get("modules").and_then(|m| m.as_table()) {
            for (stem, module) in table {
                renames.insert(stem.clone(), module.as_str().unwrap().to_string());
            }
        }
    }
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (stem, entry) in layout.modules() {
        let role = layout.layers[entry.layer].role;
        let module = renames.get(&stem).cloned().unwrap_or(stem);
        let text = compiled_text(layout, role, &entry.source, 0);
        out.entry(module).or_default().extend(entry_points(&text));
    }
    out
}

/// 2026-10-07: Every `.kernel("module", "entry")` the GPT-OSS runtime issues.
fn runtime_lookups() -> BTreeSet<(String, String)> {
    let dir = root().join("crates/model-arch/src/weight_loader/gpt_oss/runtime");
    let mut out = BTreeSet::new();
    for file in std::fs::read_dir(&dir).unwrap().flatten() {
        let path = file.path();
        if path.file_name().is_some_and(|n| n == "tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for (at, _) in text.match_indices(".kernel(") {
            let args: Vec<&str> = text[at..].split('"').skip(1).step_by(2).take(2).collect();
            out.insert((args[0].to_string(), args[1].to_string()));
        }
    }
    out
}

fn expected_absent() -> BTreeSet<(String, String)> {
    let path = root().join("kernels/strix-hip/gpt-oss-20b/MODEL.toml");
    let manifest: toml::Value = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut out = BTreeSet::new();
    if let Some(table) = manifest.get("expected_absent").and_then(|t| t.as_table()) {
        for (module, kernels) in table {
            for kernel in kernels.as_table().unwrap().keys() {
                out.insert((module.clone(), kernel.clone()));
            }
        }
    }
    out
}

#[test]
fn every_gpt_runtime_lookup_resolves_or_is_declared_absent() {
    let lookups = runtime_lookups();
    for known in [
        ("gpt_oss_rope", "gpt_oss_rope_bf16"),
        ("paged_decode", "paged_decode_attn_sink"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_fp32out"),
    ] {
        let pair = (known.0.to_string(), known.1.to_string());
        assert!(lookups.contains(&pair), "lookup scan missed {pair:?}");
    }
    let entries = compiled_entries(&gpt());
    let absent = expected_absent();
    let resolves = |(m, k): &(String, String)| entries.get(m).is_some_and(|e| e.contains(k));
    let missing: Vec<_> = lookups
        .iter()
        .filter(|l| !resolves(l) && !absent.contains(*l))
        .collect();
    assert!(missing.is_empty(), "unresolved, undeclared: {missing:?}");
    for declared in &absent {
        assert!(lookups.contains(declared), "stale declaration {declared:?}");
        assert!(
            !resolves(declared),
            "declared absent but resolves: {declared:?}"
        );
    }
}

#[test]
fn paged_decode_is_the_hd64_wrapper_as_a_declared_shadow() {
    let layout = gpt();
    let root = root();
    let wrapper = root.join("kernels/gb10/gpt-oss-20b/mxfp4/paged_decode_attn.cu");
    let (_, entry) = layout
        .modules()
        .into_iter()
        .find(|(stem, _)| stem == "paged_decode_attn")
        .expect("paged_decode_attn resolves");
    assert_eq!(entry.source, wrapper);
    assert_eq!(layout.layers[entry.layer].role, Role::Leaf);
    let text = std::fs::read_to_string(&wrapper).unwrap();
    assert!(text.contains("#define HDIM 64"));
    assert!(text.contains("../../common/paged_decode_attn.cu"));
    let shadow = layout
        .shadows
        .iter()
        .find(|s| s.name == "paged_decode_attn.cu")
        .expect("declared shadow");
    assert_eq!(
        shadow.loser,
        root.join("kernels/gb10/common/paged_decode_attn.cu")
    );
}

#[test]
fn only_mxfp4_resolves() {
    let root = root();
    for quant in ["nvfp4", "bf16", "fp8"] {
        assert!(
            discover(&root, &target("gpt-oss-20b", quant)).is_err(),
            "common-only fallback for {quant}"
        );
    }
    let quants: Vec<String> = walk(&root)
        .unwrap()
        .into_iter()
        .filter(|t| t.hardware == "strix-hip" && t.model == "gpt-oss-20b")
        .map(|t| t.quant)
        .collect();
    assert_eq!(quants, ["mxfp4"]);
}

#[test]
fn gpt_resolution_never_includes_the_common_stub_entry_points() {
    let root = root();
    // 2026-10-07: Positive control: the scan finds the stubs where they resolve.
    let stub_file = root.join("kernels/strix-hip/common/moe_w4a16_grouped_gemm.cu");
    let found = entry_points(&std::fs::read_to_string(&stub_file).unwrap());
    assert!(STUBS[..3].iter().all(|s| found.contains(*s)), "{found:?}");
    let qwen = discover(&root, &target("qwen3.6-27b", "nvfp4")).unwrap();
    let qwen_entries: BTreeSet<String> = compiled_entries(&qwen).into_values().flatten().collect();
    assert!(qwen_entries.contains("moe_fp8_grouped_gemm_v2"));

    let layout = gpt();
    let defined: BTreeSet<String> = compiled_entries(&layout).into_values().flatten().collect();
    for stub in STUBS {
        assert!(
            !defined.contains(stub),
            "GPT strix-hip resolves stub {stub}"
        );
    }
    let sources = layout.sources();
    for stub_source in ["moe_w4a16_grouped_gemm.cu", "moe_fp8_grouped_gemm.cu"] {
        let path = root.join("kernels/strix-hip/common").join(stub_source);
        assert!(
            !sources.contains(&path),
            "GPT strix-hip compiles {}",
            path.display()
        );
    }
}

#[test]
fn qwen_strix_hip_targets_resolve_no_gpt_only_file() {
    let root = root();
    let gpt_only: BTreeSet<PathBuf> = GPT_ONLY.iter().map(|p| root.join(p)).collect();
    let gpt_inputs = gpt().inputs();
    assert!(gpt_only.iter().all(|p| gpt_inputs.contains(p)));
    let qwen: Vec<Target> = walk(&root)
        .unwrap()
        .into_iter()
        .filter(|t| t.hardware == "strix-hip" && t.model.starts_with("qwen"))
        .collect();
    assert_eq!(qwen.len(), 2, "{qwen:?}");
    let gpt_dir = root.join("kernels/strix-hip/gpt-oss-20b");
    for t in qwen {
        let layout = discover(&root, &t).unwrap();
        for input in layout.inputs() {
            assert!(
                !gpt_only.contains(&input) && !input.starts_with(&gpt_dir),
                "{t} resolves GPT-only {}",
                input.display()
            );
        }
    }
}

/// 2026-10-07: hipcc has compiled a strix-hip source only if some strix-hip
/// target resolves it. Apart from its own GPT-OSS sources, the GPT target may
/// compile only what a Qwen strix-hip target already compiles.
#[test]
fn gpt_compiles_nothing_new_to_hipcc_but_its_own_sources() {
    let root = root();
    let mut proven: BTreeSet<PathBuf> = GPT_ONLY.iter().map(|p| root.join(p)).collect();
    for model in ["qwen3.6-27b", "qwen3.6-35b-a3b"] {
        proven.extend(discover(&root, &target(model, "nvfp4")).unwrap().sources());
    }
    let leaf = root.join("kernels/strix-hip/gpt-oss-20b/mxfp4");
    let layout = gpt();
    for (stem, entry) in layout.modules() {
        let source = &entry.source;
        if source.starts_with(&leaf) {
            let text = std::fs::read_to_string(source).unwrap();
            assert!(
                entry_points(&text).is_empty() && !text.contains("#include"),
                "{stem}: a GPT strix-hip leaf file is a fail-closed shadow"
            );
            continue;
        }
        assert!(
            proven.contains(source),
            "{stem}: {} has never been compiled for strix-hip",
            source.display()
        );
    }
}
