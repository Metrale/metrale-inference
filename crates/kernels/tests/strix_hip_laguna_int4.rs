// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The strix-hip (gfx1151, native HIP) Laguna-XS-2.1 INT4 target, checked
//! through `metrale_closure::layout`, the resolver build.rs compiles from.
//!
//! Owner: metrale-kernels tests.
//! Invariants checked (host-only; nothing here compiles or launches a kernel):
//! - The packed-int GEMV lookups (model-layers quant_format/packed_int.rs) resolve to the
//!   leaf's packed_int_gemv.cu under the module name the build gives it.
//! - Every other kernel lookup on Laguna's load path resolves, is declared in MODEL.toml
//!   `[expected_absent]`, or is in `GPU_GATE_GAPS` with its class; no entry is stale, and
//!   no lookup that fails its caller is a gap of a class Laguna's own path needs.
//! - Only `int4` resolves; other quants refuse the common-only fallback.
//! - No resolved source defines a strix-hip/common no-op stub entry point.
//! - No other target resolves a file of the Laguna strix-hip leaf, and the leaf adds no
//!   source hipcc has not compiled for strix-hip except its own packed-int GEMV.
//!
//! Usage: `METRALE_SKIP_BUILD=1 cargo test -p metrale-kernels --test strix_hip_laguna_int4`;
//! `-- --nocapture laguna_lookup_inventory` prints the classified lookup table.

#[allow(dead_code)]
#[path = "../build_shadow.rs"]
mod build_shadow;
#[path = "support/lookup_scan.rs"]
mod lookup_scan;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use metrale_closure::layout::{Layout, Target, discover, walk};

/// 2026-10-07: Entry points strix-hip/common defines as no-op compile stubs.
const STUBS: [&str; 4] = [
    "moe_w4a16_grouped_gemm",
    "moe_w4a16_grouped_gemm_ptrtable",
    "moe_w4a16_grouped_gemm_ptrtable_t",
    "moe_fp8_grouped_gemm_v2",
];

/// 2026-10-07: The packed-int GEMV entry points (packed_int_gemv.cu).
const PACKED_INT: [&str; 4] = [
    "packed_int4_gemv_g128",
    "packed_int8_gemv_g128",
    "moe_packed_int4_gemv_ptrtable_g128",
    "moe_packed_int8_gemv_ptrtable_g128",
];

/// 2026-10-07: The Rust files whose lookups run when a Laguna checkpoint loads: the Laguna
/// loader, the attention, dense-FFN and MoE layer constructors it calls, the model-level
/// kernels and head, and the packed-int GEMV lookup the INT4 expert path will call.
const LAGUNA_PATHS: [&str; 9] = [
    "crates/model-arch/src/weight_loader/laguna",
    "crates/model-layers/src/layers/qwen3_attention/",
    "crates/model-layers/src/layers/dense_ffn_init.rs",
    "crates/model-layers/src/layers/moe/init.rs",
    "crates/model-layers/src/quant_format/packed_int.rs",
    "crates/model-engine/src/model/impl_a1",
    "crates/model-engine/src/model/trait_impl/meta_argmax_masked.rs",
    "crates/model-engine/src/factory/lm_head_setup.rs",
    "crates/model-engine/src/model/impl_a1.rs",
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

fn laguna() -> Layout {
    discover(&root(), &target("laguna-xs-2.1", "int4")).expect("strix-hip Laguna resolves")
}

/// 2026-10-07: Module name -> entry points, with `[modules]` renames merged least specific
/// first, as build.rs merges them; entry points read by build.rs's own scanner.
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
        let module = renames.get(&stem).cloned().unwrap_or(stem);
        out.entry(module)
            .or_default()
            .extend(build_shadow::entry_points(&entry.source));
    }
    out
}

/// 2026-10-07: `(module, entry) -> (first site, required)` for the lookups on Laguna's load
/// path; `required` when some site is `.kernel(...)?`, which fails its caller on a miss.
fn laguna_lookups() -> BTreeMap<(String, String), (String, bool)> {
    let mut out: BTreeMap<(String, String), (String, bool)> = BTreeMap::new();
    for l in lookup_scan::lookups(&root()) {
        if !LAGUNA_PATHS.iter().any(|p| l.site.starts_with(p)) {
            continue;
        }
        for m in l.modules {
            let e = out
                .entry((m, l.func.clone()))
                .or_insert_with(|| (l.site.clone(), false));
            e.1 |= l.required;
        }
    }
    out
}

fn expected_absent() -> BTreeSet<(String, String)> {
    let path = root().join("kernels/strix-hip/laguna-xs-2.1/MODEL.toml");
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

#[path = "strix_hip_laguna_int4/gaps.rs"]
mod gaps;
use gaps::{GPU_GATE_GAPS, GapClass};

/// 2026-10-07: `(module, entry) -> class` over `GPU_GATE_GAPS`; an entry listed twice fails.
fn gap_index() -> BTreeMap<(String, String), GapClass> {
    let mut out = BTreeMap::new();
    for g in GPU_GATE_GAPS {
        for e in g.entries {
            let key = (g.module.to_string(), e.to_string());
            assert!(
                out.insert(key.clone(), g.class).is_none(),
                "{key:?} listed twice"
            );
        }
    }
    out
}

/// 2026-10-07: No lookup that fails its caller when missing is a gap of a class Laguna's
/// own path needs: every required unresolved lookup is an NVFP4 or FP8 expert kernel (the
/// NVFP4 MoE constructor, moe/init.rs) or behind a config feature Laguna does not select.
#[test]
fn required_gaps_are_only_foreign_formats_or_features() {
    let gaps = gap_index();
    for ((m, k), (site, required)) in laguna_lookups() {
        let Some(class) = gaps.get(&(m.clone(), k.clone())) else {
            continue;
        };
        if required {
            assert!(
                matches!(
                    class,
                    GapClass::Nvfp4Only | GapClass::Fp8WeightOnly | GapClass::OtherModelFeature
                ),
                "{m}::{k} ({site}) is required and {class:?}"
            );
        }
    }
}

#[test]
fn packed_int_gemv_lookups_resolve_to_the_leaf() {
    let lookups = laguna_lookups();
    let entries = compiled_entries(&laguna());
    let module = entries
        .get("packed_int_gemv")
        .expect("packed_int_gemv module");
    for entry in PACKED_INT {
        let key = ("packed_int_gemv".to_string(), entry.to_string());
        assert!(lookups.contains_key(&key), "lookup scan missed {key:?}");
        assert!(module.contains(entry), "{entry} not compiled");
    }
    let layout = laguna();
    let (_, gemv) = layout
        .modules()
        .into_iter()
        .find(|(stem, _)| stem == "packed_int_gemv")
        .expect("packed_int_gemv resolves");
    assert_eq!(
        gemv.source,
        root().join("kernels/strix-hip/laguna-xs-2.1/int4/packed_int_gemv.cu")
    );
}

#[test]
fn every_laguna_lookup_resolves_or_is_classified() {
    let lookups = laguna_lookups();
    for known in [
        ("quantize_nvfp4", "quantize_bf16_to_nvfp4"),
        ("paged_decode_fp8", "paged_decode_attn_splitk_fp8"),
        ("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable"),
    ] {
        let key = (known.0.to_string(), known.1.to_string());
        assert!(lookups.contains_key(&key), "lookup scan missed {key:?}");
    }
    let entries = compiled_entries(&laguna());
    let resolves = |(m, k): &(String, String)| entries.get(m).is_some_and(|e| e.contains(k));
    let absent = expected_absent();
    let gaps = gap_index();
    let missing: Vec<String> = lookups
        .iter()
        .filter(|(l, _)| !resolves(l) && !absent.contains(*l) && !gaps.contains_key(*l))
        .map(|((m, k), (site, _))| format!("{m}::{k} ({site})"))
        .collect();
    assert!(
        missing.is_empty(),
        "unresolved, unclassified:\n  {}",
        missing.join("\n  ")
    );
    for declared in absent.iter().chain(gaps.keys()) {
        assert!(lookups.contains_key(declared), "stale entry {declared:?}");
        assert!(
            !resolves(declared),
            "classified absent but resolves: {declared:?}"
        );
    }
    let both: Vec<_> = gaps.keys().filter(|k| absent.contains(*k)).collect();
    assert!(both.is_empty(), "declared twice: {both:?}");
}

/// 2026-10-07: Prints the classified table (`-- --nocapture`); the doc's lookup table is this.
#[test]
fn laguna_lookup_inventory() {
    let entries = compiled_entries(&laguna());
    let absent = expected_absent();
    let gaps = gap_index();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for ((m, k), (site, required)) in laguna_lookups() {
        let class = if entries.get(&m).is_some_and(|e| e.contains(&k)) {
            "resolves".to_string()
        } else if absent.contains(&(m.clone(), k.clone())) {
            "expected_absent".to_string()
        } else if let Some(class) = gaps.get(&(m.clone(), k.clone())) {
            format!("{class:?}")
        } else {
            "UNCLASSIFIED".to_string()
        };
        let kind = if required { "required" } else { "probe" };
        println!("{class}\t{kind}\t{m}::{k}\t{site}");
        *counts.entry(class).or_default() += 1;
    }
    println!("{counts:?}");
}

#[test]
fn only_int4_resolves() {
    let root = root();
    for quant in ["nvfp4", "bf16", "fp8", "mxfp4"] {
        assert!(
            discover(&root, &target("laguna-xs-2.1", quant)).is_err(),
            "common-only fallback for {quant}"
        );
    }
    let quants: Vec<String> = walk(&root)
        .unwrap()
        .into_iter()
        .filter(|t| t.hardware == "strix-hip" && t.model == "laguna-xs-2.1")
        .map(|t| t.quant)
        .collect();
    assert_eq!(quants, ["int4"]);
}

#[test]
fn laguna_resolution_never_includes_the_common_stub_entry_points() {
    let root = root();
    // 2026-10-07: Positive control: the scan finds the stubs where they resolve.
    let stub_file = root.join("kernels/strix-hip/common/moe_w4a16_grouped_gemm.cu");
    let found = build_shadow::entry_points(&stub_file);
    assert!(STUBS[..3].iter().all(|s| found.contains(*s)), "{found:?}");
    let qwen = discover(&root, &target("qwen3.6-27b", "nvfp4")).unwrap();
    let qwen_entries: BTreeSet<String> = compiled_entries(&qwen).into_values().flatten().collect();
    assert!(qwen_entries.contains("moe_fp8_grouped_gemm_v2"));

    let layout = laguna();
    let defined: BTreeSet<String> = compiled_entries(&layout).into_values().flatten().collect();
    for stub in STUBS {
        assert!(
            !defined.contains(stub),
            "Laguna strix-hip resolves stub {stub}"
        );
    }
    let sources = layout.sources();
    for stub_source in [
        "moe_w4a16_grouped_gemm.cu",
        "moe_fp8_grouped_gemm.cu",
        "w4a16_gemm.cu",
    ] {
        let path = root.join("kernels/strix-hip/common").join(stub_source);
        assert!(!sources.contains(&path), "compiles {}", path.display());
    }
    assert!(!sources.contains(&root.join("kernels/gb10/common/gated_delta_rule.cu")));
}

#[test]
fn no_other_target_resolves_a_laguna_strix_hip_file() {
    let root = root();
    let leaf = root.join("kernels/strix-hip/laguna-xs-2.1");
    let mut others = 0;
    for t in walk(&root).unwrap() {
        if t.hardware == "strix-hip" && t.model == "laguna-xs-2.1" {
            continue;
        }
        others += 1;
        let layout = discover(&root, &t).unwrap();
        for input in layout.inputs() {
            assert!(
                !input.starts_with(&leaf),
                "{t} resolves {}",
                input.display()
            );
        }
    }
    assert!(others > 50, "only {others} other targets");
}

/// 2026-10-07: hipcc has compiled a strix-hip source only if some strix-hip target resolves
/// it. Apart from its packed-int GEMV, the Laguna leaf compiles only what a Qwen strix-hip
/// target already compiles (at another HDIM: the attention sources build at HDIM 128 here,
/// a GPU-gate compile item).
#[test]
fn laguna_compiles_nothing_new_to_hipcc_but_its_packed_int_gemv() {
    let root = root();
    let mut proven: BTreeSet<PathBuf> = BTreeSet::new();
    for model in ["qwen3.6-27b", "qwen3.6-35b-a3b"] {
        proven.extend(discover(&root, &target(model, "nvfp4")).unwrap().sources());
    }
    let leaf = root.join("kernels/strix-hip/laguna-xs-2.1/int4");
    for (stem, entry) in laguna().modules() {
        let source = &entry.source;
        if stem == "packed_int_gemv" {
            assert_eq!(*source, leaf.join("packed_int_gemv.cu"));
            continue;
        }
        if source.starts_with(&leaf) {
            let text = std::fs::read_to_string(source).unwrap();
            assert!(
                build_shadow::entry_points(source).is_empty() && !text.contains("#include"),
                "{stem}: a Laguna strix-hip leaf shadow must define nothing"
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
