// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Every kernel family of the manifest states its math, and every entry of the
//! kernel-math table (all families, every format spelling the circuits use) renders in KaTeX
//! strict mode.
//!
//! Owner: metrale-circuit tests.
//! Invariants: the KaTeX leg needs node and KaTeX 0.18.4, which this repository does not vendor
//! (the blueprint store does), so it is `#[ignore]` and run on purpose:
//! `METRALE_KATEX=<path>/katex.mjs cargo test -p metrale-circuit --test kernel_math -- --ignored`.
//! The structural checks (every family has non-empty, balanced, public math) run always, in the
//! loader.

mod common;

use std::collections::BTreeSet;

use metrale_circuit::Format;
use metrale_circuit::explain::kernel_math;
use metrale_circuit::venn::parse_families;

const MANIFEST: &str = "kernels/gb10/common/KERNEL_FAMILIES.toml";

/// 2026-10-22: Every format spelling the circuits (blocks, precision tables, checkpoint plans)
/// and the manifest state.
fn circuit_formats() -> BTreeSet<Format> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "toml") {
                out.push(p);
            }
        }
    }
    let mut files = vec![common::root().join(MANIFEST)];
    walk(&common::root().join("kernels/circuits"), &mut files);
    let mut out = BTreeSet::new();
    for p in files {
        let text = std::fs::read_to_string(&p).unwrap();
        for tok in text.split(['"', '\'', ':', ' ', ',']) {
            if let Ok(f) = Format::parse(tok) {
                out.insert(f);
            }
        }
    }
    out
}

fn table() -> metrale_circuit::explain::KernelMath {
    let m = parse_families(&common::read(MANIFEST)).unwrap();
    let ids: Vec<String> = m.families.iter().map(|f| f.id.clone()).collect();
    kernel_math(&m, ids.iter().map(String::as_str), circuit_formats()).unwrap()
}

#[test]
fn every_family_of_the_manifest_has_its_math() {
    let text = common::read(MANIFEST);
    let declared = text.lines().filter(|l| *l == "[[family]]").count();
    let t = table();
    assert_eq!(t.families.len(), declared, "one entry per [[family]]");
    assert!(declared >= 66, "{declared}");
    for (id, f) in &t.families {
        assert!(
            !f.math.is_empty() && !f.about.is_empty() && !f.numerics.is_empty(),
            "{id}"
        );
    }
    let formats: Vec<&str> = t.formats.keys().map(String::as_str).collect();
    for want in ["bf16", "f32", "i32", "fp8/token", "nvfp4/g16"] {
        assert!(formats.contains(&want), "{want} missing from {formats:?}");
    }
}

#[test]
#[ignore = "needs node and KaTeX 0.18.4: METRALE_KATEX=<katex.mjs> cargo test -p metrale-circuit --test kernel_math -- --ignored"]
fn every_entry_renders_in_katex_strict_mode() {
    // 2026-10-23: Read when the test compiles (cargo rebuilds when it changes), so this pure
    // crate's tests read no process environment (`env-ban`); the command above sets it for both.
    let Some(katex) = option_env!("METRALE_KATEX") else {
        panic!("METRALE_KATEX=<path to katex.mjs>")
    };
    let json = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("kernel_math.json");
    std::fs::write(&json, serde_json::to_string(&table()).unwrap()).unwrap();
    let out = std::process::Command::new("node")
        .arg(common::root().join("scripts/check_kernel_math.mjs"))
        .arg(katex)
        .arg(&json)
        .output()
        .expect("node on PATH");
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    println!("{report}");
    assert!(out.status.success(), "{report}");
    assert!(report.contains("OK: KaTeX 0.18.4 strict"), "{report}");
}
