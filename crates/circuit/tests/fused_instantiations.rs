// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The owner's enforced rule (glm/megakernel/CHARTER.md, 2026-10-22): every fused,
//! generated or persistent-segment kernel is a **CUDA template instantiation of a registered
//! family** (the task-body / hook ABI), never a new hand-written fused `.cu`. This gate refuses a
//! kernel a fusion rule names (a FUSIONS.toml rule over two or more ops) unless it is an
//! instantiation or sits on the shrink-only allowlist of the hand-fused kernels that predate the
//! rule (`kernels/<class>/common/FUSED_HANDWRITTEN.toml`).
//!
//! A kernel `module::func` is an instantiation when its module's source (`<module>.cu`, every copy
//! under `kernels/<class>/`) starts with `// metrale: instantiates <family> (<template>.cuh)`,
//! `#include`s that template, holds no loop, shared memory or inline PTX of its own (the
//! arithmetic and the memory schedule are the template's), and the family is registered in
//! `KERNEL_FAMILIES.toml` with the kernel in its `kernels`.
//!
//! Owner: metrale-circuit tests.
//! Invariants:
//! - Shrink-only: an allowlisted kernel that is now an instantiation, or no longer named by a
//!   fusion rule, fails the gate until its entry is removed; the allowlist never exceeds its
//!   seeded size (`CEILING`), so growing it is a visible edit of this file.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// 2026-10-23: The classes the gate covers (each has a FUSIONS.toml).
const CLASSES: [&str; 1] = ["gb10"];
/// 2026-10-23: The allowlist's seeded size per class: the hand-fused kernels FUSIONS.toml named
/// on 2026-10-23 (megakernel branch @41654eeb1). It may only fall.
const CEILING: [(&str, usize); 1] = [("gb10", 103)];
/// 2026-10-23: The marker an instantiation's first line carries.
const MARKER: &str = "// metrale: instantiates ";
/// 2026-10-23: Constructs an instantiation must not contain: they belong to the template.
const FORBIDDEN: [&str; 6] = ["for (", "for(", "while (", "while(", "__shared__", "asm("];

/// 2026-10-23: Every kernel a fusion rule (two or more pattern ops) names, `module::func`.
fn fused_kernels(fusions: &str) -> BTreeSet<String> {
    let t: toml::Table = toml::from_str(fusions).expect("FUSIONS.toml");
    let mut out = BTreeSet::new();
    for r in t["rule"].as_array().expect("[[rule]]") {
        if r["pattern"].as_array().map_or(0, Vec::len) < 2 {
            continue;
        }
        for k in r.get("kernels").and_then(|k| k.as_array()).into_iter().flatten() {
            out.insert(format!(
                "{}::{}",
                k["module"].as_str().expect("module"),
                k["func"].as_str().expect("func")
            ));
        }
    }
    out
}

/// 2026-10-23: Family id to the kernels it lists.
fn families(text: &str) -> BTreeMap<String, BTreeSet<String>> {
    let t: toml::Table = toml::from_str(text).expect("KERNEL_FAMILIES.toml");
    t["family"]
        .as_array()
        .expect("[[family]]")
        .iter()
        .map(|f| {
            let ks = f
                .get("kernels")
                .and_then(|k| k.as_array())
                .into_iter()
                .flatten()
                .filter_map(|k| k.as_str().map(String::from))
                .collect();
            (f["id"].as_str().expect("id").to_string(), ks)
        })
        .collect()
}

fn sources_of(dir: &Path, module: &str, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("kernels dir").flatten() {
        let p = e.path();
        if p.is_dir() {
            sources_of(&p, module, out);
        } else if p.file_name().is_some_and(|n| *n == *format!("{module}.cu")) {
            out.push(p);
        }
    }
}

/// 2026-10-23: Why `kernel` is not an instantiation of a registered family; `None` when it is.
fn not_instantiation(
    class_dir: &Path,
    kernel: &str,
    fams: &BTreeMap<String, BTreeSet<String>>,
) -> Option<String> {
    let module = kernel.split("::").next().expect("module::func");
    let mut files = Vec::new();
    sources_of(class_dir, module, &mut files);
    if files.is_empty() {
        return Some(format!("no `{module}.cu` under {}", class_dir.display()));
    }
    for f in files {
        let text = std::fs::read_to_string(&f).expect("source");
        let Some(rest) = text.lines().next().and_then(|l| l.strip_prefix(MARKER)) else {
            return Some(format!("{} has no `{MARKER}...` first line", f.display()));
        };
        let Some((family, tpl)) = rest.split_once(" (").map(|(a, b)| (a, b.trim_end_matches(')')))
        else {
            return Some(format!("{}: marker is `{MARKER}<family> (<t>.cuh)`", f.display()));
        };
        if !text.contains(&format!("#include \"{tpl}\"")) {
            return Some(format!("{} does not include its template `{tpl}`", f.display()));
        }
        if let Some(bad) = FORBIDDEN.iter().find(|b| text.contains(*b)) {
            return Some(format!("{} holds `{bad}`: that belongs to the template", f.display()));
        }
        if !fams.get(family).is_some_and(|ks| ks.contains(kernel)) {
            return Some(format!("family `{family}` is not registered with `{kernel}`"));
        }
    }
    None
}

fn allowlist(class: &str) -> BTreeSet<String> {
    let rel = format!("kernels/{class}/common/FUSED_HANDWRITTEN.toml");
    let t: toml::Table = toml::from_str(&common::read(&rel)).expect("allowlist");
    t["handwritten"]
        .as_array()
        .expect("handwritten = [...]")
        .iter()
        .map(|v| v.as_str().expect("kernel").to_string())
        .collect()
}

/// 2026-10-23: The violations of `class`: fused kernels that are not instantiations.
fn violations(class: &str) -> BTreeMap<String, String> {
    let fams = families(&common::read(&format!(
        "kernels/{class}/common/KERNEL_FAMILIES.toml"
    )));
    fused_kernels(&common::read(&format!("kernels/{class}/common/FUSIONS.toml")))
        .into_iter()
        .filter_map(|k| {
            let dir = common::root().join("kernels").join(class);
            not_instantiation(&dir, &k, &fams).map(|why| (k, why))
        })
        .collect()
}

#[test]
fn every_fused_kernel_is_a_family_instantiation_or_allowlisted() {
    for class in CLASSES {
        let allowed = allowlist(class);
        let bad: Vec<String> = violations(class)
            .into_iter()
            .filter(|(k, _)| !allowed.contains(k))
            .map(|(k, why)| format!("{k}: {why}"))
            .collect();
        assert!(
            bad.is_empty(),
            "{class}: fused kernels that are not CUDA template instantiations of a registered \
             family (the owner's rule, 2026-10-22; instantiate a family template instead):\n  {}",
            bad.join("\n  ")
        );
    }
}

#[test]
fn the_allowlist_only_shrinks() {
    for class in CLASSES {
        let allowed = allowlist(class);
        let v = violations(class);
        let stale: Vec<&String> = allowed.iter().filter(|k| !v.contains_key(*k)).collect();
        assert!(
            stale.is_empty(),
            "{class}: allowlisted kernels that are no longer hand-fused fusion kernels; remove \
             them from FUSED_HANDWRITTEN.toml and lower CEILING: {stale:?}"
        );
        let ceiling = CEILING.iter().find(|(c, _)| *c == class).expect("ceiling").1;
        assert!(
            allowed.len() <= ceiling,
            "{class}: the allowlist grew to {} past its ceiling {ceiling}",
            allowed.len()
        );
    }
}

/// 2026-10-23: The check catches what it claims, on a scratch tree: a clean instantiation of a
/// registered family passes; one that loops, one with no marker, one that skips its template and
/// one its family does not list are each refused, with the reason.
#[test]
fn the_instantiation_check_refuses_handwritten_arithmetic() {
    let dir = std::env::temp_dir().join(format!("fused_inst_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    let fams = BTreeMap::from([(
        "fam".to_string(),
        BTreeSet::from(["k_ok::k_ok".to_string(), "k_loop::k_loop".to_string()]),
    )]);
    let good = |name: &str| {
        format!(
            "// metrale: instantiates fam (fam_body.cuh)\n#include \"fam_body.cuh\"\n\
             extern \"C\" __global__ void {name}(float* y) {{ fam_body<Id>(y); }}\n"
        )
    };
    let cases = [
        ("k_ok", good("k_ok"), None),
        (
            "k_loop",
            good("k_loop").replace("fam_body<Id>(y);", "for (int i = 0; i < 4; ++i) y[i] = 0;"),
            Some("belongs to the template"),
        ),
        ("k_bare", "extern \"C\" __global__ void k_bare() {}\n".into(), Some("first line")),
        (
            "k_noinc",
            good("k_noinc").replace("#include \"fam_body.cuh\"\n", ""),
            Some("does not include"),
        ),
        ("k_unreg", good("k_unreg"), Some("is not registered")),
    ];
    for (m, body, want) in cases {
        std::fs::write(dir.join(format!("{m}.cu")), body).expect("write");
        let got = not_instantiation(&dir, &format!("{m}::{m}"), &fams);
        match want {
            None => assert_eq!(got, None, "{m}"),
            Some(w) => assert!(got.as_deref().is_some_and(|g| g.contains(w)), "{m}: {got:?}"),
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
