// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `recipe_width` over the committed tree.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-15: Every registered gate resolves a width on GB10 without error, so a bound
/// recipe id missing from the tree, or one that no longer parses, fails here and not at
/// the start of a campaign. No committed GB10 default binds a multi-node recipe yet: the
/// GLM gates bind none, so they read one box until the 3-node recipe is bound to them
/// (then `span::hold` asks for three boxes and this pin moves with it).
#[test]
fn every_registered_gate_resolves_its_width_from_the_committed_tree() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the workspace root");
    let widths: Vec<(&str, usize)> = metrale_bench::registry::all()
        .iter()
        .map(|d| {
            let w = recipe_width(root, "gb10", d.id).unwrap_or_else(|e| panic!("{}: {e:#}", d.id));
            (d.id, w)
        })
        .collect();
    assert!(widths.iter().any(|(id, _)| *id == "ttft-cold-glm-512k"));
    assert!(widths.iter().all(|(_, w)| *w == 1), "{widths:?}");
}

/// 2026-10-15: A tree whose default entry binds a recipe with `min_nodes: 3` reads three
/// boxes, and the same tree with that recipe at one node reads one: the width is the
/// bound recipe's `min_nodes`, not a constant. A gate with no entry on the class is one.
#[test]
fn the_width_is_the_bound_recipes_min_nodes() {
    let tree = |min_nodes: usize| {
        let root = tempfile::tempdir().expect("tempdir");
        let m = root.path().join("kernels/gb10/modelA");
        std::fs::create_dir_all(&m).unwrap();
        std::fs::create_dir_all(root.path().join("recipes/fam")).unwrap();
        std::fs::write(
            root.path().join("kernels/gb10/HARDWARE.toml"),
            "[hardware]\nvendor = \"nvidia\"\n",
        )
        .unwrap();
        std::fs::write(m.join("MODEL.toml"), "[behavior]\n").unwrap();
        std::fs::write(
            m.join("BENCH.toml"),
            "[[benchmarks]]\nquant = \"nvfp4\"\ncheckpoint = \"org/A\"\n\
             gate = \"ttft-cold-gate\"\ndefault = true\nstatus = \"measured\"\n\
             recipe = \"fam/wide\"\n[benchmarks.metrics.median_ms]\nmax = 500.0\n",
        )
        .unwrap();
        std::fs::write(
            root.path().join("recipes/fam/wide.yaml"),
            format!(
                "recipe_version: \"2\"\nmodel: org/A\ncontainer: c\nruntime: metrale\n\
                 min_nodes: {min_nodes}\nmax_nodes: {min_nodes}\ndefaults:\n  port: 8888\n"
            ),
        )
        .unwrap();
        root
    };
    let wide = tree(3);
    assert_eq!(
        recipe_width(wide.path(), "gb10", "ttft-cold-gate").unwrap(),
        3
    );
    assert_eq!(
        recipe_width(wide.path(), "gb10", "decode-floor").unwrap(),
        1
    );
    let one = tree(1);
    assert_eq!(
        recipe_width(one.path(), "gb10", "ttft-cold-gate").unwrap(),
        1
    );
}
