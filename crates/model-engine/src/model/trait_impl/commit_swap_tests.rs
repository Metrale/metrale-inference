// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Where the commit swap must be honoured, checked in the source: a graph that bakes
//! SSM addresses is either keyed by the slot's role or preceded by the return home.
//!
//! Owner: model-engine speculative decoding.
//! Invariants: none beyond the checks.

fn src(rel: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
        .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

fn rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// 2026-10-21: Every slot-graph key outside its own tests comes from `slot_graph_key`, which
/// reads the slot's role: a key built from the slot alone would replay a graph that baked the
/// other buffers.
#[test]
fn every_slot_graph_key_carries_the_swap_role() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rs_files(&root, &mut files);
    let mut offenders = Vec::new();
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        if name == "commit_swap.rs" || name.ends_with("_tests.rs") || name == "slot_graph_key.rs" {
            continue;
        }
        if std::fs::read_to_string(&f)
            .unwrap()
            .contains("SlotGraphKey::new(")
        {
            offenders.push(f.display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "SlotGraphKey::new outside slot_graph_key: {offenders:?}"
    );
    for f in ["verify_b.rs", "verify_c.rs", "verify_c2.rs", "decode_a.rs"] {
        assert!(
            src(&format!("src/model/trait_impl/{f}")).contains(".slot_graph_key(seq.slot_idx"),
            "{f} keys its graph without the swap role"
        );
    }
}

/// 2026-10-21: Only the role-keyed single-sequence verifies arm a swap, and the two
/// multi-sequence graph paths (slot-vector keys) return a swapped slot home first.
#[test]
fn only_role_keyed_verifies_arm_and_batched_steps_return_home() {
    for f in ["verify_b.rs", "verify_c.rs", "verify_c2.rs"] {
        assert!(
            src(&format!("src/model/trait_impl/{f}"))
                .contains("self.arm_commit_swap(seq.slot_idx)"),
            "{f} does not arm"
        );
    }
    let d = src("src/model/trait_impl/decode_a2.rs");
    let main = &d[d.find("fn decode_batch_compute_main_with").unwrap()..];
    let key = main.find("batch_decode_graph_key(").unwrap();
    let home = main
        .find("self.unswap_for_batch(s)?")
        .expect("batched decode returns home");
    assert!(home < key, "the return home must precede the graph key");
    let v = src("src/model/trait_impl/verify_e.rs");
    let body = &v[v.find("fn decode_verify_batched_dispatch").unwrap()..];
    let home = body
        .find("self.unswap_for_batch(s)")
        .expect("batched verify returns home");
    let wy = body.find("upload_verify_wy_tables(").unwrap();
    assert!(
        home < wy,
        "the return home must precede the WY tables and the graph key"
    );
    let mut files = Vec::new();
    rs_files(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let arms = files
        .iter()
        .filter(|f| {
            let n = f.file_name().unwrap().to_string_lossy().into_owned();
            !n.ends_with("_tests.rs")
                && n != "commit_swap.rs"
                && std::fs::read_to_string(f)
                    .unwrap()
                    .contains("arm_commit_swap(")
        })
        .count();
    assert_eq!(arms, 3, "an arm outside the three role-keyed verifies");
}
