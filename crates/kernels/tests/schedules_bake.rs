// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The SCHEDULES.toml bake (`build_schedules.rs`): parse refusals, the source-digest
//! staleness rule, the generated literal, and the constants build.rs baked for this checkout.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! An integration test because cargo does not run a build script's own unit tests. Every test but
//! the last feeds texts and bytes to the pure functions; the last reads the real tree.

#[path = "../build_schedules.rs"]
mod build_schedules;

use build_schedules::{Baked, bake, literal, parse, read_tree, source_digest};
use std::collections::BTreeMap;
use std::path::PathBuf;

const FILE: &str = "kernels/gb10/common/SCHEDULES.toml";
const TC: &str = "kernels/gb10/common/w4a16_gemv_tc.cu";
const TC_H: &str = "kernels/gb10/common/w4a16_gemv_tc.cuh";
const W8: &str = "kernels/gb10/common/w8a8_gemm.cu";

fn sources() -> BTreeMap<String, Vec<u8>> {
    [
        (TC, b"__global__ void tc8() {}".to_vec()),
        (TC_H, b"#define TILE 8".to_vec()),
        (W8, b"__global__ void w8() {}".to_vec()),
    ]
    .into_iter()
    .map(|(p, b)| (p.to_string(), b))
    .collect()
}

fn digest_of(files: &BTreeMap<String, Vec<u8>>, paths: &[&str]) -> String {
    source_digest(paths.iter().map(|p| (*p, files[*p].as_slice())))
}

fn entry(rows: &str, kernel: &str, family: &str, numerics: &str, enabled: &str) -> String {
    format!(
        "[[schedule]]\nop = \"linear\"\nweight = \"nvfp4/g16\"\nactivation = \"bf16\"\n\
         k = 5120\nn = 17408\nrows = {rows}\nkernel = \"{kernel}\"\nfamily = \"{family}\"\n\
         default = \"w4a16_gemv::w4a16_gemv_sw\"\nnumerics = \"{numerics}\"\n\
         enabled = \"{enabled}\"\nmedian_us = 12.3\ndefault_us = 0\nfloor_us = 9.8\n\
         measured = \"dgx1 2026-10-11T03:10:00Z\"\n"
    )
}

/// 2026-10-10: Two families over `files`' digests at sweep time, one schedule each.
fn text(files: &BTreeMap<String, Vec<u8>>) -> String {
    format!(
        "schema = 1\nhardware = \"gb10\"\ngenerated_by = \"met envelope schedules\"\n\
         [sources.w4a16_tc]\nfiles = [\"{TC}\", \"{TC_H}\"]\nsha256 = \"{}\"\n\
         [sources.w8a8]\nfiles = [\"{W8}\"]\nsha256 = \"{}\"\n{}{}",
        digest_of(files, &[TC, TC_H]),
        digest_of(files, &[W8]),
        entry(
            "[1, 4]",
            "w4a16_gemv_tc::tc4",
            "w4a16_tc",
            "bit_identical",
            "default"
        ),
        entry("[5, 8]", "w8a8_gemm::w8", "w8a8", "differs", "opt_in"),
    )
}

fn baked(text: &str, now: &BTreeMap<String, Vec<u8>>) -> Baked {
    bake(&parse(FILE, "gb10", text).expect("parses"), now)
}

fn kernels(b: &Baked) -> Vec<&str> {
    b.schedules.iter().map(|e| e.kernel.as_str()).collect()
}

fn refusal(text: &str) -> String {
    let e = parse(FILE, "gb10", text).expect_err("refused");
    assert!(e.contains(FILE), "the error names the file: {e}");
    e
}

#[test]
fn unchanged_sources_keep_every_schedule() {
    let files = sources();
    let b = baked(&text(&files), &files);
    assert_eq!(kernels(&b), ["w4a16_gemv_tc::tc4", "w8a8_gemm::w8"]);
    assert!(b.stale.is_empty());
    assert_eq!((b.schedules[0].rows_lo, b.schedules[0].rows_hi), (1, 4));
}

#[test]
fn a_changed_file_drops_only_its_family_and_names_it() {
    let swept = sources();
    let mut now = swept.clone();
    now.get_mut(TC_H).unwrap().push(b'\n');
    let b = baked(&text(&swept), &now);
    assert_eq!(kernels(&b), ["w8a8_gemm::w8"]);
    assert_eq!(b.stale, ["w4a16_tc"]);
}

#[test]
fn a_deleted_file_makes_its_family_stale() {
    let swept = sources();
    let mut now = swept.clone();
    now.remove(W8);
    let b = baked(&text(&swept), &now);
    assert_eq!(kernels(&b), ["w4a16_gemv_tc::tc4"]);
    assert_eq!(b.stale, ["w8a8"]);
}

/// 2026-10-10: The digest is SHA-256 over path bytes then content bytes, in listed order, with no
/// separator: ("a", "bc") hashes as "abc" (FIPS 180-2 vector). So a rename or a reorder with the
/// same bytes is a change.
#[test]
fn the_digest_covers_path_then_bytes_in_listed_order() {
    assert_eq!(
        source_digest([("a", b"bc".as_slice())]),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let files = sources();
    assert_ne!(
        digest_of(&files, &[TC, TC_H]),
        digest_of(&files, &[TC_H, TC])
    );
    let renamed = source_digest([("kernels/gb10/common/other.cu", files[TC].as_slice())]);
    assert_ne!(renamed, digest_of(&files, &[TC]));
}

#[test]
fn an_uppercase_recorded_digest_still_matches() {
    let files = sources();
    let t = text(&files).replace(
        &digest_of(&files, &[W8]),
        &digest_of(&files, &[W8]).to_uppercase(),
    );
    assert!(baked(&t, &files).stale.is_empty());
}

#[test]
fn schema_2_is_refused() {
    let t = text(&sources()).replace("schema = 1", "schema = 2");
    assert!(refusal(&t).contains("schema 2"));
}

#[test]
fn an_unknown_key_is_refused_at_every_level() {
    let base = text(&sources());
    for (from, to) in [
        ("schema = 1\n", "schema = 1\nwinner = 3\n"),
        ("[sources.w8a8]\n", "[sources.w8a8]\nflags = []\n"),
        (
            "enabled = \"opt_in\"\n",
            "enabled = \"opt_in\"\nsplit_k = 2\n",
        ),
    ] {
        let e = refusal(&base.replacen(from, to, 1));
        assert!(e.contains("has no key"), "{e}");
    }
}

#[test]
fn a_missing_key_or_a_bad_value_is_refused() {
    let base = text(&sources());
    for (from, to) in [
        ("generated_by = \"met envelope schedules\"\n", ""),
        ("kernel = \"w8a8_gemm::w8\"\n", ""),
        ("numerics = \"differs\"", "numerics = \"faster\""),
        ("enabled = \"opt_in\"", "enabled = \"always\""),
        ("rows = [5, 8]", "rows = [8, 5]"),
        ("rows = [5, 8]", "rows = [5]"),
        ("hardware = \"gb10\"", "hardware = \"hopper\""),
    ] {
        assert!(base.contains(from), "{from}");
        refusal(&base.replacen(from, to, 1));
    }
}

/// 2026-10-10: The numerics policy: a bits-changing winner is never enabled by default.
#[test]
fn a_default_enabled_entry_that_changes_bits_is_refused() {
    for numerics in ["differs", "new"] {
        let t = text(&sources()).replace(
            "numerics = \"bit_identical\"",
            &format!("numerics = \"{numerics}\""),
        );
        assert!(refusal(&t).contains("opt_in"));
    }
}

#[test]
fn overlapping_rows_of_one_shape_are_refused_and_adjacent_rows_are_not() {
    let files = sources();
    let overlap = text(&files).replace("rows = [5, 8]", "rows = [4, 8]");
    assert!(refusal(&overlap).contains("overlap"));
    assert_eq!(baked(&text(&files), &files).schedules.len(), 2);
}

#[test]
fn a_family_without_sources_or_a_path_leaving_the_repo_is_refused() {
    let base = text(&sources());
    let orphan = base.replace("family = \"w8a8\"", "family = \"w8a8_split\"");
    assert!(refusal(&orphan).contains("w8a8_split"));
    for path in ["../secret.cu", "/etc/passwd"] {
        let t = base.replacen(W8, path, 1);
        assert!(refusal(&t).contains("repo-relative"));
    }
}

/// 2026-10-10: The literal [`literal`] writes, minus comments and whitespace, is the token text
/// of items that compile against the runtime types.
macro_rules! compiled {
    ($($t:tt)*) => {
        #[allow(dead_code)]
        mod compiled_sample {
            use metrale_kernels::{Enabled, Numerics, Schedule};
            $($t)*
        }
        const COMPILED_TEXT: &str = stringify!($($t)*);
    };
}

compiled! {
    pub const TARGET_SCHEDULES: &[Schedule] = &[
        Schedule { op: "linear", weight: "nvfp4/g16", activation: "bf16", k: 5120, n: 17408,
            rows_lo: 1, rows_hi: 4, kernel: "w4a16_gemv_tc::tc4", family: "w4a16_tc",
            default: Some("w4a16_gemv::w4a16_gemv_sw"), numerics: Numerics::BitIdentical,
            enabled: Enabled::Default },
        Schedule { op: "linear", weight: "nvfp4/g16", activation: "bf16", k: 5120, n: 17408,
            rows_lo: 5, rows_hi: 8, kernel: "w8a8_gemm::w8", family: "w8a8", default: None,
            numerics: Numerics::New, enabled: Enabled::OptIn },
    ];
    pub const TARGET_SCHEDULES_STALE: &[&str] = &["w4a16_old", "w8a8_old"];
}

fn tokens(s: &str) -> String {
    s.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .flat_map(str::chars)
        .filter(|c| !c.is_whitespace())
        .collect()
}

#[test]
fn the_literal_is_the_text_of_compiling_items() {
    let files = sources();
    let t = text(&files)
        .replace(
            "numerics = \"differs\"\nenabled = \"opt_in\"\n",
            "numerics = \"new\"\nenabled = \"opt_in\"\n",
        )
        .replacen(
            "default = \"w4a16_gemv::w4a16_gemv_sw\"",
            "default = \"\"",
            2,
        )
        .replacen(
            "default = \"\"",
            "default = \"w4a16_gemv::w4a16_gemv_sw\"",
            1,
        );
    let mut b = baked(&t, &files);
    b.stale = vec!["w4a16_old".into(), "w8a8_old".into()];
    assert_eq!(tokens(&literal(&b)), tokens(COMPILED_TEXT));
}

/// 2026-10-10: What build.rs baked for this checkout is what the bake of the real tree gives,
/// so the generated constants follow the file (and are empty while it is absent).
#[test]
fn the_baked_constants_are_the_bake_of_this_tree() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf();
    let hw = metrale_kernels::TARGET_DEFAULTS.hw;
    let (b, _) = read_tree(&root, hw).expect("this tree's SCHEDULES.toml bakes");
    let baked: Vec<&str> = metrale_kernels::TARGET_SCHEDULES
        .iter()
        .map(|s| s.kernel)
        .collect();
    assert_eq!(baked, kernels(&b));
    assert_eq!(metrale_kernels::TARGET_SCHEDULES_STALE, b.stale.as_slice());
}
