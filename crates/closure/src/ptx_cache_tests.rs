// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests of the PTX cache's pure half: each key input moves the
//! key, a sidecar that does not vouch for the bytes is a miss, and eviction is
//! least-recently-used, bounded by the cap, and never touches this build's
//! entries.
//!
//! Owner: metrale-closure.
//! Invariants: none beyond the types.

use super::*;
use std::time::Duration;

struct Fixture {
    argv: Vec<String>,
    env: Vec<(String, String)>,
    files: Vec<FileFact>,
    unresolved: Vec<String>,
}

fn fixture() -> Fixture {
    Fixture {
        argv: [
            "--ptx",
            "-arch=sm_121f",
            "-O3",
            "--fmad=false",
            "-I/out/stage",
            "/out/stage/dflash_fp8_gemm.cu",
            "-o",
            "/out/t3__dflash_fp8_gemm.ptx",
        ]
        .map(String::from)
        .to_vec(),
        env: vec![("NVCC_APPEND_FLAGS".into(), "-lineinfo".into())],
        files: vec![
            FileFact::of(Path::new("/out/stage/dflash_fp8_gemm.cu"), b"#include x\n"),
            FileFact::of(Path::new("/out/stage/w4a16_gemm.cu"), b"#define TILE 64\n"),
        ],
        unresolved: Vec::new(),
    }
}

fn key(f: &Fixture) -> String {
    cache_key(&KeyInputs {
        compiler: "/cuda/bin/nvcc\x1fnvcc version 13.0",
        argv: &f.argv,
        nvcc_env: &f.env,
        files: &f.files,
        unresolved: &f.unresolved,
    })
}

/// 2026-10-17: Any argv change is a different key: a flag, its order, the
/// arch, and the source path (a device `assert` embeds it in the PTX).
#[test]
fn an_argv_change_alters_the_key() {
    let base = key(&fixture());
    let mut f = fixture();
    f.argv.insert(3, "-lineinfo".into());
    assert_ne!(base, key(&f), "an added flag");
    let mut f = fixture();
    f.argv.swap(1, 2);
    assert_ne!(base, key(&f), "flag order is meaningful to nvcc");
    let mut f = fixture();
    f.argv[1] = "-arch=sm_90a".into();
    assert_ne!(base, key(&f), "the arch");
    let mut f = fixture();
    f.argv[5] = "/elsewhere/stage/dflash_fp8_gemm.cu".into();
    assert_ne!(base, key(&f), "the source path");
}

/// 2026-10-17: Editing a header the source reaches through a search directory
/// moves the key even though no argv token and no path changes. This is the
/// negative control's shape: the qwen `w4a16_gemm.cu` edit must miss for the
/// glm `dflash_fp8_gemm.cu`.
#[test]
fn an_include_closure_content_change_alters_the_key() {
    let base = key(&fixture());
    let mut f = fixture();
    f.files[1] = FileFact::of(Path::new("/out/stage/w4a16_gemm.cu"), b"#define TILE 128\n");
    assert_ne!(base, key(&f));

    let mut f = fixture();
    f.files
        .push(FileFact::of(Path::new("/out/stage/new.cuh"), b""));
    assert_ne!(base, key(&f), "a file entering the closure");

    let mut f = fixture();
    f.unresolved.push("dflash_fp8_gemm.cu -> gone.cuh".into());
    assert_ne!(base, key(&f), "an include becoming unresolvable");
}

/// 2026-10-17: An `NVCC_*` variable is read by nvcc behind the argv's back, so
/// setting, changing or removing one moves the key.
#[test]
fn an_nvcc_env_change_alters_the_key() {
    let base = key(&fixture());
    let mut f = fixture();
    f.env[0].1 = "-G".into();
    assert_ne!(base, key(&f), "a changed value");
    let mut f = fixture();
    f.env.push(("NVCC_CCBIN".into(), "/usr/bin/g++-13".into()));
    assert_ne!(base, key(&f), "an added variable");
    let mut f = fixture();
    f.env.clear();
    assert_ne!(base, key(&f), "a removed variable");
}

/// 2026-10-17: The sorted inputs are order-free, and the framing keeps a value
/// from sliding between fields: the same token as an env value and as an argv
/// token gives different keys.
#[test]
fn the_key_is_order_free_where_order_is_not_meaningful_and_framed() {
    let base = key(&fixture());
    let mut f = fixture();
    f.files.reverse();
    f.env.push(("NVCC_CCBIN".into(), "c".into()));
    f.env.reverse();
    let mut g = fixture();
    g.env.push(("NVCC_CCBIN".into(), "c".into()));
    assert_eq!(key(&f), key(&g), "file and env order must not matter");
    assert_eq!(base, key(&fixture()), "repeatable");

    let mut f = fixture();
    f.env.clear();
    f.argv.push("NVCC_APPEND_FLAGS=-lineinfo".into());
    assert_ne!(base, key(&f));
}

/// 2026-10-17: A sidecar is trusted only when it names the bytes' digest:
/// changed bytes, a truncated entry, a stale or empty sidecar all miss.
#[test]
fn a_sidecar_mismatch_is_a_miss() {
    let ptx = b".version 9.0\n.target sm_121f\n";
    let good = sidecar_text(ptx);
    assert!(verify(ptx, &good));
    assert!(
        !verify(b".version 9.0\n.target sm_90a\n", &good),
        "changed bytes"
    );
    assert!(!verify(&ptx[..10], &good), "truncated entry");
    assert!(!verify(ptx, &sidecar_text(b"other")), "stale sidecar");
    assert!(!verify(ptx, ""), "empty sidecar from a torn write");
    assert!(
        !verify(ptx, good.trim_end()),
        "a sidecar missing its newline"
    );
}

fn entry(name: &str, bytes: u64, age_s: u64, now: SystemTime) -> CacheEntry {
    CacheEntry {
        path: PathBuf::from(name),
        bytes,
        last_used: now - Duration::from_secs(age_s),
    }
}

/// 2026-10-17: Oldest first, and only until the total fits the cap.
#[test]
fn eviction_is_least_recently_used_and_stops_at_the_cap() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let entries = [
        entry("new", 40, 10, now),
        entry("oldest", 40, 300, now),
        entry("mid", 40, 200, now),
        entry("old", 40, 250, now),
    ];
    assert_eq!(
        evict(&entries, 80, now),
        [PathBuf::from("oldest"), PathBuf::from("old")]
    );
    assert_eq!(
        evict(&entries, 160, now),
        Vec::<PathBuf>::new(),
        "at the cap"
    );
    assert_eq!(evict(&entries, 159, now), [PathBuf::from("oldest")]);
}

/// 2026-10-17: An entry this build used (at or after its start) is never
/// evicted, even when the cache stays over the cap as a result.
#[test]
fn eviction_spares_entries_this_build_used() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let start = now - Duration::from_secs(60);
    let entries = [
        entry("this-build", 100, 0, now),
        entry("at-start", 100, 60, now),
        entry("before", 100, 61, now),
    ];
    assert_eq!(evict(&entries, 0, start), [PathBuf::from("before")]);
}

/// 2026-10-17: Equal timestamps (coarse mtime) are broken by path, so two
/// runs over one directory choose the same victims.
#[test]
fn eviction_ties_break_by_path() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let entries = [
        entry("b", 10, 5, now),
        entry("a", 10, 5, now),
        entry("c", 10, 5, now),
    ];
    assert_eq!(
        evict(&entries, 15, now),
        [PathBuf::from("a"), PathBuf::from("b")]
    );
}

/// 2026-10-17: nvcc's quoted-include search order: `-I` forms in order, then
/// system dirs, with `NVCC_PREPEND_FLAGS` before and `NVCC_APPEND_FLAGS` after
/// the argv; options that merely start like one are not search dirs.
#[test]
fn search_dirs_follow_nvcc_flag_forms_and_order() {
    let argv = [
        "-isystem",
        "/sys1",
        "-I/a",
        "-I",
        "/b",
        "--include-path=/c",
        "--include-path",
        "/d",
        "--include-pathx",
        "-o",
        "/out.ptx",
    ]
    .map(String::from);
    let env = [
        ("NVCC_PREPEND_FLAGS".to_string(), "-I/pre".to_string()),
        (
            "NVCC_APPEND_FLAGS".to_string(),
            "--system-include=/sys2 -I/post".to_string(),
        ),
    ];
    let got = search_dirs(&argv, &env);
    let want: Vec<PathBuf> = ["/pre", "/a", "/b", "/c", "/d", "/post", "/sys1", "/sys2"]
        .map(PathBuf::from)
        .to_vec();
    assert_eq!(got, want);
}

/// 2026-10-17: The switch and the cap admit only explicit values: no typo
/// picks a mode and no missing cap falls back to a hidden default.
#[test]
fn the_switch_and_cap_reject_anything_implicit() {
    assert_eq!(parse_switch(None), Ok(false));
    assert_eq!(parse_switch(Some("0")), Ok(false));
    assert_eq!(parse_switch(Some("1")), Ok(true));
    for bad in ["", "true", "yes", " 1"] {
        assert!(parse_switch(Some(bad)).is_err(), "{bad:?}");
    }
    assert_eq!(parse_cap(Some("1048576")), Ok(1_048_576));
    for bad in [None, Some(""), Some("0"), Some("-1"), Some("2GiB")] {
        assert!(parse_cap(bad).is_err(), "{bad:?}");
    }
}

/// 2026-10-17: Entries fan out by the key's first two characters under the
/// schema directory, and the sidecar sits beside the entry.
#[test]
fn entry_layout_is_versioned_and_fanned_out() {
    let rel = entry_rel("ab12cd");
    assert_eq!(rel, Path::new("v1/ab/ab12cd.ptx"));
    assert_eq!(sidecar_of(&rel), Path::new("v1/ab/ab12cd.ptx.sha256"));
}
