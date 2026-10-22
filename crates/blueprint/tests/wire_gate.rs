// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `wire-types` gate, two rules:
//! - SSOT: this crate defines no type that `metrale-circuit` or `metrale-kernel-target`
//!   defines. The circuit instance, its plans, kernel ids, compute units and the compiled
//!   kernel target travel as those crates' own types (serde derives); a mirror type here, and
//!   the conversion it needs, is refused by name.
//! - The body types this crate does own hold no `HashMap` (iteration order), no `usize`
//!   (platform width) and no `f32`/`f64` (formatting), so a value re-encodes to the same bytes.
//!
//! Both scan code, not comments; negative self-tests prove each fires.

use std::path::{Path, PathBuf};

const OWNED: &[(&str, &str)] = &[
    ("types/mod.rs", include_str!("../src/types/mod.rs")),
    (
        "types/checkpoint.rs",
        include_str!("../src/types/checkpoint.rs"),
    ),
    ("types/kernel.rs", include_str!("../src/types/kernel.rs")),
    ("types/plan.rs", include_str!("../src/types/plan.rs")),
    (
        "types/settings.rs",
        include_str!("../src/types/settings.rs"),
    ),
    ("frame/locator.rs", include_str!("../src/frame/locator.rs")),
];

const BANNED: &[&str] = &["HashMap", "HashSet", "usize", "isize", "f32", "f64"];

/// 2026-10-22: Names both crates define for different things, each with why it is not a copy.
/// The list may only shrink.
const SHARED_NAMES: &[(&str, &str)] = &[(
    "Header",
    "the `.kbp` frame header here; a rendering's header lines (`Vec<(String, String)>`) there",
)];

fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

fn offenders(text: &str) -> Vec<(usize, &'static str)> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        for word in code(line).split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            if let Some(b) = BANNED.iter().find(|b| **b == word) {
                out.push((i + 1, *b));
            }
        }
    }
    out
}

/// 2026-10-22: The names `text` declares as a `struct`, `enum`, `union` or `trait` at any
/// visibility, or as a `pub` `type` alias (an associated `type Err` is not a type of the crate).
/// A declaration starts its line (after indentation, inside a macro too).
fn declared(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let words: Vec<&str> = code(line)
            .split(|c: char| c.is_whitespace() || c == '<' || c == '(' || c == '{' || c == ';')
            .filter(|w| !w.is_empty())
            .collect();
        let public = words.first().is_some_and(|w| w.starts_with("pub"));
        let at = if public {
            words
                .iter()
                .position(|w| !w.starts_with("pub") && !w.ends_with(')'))
        } else {
            Some(0)
        };
        let Some(at) = at else { continue };
        let kind = words.get(at).copied().unwrap_or_default();
        let wanted =
            matches!(kind, "struct" | "enum" | "union" | "trait") || (kind == "type" && public);
        if let Some(name) = words
            .get(at + 1)
            .filter(|n| wanted && n.starts_with(char::is_uppercase))
        {
            out.push(name.to_string());
        }
    }
    out
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn names_in(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    rust_files(dir, &mut files);
    files.sort();
    files
        .into_iter()
        .flat_map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            declared(&text).into_iter().map(move |n| (n, p.clone()))
        })
        .collect()
}

/// 2026-10-22: Every name this crate declares that a carried crate also declares, less the
/// [`SHARED_NAMES`].
fn duplicates(blueprint: &[(String, PathBuf)], circuit: &[(String, PathBuf)]) -> Vec<String> {
    let mut out: Vec<String> = blueprint
        .iter()
        .filter(|(n, _)| circuit.iter().any(|(c, _)| c == n))
        .filter(|(n, _)| !SHARED_NAMES.iter().any(|(s, _)| s == n))
        .map(|(n, p)| format!("{n} ({})", p.display()))
        .collect();
    out.dedup();
    out
}

#[test]
fn owned_body_types_hold_no_banned_type() {
    for (file, text) in OWNED {
        let found = offenders(text);
        let allowed = usize::from(*file == "types/settings.rs");
        assert_eq!(found.len(), allowed, "{file}: {found:?}");
    }
}

#[test]
fn the_crate_defines_no_type_the_carried_crates_define() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ours = names_in(&root.join("src"));
    let mut theirs = names_in(&root.join("../circuit/src"));
    theirs.extend(names_in(&root.join("../kernel-target/src")));
    for (found, name) in [
        (&ours, "Blueprint"),
        (&theirs, "Circuit"),
        (&theirs, "KernelTarget"),
    ] {
        assert!(
            found.iter().any(|(n, _)| n == name),
            "the scan did not find `{name}`"
        );
    }
    assert_eq!(duplicates(&ours, &theirs), Vec::<String>::new());
    for (name, why) in SHARED_NAMES {
        assert!(
            ours.iter().any(|(n, _)| n == name) && theirs.iter().any(|(n, _)| n == name),
            "SHARED_NAMES lists `{name}` ({why}) but it is no longer shared: remove it"
        );
    }
}

#[test]
fn the_gate_fires_on_a_smuggled_hashmap_and_float() {
    let smuggled = "pub struct Evil {\n    pub m: std::collections::HashMap<String, u64>,\n    pub x: f64, // fine here\n}\n";
    assert_eq!(offenders(smuggled), vec![(2, "HashMap"), (3, "f64")]);
    assert!(offenders("// a comment naming usize and f32\n").is_empty());
}

/// 2026-10-22: The Phase-1 mirrors (a wire `LayerKind`, `Mode`, `KernelId`, `KernelTarget`) are
/// what the SSOT rule refuses; a name only mentioned in a comment, or a shared name with its
/// reason, is not.
#[test]
fn the_gate_fires_on_a_mirrored_circuit_type() {
    let circuit = vec![
        ("LayerKind".to_string(), PathBuf::from("ir.rs")),
        ("Mode".to_string(), PathBuf::from("rules.rs")),
        ("KernelId".to_string(), PathBuf::from("rules.rs")),
        ("Header".to_string(), PathBuf::from("render.rs")),
        ("KernelTarget".to_string(), PathBuf::from("lib.rs")),
    ];
    let text = "/// mirror of `metrale_circuit::LayerKind`\n#[derive(Clone)]\npub enum LayerKind {\n}\npub(crate) struct KernelId<'a>(&'a str);\npub struct Header;\n// pub enum Mode {}\nimpl X { type Mode = u8; }\n    type Mode = u8;\npub struct KernelTarget {\n}\npub use metrale_kernel_target::KernelTarget;\n";
    let ours: Vec<(String, PathBuf)> = declared(text)
        .into_iter()
        .map(|n| (n, PathBuf::from("types/circuit.rs")))
        .collect();
    assert_eq!(
        duplicates(&ours, &circuit),
        vec![
            "LayerKind (types/circuit.rs)".to_string(),
            "KernelId (types/circuit.rs)".to_string(),
            "KernelTarget (types/circuit.rs)".to_string()
        ]
    );
}
