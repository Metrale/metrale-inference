// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `wire-types` gate: the body's wire types hold no `HashMap` (iteration order),
//! no `usize` (platform width) and no `f32`/`f64` (formatting), so a value re-encodes to the
//! same bytes on every host. The gate scans the code (not comments) of every file under
//! `src/types/`, and a negative self-test proves it fires.

const FILES: &[(&str, &str)] = &[
    ("types/mod.rs", include_str!("../src/types/mod.rs")),
    (
        "types/checkpoint.rs",
        include_str!("../src/types/checkpoint.rs"),
    ),
    ("types/circuit.rs", include_str!("../src/types/circuit.rs")),
    ("types/kernel.rs", include_str!("../src/types/kernel.rs")),
    ("types/plan.rs", include_str!("../src/types/plan.rs")),
    (
        "types/settings.rs",
        include_str!("../src/types/settings.rs"),
    ),
    ("frame/locator.rs", include_str!("../src/frame/locator.rs")),
];

const BANNED: &[&str] = &["HashMap", "HashSet", "usize", "isize", "f32", "f64"];

fn offenders(text: &str) -> Vec<(usize, &'static str)> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for word in code.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            if let Some(b) = BANNED.iter().find(|b| **b == word) {
                out.push((i + 1, *b));
            }
        }
    }
    out
}

/// 2026-10-22: Code that is not a wire field but must use `usize` (rendering a decimal) is
/// listed here by file and count; the list may only shrink.
const ALLOWED: &[(&str, usize)] = &[("types/settings.rs", 1)];

#[test]
fn wire_types_hold_no_banned_type() {
    for (file, text) in FILES {
        let found = offenders(text);
        let allowed = ALLOWED
            .iter()
            .find(|(f, _)| f == file)
            .map_or(0, |(_, n)| *n);
        assert_eq!(found.len(), allowed, "{file}: {found:?}");
    }
}

#[test]
fn the_gate_fires_on_a_smuggled_hashmap_and_float() {
    let smuggled = "pub struct Evil {\n    pub m: std::collections::HashMap<String, u64>,\n    pub x: f64, // fine here\n}\n";
    assert_eq!(offenders(smuggled), vec![(2, "HashMap"), (3, "f64")]);
    assert!(offenders("// a comment naming usize and f32\n").is_empty());
}
