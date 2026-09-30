// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Checks, in every target tree, that an entry point declares
//! `extern "C" __device__ unsigned int <entry>_a_e4m3 = 1;` exactly when it casts its BF16 A
//! operand to E4M3 with no scale.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! An unscaled cast saturates past 448. The runtime reads the symbol when a handle is
//! resolved (`GpuBackend::kernel_casts_a_to_e4m3`), and debug builds check the activation
//! range of every launch through a dense launcher that declares it. A casting kernel without
//! the symbol would go unchecked, and a symbol on a kernel that keeps A in BF16 would refuse
//! values it handles, so both directions are failures. The cast is recognised by the
//! `bf16x4_to_e4m3x4` helper every such tree uses, in the entry's body or in a macro or
//! `*_impl` helper the body calls.

use metrale_closure::layout::{discover, walk};

#[path = "support/cu_source.rs"]
mod cu_source;
use cu_source::{expanded_body, global_entries, workspace_root};

const CAST: &str = "bf16x4_to_e4m3x4(";

/// 2026-09-29: Whether `text` sets `<entry>_a_e4m3` to a non-zero value.
fn declares_a_e4m3(text: &str, entry: &str) -> bool {
    let decl = format!("unsigned int {entry}_a_e4m3 =");
    text.find(&decl).is_some_and(|at| {
        let value = text[at + decl.len()..]
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        value.parse::<u32>().is_ok_and(|v| v != 0)
    })
}

#[test]
fn every_entry_declares_an_unscaled_e4m3_a_cast_exactly_when_it_makes_one() {
    let root = workspace_root();
    let (mut casting, mut failures) = (0usize, Vec::new());
    for target in walk(&root).expect("the tree resolves") {
        let layout = discover(&root, &target).unwrap_or_else(|e| panic!("{target}: {e}"));
        for (_stem, entry) in layout.modules() {
            let Ok(text) = std::fs::read_to_string(&entry.source) else {
                continue;
            };
            for (name, _) in global_entries(&text) {
                let casts = expanded_body(&text, &name).is_some_and(|b| b.contains(CAST));
                let declared = declares_a_e4m3(&text, &name);
                casting += usize::from(casts);
                if casts != declared {
                    failures.push(format!(
                        "{target}: {}: {name} casts A to E4M3: {casts}, declares `_a_e4m3`: {declared}",
                        entry.source.display()
                    ));
                }
            }
        }
    }
    assert!(
        casting > 50,
        "only {casting} casting (target, entry) pairs found"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 2026-09-29: The cast is found through a `\`-continued macro and a templated `*_impl`
/// helper, a `__launch_bounds__` entry is read, and a kernel that keeps A in BF16 must not
/// declare it.
#[test]
fn the_scanner_finds_the_cast_through_macros_and_helpers() {
    let text = "#define STEP(p) \\\n    unsigned a0 = bf16x4_to_e4m3x4(p); \\\n    mma(a0);\n\
        extern \"C\" __global__ void via_macro(int n) { STEP(&sA[0]); }\n\
        template <int G> __device__ void cast_impl(int n) { unsigned a = bf16x4_to_e4m3x4(&sA[n]); }\n\
        extern \"C\" __global__\n__launch_bounds__(128, 3)\nvoid via_helper(int n) { cast_impl<32>(n); }\n\
        extern \"C\" __global__ void bf16_only(int n) { mma_bf16(n); }\n\
        extern \"C\" __device__ unsigned int via_macro_a_e4m3 = 1;\n\
        extern \"C\" __device__ unsigned int bf16_only_a_e4m3 = 0;\n";
    let names: Vec<String> = global_entries(text).into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["via_macro", "via_helper", "bf16_only"]);
    for (name, casts) in [
        ("via_macro", true),
        ("via_helper", true),
        ("bf16_only", false),
    ] {
        assert_eq!(
            expanded_body(text, name).unwrap().contains(CAST),
            casts,
            "{name}"
        );
    }
    assert!(declares_a_e4m3(text, "via_macro"));
    assert!(!declares_a_e4m3(text, "via_helper"));
    assert!(
        !declares_a_e4m3(text, "bf16_only"),
        "0 is not a declaration"
    );
}
