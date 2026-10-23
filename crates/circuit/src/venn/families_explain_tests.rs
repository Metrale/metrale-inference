// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The explanation checks: each refusal on its own, and the accepted form.

use super::*;

fn file(about: &str, numerics: &str, math: &[&str]) -> ExplainFile {
    ExplainFile {
        about: about.into(),
        numerics: numerics.into(),
        math: math.iter().map(|m| m.to_string()).collect(),
    }
}

#[test]
fn a_complete_explanation_loads_with_its_whitespace_collapsed() {
    let e = file(
        "One block per row,\n  the mean square $\\overline{x^2}$ in FP32.",
        "RNE to BF16.",
        &["  y_i = \\frac{x_i}{\\sqrt{\\overline{x^2}+\\epsilon}}  "],
    )
    .validate()
    .unwrap();
    assert_eq!(
        e.about,
        "One block per row, the mean square $\\overline{x^2}$ in FP32."
    );
    assert_eq!(
        e.math,
        ["y_i = \\frac{x_i}{\\sqrt{\\overline{x^2}+\\epsilon}}"]
    );
}

#[test]
fn every_part_is_required() {
    let ok = "y = x";
    for (bad, why) in [
        (file("", "n", &[ok]), "explain.about is empty"),
        (file("a", " \n", &[ok]), "explain.numerics is empty"),
        (file("a", "n", &[]), "explain.math is empty"),
        (file("a", "n", &["  "]), "empty math"),
    ] {
        let e = bad.validate().unwrap_err();
        assert!(e.contains(why), "{e}");
    }
}

#[test]
fn malformed_latex_is_refused() {
    for (math, why) in [
        ("\\frac{a}{b", "unbalanced braces"),
        ("a}", "without"),
        ("\\begin{aligned} a", "\\begin/\\end"),
        ("x = $y$", "no `$`"),
        ("\u{3b1} = 1", "non-ASCII"),
    ] {
        let e = file("a", "n", &[math]).validate().unwrap_err();
        assert!(e.contains(why), "{math}: {e}");
    }
    // escaped braces are text, not grouping
    file("a", "n", &["\\{x\\} \\\\ \\{"]).validate().unwrap();
    let e = file("a $x", "n", &["y"]).validate().unwrap_err();
    assert!(e.contains("unpaired"), "{e}");
    let e = file("a $\\frac{x$ b", "n", &["y"]).validate().unwrap_err();
    assert!(e.contains("unbalanced"), "{e}");
}

#[test]
fn private_names_are_refused_in_prose_and_math() {
    for (about, math) in [
        ("see rms_norm.cu for the tiling", "y"),
        ("defined in kernels/gb10/common", "y"),
        ("fine", "\\text{crates/circuit}"),
    ] {
        let e = file(about, "n", &[math]).validate().unwrap_err();
        assert!(e.contains("published"), "{e}");
    }
}
