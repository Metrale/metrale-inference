// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The schema refuses every incomplete or contradictory contract.

use super::*;

const OK: &str = r#"
schema = 1
hardware = "gb10"
seed = 7
[[contract]]
family = "w4a16_gemv"
kernels = ["w4a16_gemv::w4a16_gemv_sw"]
op = "linear"
reference = "linear"
class = "derived"
reduction.k = [{ level = "thread", width = "k/2048", order = "sequential" }]
reduction.group = [{ level = "thread", width = "16", order = "sequential" }]
ftz = false
approx = { exp2 = "2^-22" }
scale_fold = "group"
inputs = ["gaussian", "outliers"]
mutations = ["corrupt_block_scale", "accumulate:bf16"]
"#;

#[test]
fn a_complete_contract_parses() {
    let c = parse_contracts(OK).unwrap();
    assert_eq!(c.seed, 7);
    let k = &c.contracts[0];
    assert_eq!(k.class, Class::Derived);
    assert_eq!(k.scale_fold, ScaleFold::Group);
    assert_eq!(k.approx["exp2"], 2f64.powi(-22));
    assert_eq!(k.mutations.len(), 2);
}

#[test]
fn incomplete_or_contradictory_contracts_are_refused() {
    let cases = [
        ("ftz = false\n", "", "a derived contract states `ftz`"),
        (
            "class = \"derived\"",
            "class = \"bit_identical\"",
            "needs `against`",
        ),
        (
            "mutations = [\"corrupt_block_scale\", \"accumulate:bf16\"]",
            "mutations = []",
            "cannot prove",
        ),
        (
            "inputs = [\"gaussian\", \"outliers\"]",
            "inputs = [\"outliers\"]",
            "gaussian",
        ),
        (
            "scale_fold = \"group\"",
            "scale_fold = \"sometimes\"",
            "scale_fold",
        ),
        (
            "order = \"sequential\" }]\nreduction.group",
            "order = \"zigzag\" }]\nreduction.group",
            "order",
        ),
        ("\"accumulate:bf16\"]", "\"accumulate:f32\"]", "mutation"),
        ("seed = 7", "seed = 7\nextra = 1", "unknown field"),
    ];
    for (from, to, why) in cases {
        let text = OK.replacen(from, to, 1);
        assert_ne!(text, OK, "the edit `{from}` did not apply");
        let e = parse_contracts(&text).expect_err(why).to_string();
        assert!(e.contains(why), "{why}: got `{e}`");
    }
    let bit = OK.replacen(
        "class = \"derived\"",
        "class = \"bit_identical\"\nagainst = \"w4a16_gemv::w4a16_gemv_qg\"",
        1,
    );
    let e = parse_contracts(&bit).unwrap_err().to_string();
    assert!(e.contains("for derived contracts only"), "{e}");
}
