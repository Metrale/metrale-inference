// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Every format spelling the circuits use explains itself with well-formed LaTeX,
//! and the explanation states the parameters of its spelling.

use crate::format::Format;
use crate::venn::families::explain::{prose, tex};

const SPELLINGS: [&str; 10] = [
    "bf16",
    "f32",
    "i32",
    "fp8/tensor",
    "fp8/token",
    "fp8/channel",
    "fp8/g128",
    "fp8/block128x128",
    "nvfp4/g16",
    "mxfp4/g32",
];

#[test]
fn every_format_explains_itself_well_formed() {
    for s in SPELLINGS {
        let e = Format::parse(s).unwrap().explain();
        prose(s, &e.title).unwrap();
        prose(s, &e.description).unwrap();
        assert!(!e.math.is_empty(), "{s}");
        for m in e.math.iter().chain(&e.encode) {
            tex(s, m).unwrap_or_else(|err| panic!("{s}: {err}"));
            assert!(!m.contains('$'), "{s}: display math");
        }
    }
}

#[test]
fn the_explanation_follows_the_parameters() {
    let g16 = Format::parse("nvfp4/g16").unwrap().explain();
    let g32 = Format::parse("nvfp4/g32").unwrap().explain();
    assert!(g16.math[1].contains("j/16") && g32.math[1].contains("j/32"));
    assert_ne!(g16.title, g32.title);
    let token = Format::parse("fp8/token").unwrap().explain();
    let group = Format::parse("fp8/g128").unwrap().explain();
    assert!(token.math[1].contains("s_{r}"), "{}", token.math[1]);
    assert!(group.math[1].contains("j/128"), "{}", group.math[1]);
    let enc = token.encode.unwrap();
    assert!(enc.contains("448") && enc.contains("10^{-12}"), "{enc}");
    assert!(Format::F32.explain().encode.is_none());
    assert!(
        Format::parse("mxfp4/g32")
            .unwrap()
            .explain()
            .encode
            .is_none()
    );
}
