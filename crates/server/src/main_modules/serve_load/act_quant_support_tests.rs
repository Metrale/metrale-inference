// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The load-time support table of `--activation-quantization`.

use super::*;

fn v(s: &str) -> ActivationQuantization {
    ActivationQuantization::parse(s).expect("valid")
}

const MOE: ModelKind = ModelKind {
    qwen_hybrid: true,
    fp8_moe: true,
    fp8_head: false,
};
const DENSE: ModelKind = ModelKind {
    qwen_hybrid: true,
    fp8_moe: false,
    fp8_head: false,
};

/// 2026-09-30: `adaptive` fixes nothing, so every model honours it and nothing is reported.
#[test]
fn adaptive_is_supported_everywhere() {
    for kind in [MOE, DENSE] {
        assert!(support(&v("adaptive"), kind).unwrap().is_empty());
    }
}

/// 2026-09-30: The default runs everywhere: the families a model lacks a path for are reported,
/// never refused.
#[test]
fn declared_reports_what_a_model_does_not_honour() {
    assert!(
        support(&v("declared"), MOE)
            .unwrap()
            .contains(&ProjFamily::Ffn)
    );
    let dense = support(&v("declared"), DENSE).unwrap();
    for f in [
        ProjFamily::Gdn,
        ProjFamily::Attn,
        ProjFamily::Ffn,
        ProjFamily::Moe,
    ] {
        assert!(dense.contains(&f), "{f:?}");
    }
    assert!(!dense.contains(&ProjFamily::LmHead));
}

/// 2026-09-30: A format the checkpoint cannot run is refused, naming the family and the reason.
#[test]
fn impossible_formats_are_refused() {
    for (s, fam) in [
        ("adaptive,moe:nvfp4", "moe"),
        ("adaptive,gdn:fp8", "gdn"),
        ("adaptive,attn:nvfp4", "attn"),
        ("adaptive,lm_head:fp8", "lm_head"),
        ("adaptive,moe:1=bf16;2-=fp8", "moe"),
    ] {
        let e = format!("{:#}", support(&v(s), MOE).expect_err(s));
        assert!(e.contains(fam), "{s}: {e}");
    }
    let fp8_head = ModelKind {
        fp8_head: true,
        ..MOE
    };
    assert!(support(&v("adaptive,lm_head:fp8"), fp8_head).is_ok());
    assert!(support(&v("adaptive,moe:fp8"), MOE).is_ok());
    assert!(support(&v("adaptive,moe:fp8"), DENSE).is_ok());
}
