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
    declared_tier: true,
    glm5_next: false,
};
const DENSE: ModelKind = ModelKind {
    qwen_hybrid: true,
    fp8_moe: false,
    fp8_head: false,
    declared_tier: true,
    glm5_next: false,
};
const DENSE_NVFP4: ModelKind = ModelKind {
    declared_tier: false,
    ..DENSE
};

/// 2026-09-30: `adaptive` fixes nothing, so every model honours it and nothing is reported.
#[test]
fn adaptive_is_supported_everywhere() {
    for kind in [MOE, DENSE] {
        assert!(support(&v("adaptive"), kind).unwrap().is_empty());
    }
}

/// 2026-09-30: The default runs everywhere: the families a model lacks a path for are reported,
/// never refused, and a family the model does not have is not reported.
#[test]
fn declared_reports_what_a_model_does_not_honour() {
    assert!(support(&v("declared"), MOE).unwrap().is_empty());
    // 2026-09-30: The dense model under the declared weight tier honours every family it has.
    assert!(support(&v("declared"), DENSE).unwrap().is_empty());
    // 2026-09-30: Under the nvfp4 tier its projections are not the declared ones.
    assert_eq!(
        support(&v("declared"), DENSE_NVFP4).unwrap(),
        vec![ProjFamily::Gdn, ProjFamily::Attn, ProjFamily::Ffn]
    );
    assert!(
        support(&v("adaptive,ffn:nvfp4"), DENSE_NVFP4)
            .unwrap()
            .is_empty()
    );
    // 2026-09-30: Not a Qwen hybrid: only the LM head is honoured.
    let other = ModelKind {
        qwen_hybrid: false,
        fp8_moe: false,
        ..DENSE
    };
    assert_eq!(support(&v("declared"), other).unwrap().len(), 4);
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
    for s in [
        "adaptive,ffn:bf16",
        "adaptive,gdn:bf16",
        "adaptive,attn:nvfp4",
    ] {
        assert!(support(&v(s), DENSE).is_err(), "{s}");
    }
    assert!(support(&v("adaptive,gdn:fp8"), DENSE).is_ok());
}

/// 2026-10-01: The cross-sequence prefill levers are refused beside any fixed format, by their
/// resolved values, and never under `adaptive`.
#[test]
fn prefill_levers_are_refused_beside_a_fixed_format() {
    assert!(prefill_lever_refusal(&v("adaptive"), true, true, true).is_none());
    assert!(prefill_lever_refusal(&v("declared"), false, false, false).is_none());
    for (varlen, codispatch, first, needle) in [
        (true, false, false, "--prefill-varlen-batch"),
        (false, true, false, "--prefill-codispatch"),
        (false, false, true, "METRALE_Q12_BATCHED_FIRST_CHUNK"),
    ] {
        for s in ["declared", "adaptive,ffn:nvfp4"] {
            let why = prefill_lever_refusal(&v(s), varlen, codispatch, first).expect(s);
            assert!(why.contains(needle), "{s}: {why}");
        }
    }
    // 2026-10-01: Co-dispatch turns the batched first chunk on by itself; it is named once.
    let why = prefill_lever_refusal(&v("declared"), false, true, true).unwrap();
    assert!(!why.contains("Q12"), "{why}");
}

/// 2026-10-01: On the nvfp4 tier the dense GDN and attention projections honour a fixed `nvfp4`
/// (their weights are NVFP4) and nothing else; on another model kind `nvfp4` for those families
/// is refused, so the attention and GDN layers' mx arms engage only where every decode site
/// has one.
#[test]
fn nvfp4_attention_and_gdn_are_honoured_on_the_dense_nvfp4_tier_only() {
    for s in [
        "adaptive,gdn:nvfp4",
        "adaptive,attn:nvfp4",
        "nvfp4,lm_head:declared",
    ] {
        assert!(support(&v(s), DENSE_NVFP4).unwrap().is_empty(), "{s}");
    }
    assert_eq!(
        support(&v("adaptive,gdn:bf16"), DENSE_NVFP4).unwrap(),
        vec![ProjFamily::Gdn]
    );
    let other = ModelKind {
        qwen_hybrid: false,
        fp8_moe: false,
        ..DENSE_NVFP4
    };
    for (s, fam) in [
        ("adaptive,gdn:nvfp4", "gdn"),
        ("adaptive,attn:nvfp4", "attn"),
    ] {
        let e = format!("{:#}", support(&v(s), other).expect_err(s));
        assert!(e.contains(fam), "{s}: {e}");
    }
    assert_eq!(support(&v("declared"), other).unwrap().len(), 4);
}

/// 2026-10-08: GLM-5.3: the default honours its MLP families (declared W4A4) and reports the
/// families it has no row-invariant path for; fp8 on its MLP is refused, bf16 runs adaptive.
#[test]
fn glm_honours_its_declared_mlp_and_refuses_fp8() {
    let glm = ModelKind {
        qwen_hybrid: false,
        fp8_moe: false,
        fp8_head: false,
        declared_tier: true,
        glm5_next: true,
    };
    assert_eq!(
        support(&v("declared"), glm).unwrap(),
        vec![ProjFamily::Gdn, ProjFamily::Attn]
    );
    assert!(
        support(&v("adaptive,moe:nvfp4,ffn:nvfp4"), glm)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        support(&v("adaptive,moe:bf16"), glm).unwrap(),
        vec![ProjFamily::Moe]
    );
    for flag in ["adaptive,moe:fp8", "adaptive,ffn:fp8"] {
        let e = support(&v(flag), glm).unwrap_err().to_string();
        assert!(e.contains("FP8-activation MLP"), "{flag}: {e}");
    }
    // 2026-10-08: The same flags on a model that is not GLM keep their old answers.
    let other = ModelKind {
        glm5_next: false,
        ..glm
    };
    assert_eq!(
        support(&v("adaptive,moe:fp8"), other).unwrap(),
        vec![ProjFamily::Moe]
    );
}
