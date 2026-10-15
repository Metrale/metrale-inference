// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `kkb-embed/1` and embedding spaces (PLAN W1.1, ADDENDUM E1/E2).

use metprof_kb::embed::{Embedder, Embedding, HASHED_DIM, Hashed, Space, fnv1a64, hashed};

/// The non-zero (index, value) entries of five texts, frozen at the first
/// computation (the weights were checked by hand: ln 4 / ln 2 = 2 and
/// ln 3 / ln 2 = 1.585 in the last one). A change here re-embeds every point.
const GOLDEN: [(&str, &[(usize, f32)]); 5] = [
    (
        "flash_fwd_kernel",
        &[
            (2, -0.3015113),
            (5, 0.3015113),
            (25, 0.3015113),
            (41, -0.3015113),
            (100, 0.3015113),
            (118, -0.3015113),
            (142, -0.3015113),
            (199, 0.3015113),
            (208, -0.3015113),
            (214, 0.3015113),
            (239, 0.3015113),
        ],
    ),
    (
        "fused moe gemm",
        &[
            (25, -0.3162278),
            (38, 0.3162278),
            (83, -0.3162278),
            (127, -0.3162278),
            (171, 0.3162278),
            (175, 0.3162278),
            (194, 0.3162278),
            (218, 0.3162278),
            (232, -0.3162278),
            (234, -0.3162278),
        ],
    ),
    (
        "rms_norm::fused_add",
        &[
            (17, -0.3015113),
            (25, -0.3015113),
            (103, 0.3015113),
            (118, 0.3015113),
            (127, -0.3015113),
            (159, 0.3015113),
            (175, 0.3015113),
            (188, -0.3015113),
            (195, 0.3015113),
            (197, 0.3015113),
            (218, 0.3015113),
        ],
    ),
    (
        "attention decode split-k",
        &[
            (19, 0.2182179),
            (24, -0.2182179),
            (29, 0.2182179),
            (46, -0.2182179),
            (65, -0.2182179),
            (67, -0.2182179),
            (73, 0.2182179),
            (79, 0.2182179),
            (142, 0.2182179),
            (146, 0.2182179),
            (170, 0.2182179),
            (192, 0.2182179),
            (216, 0.2182179),
            (222, -0.2182179),
            (234, 0.4364358),
            (245, -0.2182179),
            (247, 0.2182179),
            (252, -0.2182179),
        ],
    ),
    (
        "gemm gemm gemm tile",
        &[
            (17, 0.2207979),
            (48, 0.4415957),
            (111, 0.3499563),
            (194, 0.4415957),
            (232, -0.4415957),
            (234, -0.4415957),
            (253, -0.2207979),
        ],
    ),
];

#[test]
fn fnv1a64_matches_the_published_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn golden_vectors_are_stable() {
    for (text, sparse) in GOLDEN {
        let v = hashed(text).expect("embeds");
        assert_eq!(v.len(), HASHED_DIM);
        let mut want = vec![0f32; HASHED_DIM];
        for (i, x) in sparse {
            want[*i] = *x;
        }
        for (i, (got, want)) in v.iter().zip(&want).enumerate() {
            assert!((got - want).abs() < 1e-6, "{text:?}[{i}]: {got} vs {want}");
        }
    }
}

#[test]
fn vectors_are_unit_and_cosine_is_symmetric() {
    let e = Hashed::new();
    let texts: Vec<&str> = GOLDEN.iter().map(|(t, _)| *t).collect();
    let vs = e.embed(&texts).expect("embeds");
    for v in &vs {
        assert_eq!(v.space.id(), "kkb-embed/1");
        let n: f64 = v
            .vector
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((n - 1.0).abs() < 1e-6, "norm {n}");
        assert!((v.cosine(v).expect("same space") - 1.0).abs() < 1e-6);
    }
    for a in &vs {
        for b in &vs {
            let (ab, ba) = (a.cosine(b).expect("ab"), b.cosine(a).expect("ba"));
            assert!((ab - ba).abs() < 1e-7);
        }
    }
}

#[test]
fn token_order_changes_the_vector() {
    let a = hashed("fused moe gemm").expect("a");
    let b = hashed("gemm moe fused").expect("b");
    assert_ne!(a, b, "bigrams must make word order visible");
}

#[test]
fn a_text_without_features_is_refused() {
    assert!(hashed("  -- // ").is_err());
}

#[test]
fn vectors_of_two_spaces_never_mix() {
    let hashed_vec = Hashed::new().embed(&["fused moe gemm"]).expect("embeds");
    let dense = Space::dense("openrouter", "vendor/embed-a", HASHED_DIM).expect("valid");
    // Same dimension and identical numbers: still refused, by space id.
    let foreign = Embedding {
        space: dense.clone(),
        vector: hashed_vec[0].vector.clone(),
    };
    let err = hashed_vec[0].cosine(&foreign).expect_err("mixed spaces");
    assert!(err.0.contains("never mix"), "{}", err.0);
    assert!(Space::hashed().require(&dense).is_err());
    let other_model = Space::dense("openrouter", "vendor/embed-b", HASHED_DIM).expect("valid");
    assert!(dense.require(&other_model).is_err());
    assert!(dense.require(&dense.clone()).is_ok());
}

#[test]
fn space_ids_parse_back_to_themselves() {
    for id in [
        "kkb-embed/1",
        "openai_compatible:openrouter:vendor/model:free@2048",
        "openai_compatible:127.0.0.1:local-embed@1024",
        "openai_compatible:[::1]:local-embed@8",
    ] {
        assert_eq!(Space::parse(id).expect(id).id(), id);
    }
    for bad in [
        "kkb-embed/2",
        "dense:model@8",
        "openai_compatible:host:model@0",
        "openai_compatible:host:model@x",
        "openai_compatible:Host:model@8",
        "openai_compatible:host:@8",
    ] {
        assert!(Space::parse(bad).is_err(), "{bad}");
    }
}
