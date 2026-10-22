// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The production trust anchor and the three CI certificates (public data only, the
//! files committed under `anchors/` and `certs/`). Nothing here signs with a production key:
//! the secrets live only in GitHub. A chain is shown to pass by verifying the certificate's
//! own signature, and by `verify_signed` failing only at the last step (the payload signature,
//! which a placeholder cannot satisfy); every earlier refusal has its own code. Every instant is
//! a fixed clock, never wall time.

use metrale_abstract_io::UnixMs;
use metrale_blueprint_trust::{
    ANCHORS, Anchor, HybridSig, KeyCert, Kid, Purpose, Revoked, Scope, SigBlock, SignedCert,
    Signer, Suite, TrustAnchors, TrustPolicy, certify, verify_hybrid, verify_signed,
};
use metrale_error::ErrorCode;

/// 2026-10-22: 2026-10-22T00:00:00Z.
const OCT_22_2026_MS: i64 = 1_792_627_200_000;
const DAY_MS: i64 = 86_400_000;

const CERTS: [(&str, &[u8], Scope); 3] = [
    (
        "kbp-blueprints-2026-10",
        include_bytes!("../certs/kbp-blueprints-2026-10.cert"),
        Scope::Blueprints,
    ),
    (
        "kbp-artifacts-2026-10",
        include_bytes!("../certs/kbp-artifacts-2026-10.cert"),
        Scope::Artifacts,
    ),
    (
        "kbp-index-2026-10",
        include_bytes!("../certs/kbp-index-2026-10.cert"),
        Scope::Index,
    ),
];

const SCOPES: [(Scope, Purpose); 3] = [
    (Scope::Blueprints, Purpose::KbpBody),
    (Scope::Artifacts, Purpose::KernelArtifact),
    (Scope::Index, Purpose::KernelIndex),
];

fn signed_cert(bytes: &[u8]) -> SignedCert {
    postcard::from_bytes(bytes).unwrap()
}

fn key_cert(c: &SignedCert) -> KeyCert {
    postcard::from_bytes(&c.cert).unwrap()
}

/// 2026-10-22: A block naming the certified key, carrying its certificate, and a placeholder
/// payload signature: it can pass every check except the last.
fn block(c: &SignedCert) -> SigBlock {
    SigBlock {
        kid: c.subject.clone(),
        chain: vec![c.clone()],
        sig: HybridSig {
            mldsa: vec![0; 3309],
            ed25519: vec![0; 64],
        },
    }
}

fn verify(anchors: &TrustAnchors, b: &SigBlock, scope: Scope, now: i64) -> Option<ErrorCode> {
    let revoked = Revoked::embedded().unwrap();
    let policy = TrustPolicy {
        anchors,
        revoked: &revoked,
        now: UnixMs(now),
    };
    let purpose = SCOPES.iter().find(|(s, _)| *s == scope).unwrap().1;
    verify_signed(&policy, b, scope, purpose, b"payload", "test")
        .err()
        .map(|e| e.code())
}

fn check(anchors: &TrustAnchors, c: &SignedCert, scope: Scope, now: i64) -> ErrorCode {
    verify(anchors, &block(c), scope, now).expect("a placeholder never verifies")
}

#[test]
fn the_compiled_anchor_is_the_committed_root_file() {
    let file: Anchor =
        postcard::from_bytes(include_bytes!("../anchors/kbp-root-2026.anchor")).unwrap();
    assert_eq!(ANCHORS.len(), 1);
    let compiled = TrustAnchors::compiled().unwrap();
    let root = compiled.get(&Kid::new("kbp-root-2026").unwrap()).unwrap();
    assert_eq!(root, &file, "the const decode equals the postcard decode");
    assert_eq!(root.keys.suite, Suite::MlDsa65Ed25519);
    assert_eq!(root.keys.mldsa.len(), 1952, "an ML-DSA-65 verifying key");
}

#[test]
fn each_certificate_chains_to_the_compiled_anchor() {
    let anchors = TrustAnchors::compiled().unwrap();
    for (name, bytes, scope) in CERTS {
        let c = signed_cert(bytes);
        assert_eq!(c.subject.as_str(), name);
        assert_eq!(c.issuer.as_str(), "kbp-root-2026");
        let root = anchors.get(&c.issuer).expect("issued by a compiled anchor");
        verify_hybrid(
            &root.keys,
            Purpose::KeyCert,
            &c.cert,
            &c.sig,
            "certificate",
            &c.issuer,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let k = key_cert(&c);
        assert_eq!((k.kid.as_str(), k.scope), (name, scope));
        assert_eq!(k.keys.suite, Suite::MlDsa65Ed25519);
        assert_eq!(
            check(&anchors, &c, scope, k.not_before_ms),
            ErrorCode::BlueprintSignature,
            "{name}: chain, revocation, window and scope pass; only the placeholder fails"
        );
    }
}

#[test]
fn each_certificate_signs_only_its_own_scope() {
    let anchors = TrustAnchors::compiled().unwrap();
    for (name, bytes, own) in CERTS {
        let c = signed_cert(bytes);
        let now = key_cert(&c).not_before_ms;
        for (scope, _) in SCOPES {
            let want = if scope == own {
                ErrorCode::BlueprintSignature
            } else {
                ErrorCode::BlueprintScope
            };
            assert_eq!(
                check(&anchors, &c, scope, now),
                want,
                "{name} for {scope:?}"
            );
        }
    }
}

#[test]
fn each_certificate_is_valid_for_ninety_days_from_2026_10_22() {
    let anchors = TrustAnchors::compiled().unwrap();
    for (name, bytes, scope) in CERTS {
        let c = signed_cert(bytes);
        let k = key_cert(&c);
        assert!(
            (OCT_22_2026_MS..OCT_22_2026_MS + DAY_MS).contains(&k.not_before_ms),
            "{name}: issued on 2026-10-22 UTC ({})",
            k.not_before_ms
        );
        assert_eq!(k.not_after_ms - k.not_before_ms, 90 * DAY_MS, "{name}");
        let at = |now| check(&anchors, &c, scope, now);
        assert_eq!(
            at(k.not_before_ms - 1),
            ErrorCode::BlueprintCertExpired,
            "{name}"
        );
        assert_eq!(at(k.not_before_ms), ErrorCode::BlueprintSignature, "{name}");
        assert_eq!(
            at(k.not_after_ms - 1),
            ErrorCode::BlueprintSignature,
            "{name}"
        );
        assert_eq!(
            at(k.not_after_ms),
            ErrorCode::BlueprintCertExpired,
            "{name}"
        );
        assert_eq!(
            at(OCT_22_2026_MS + 91 * DAY_MS),
            ErrorCode::BlueprintCertExpired,
            "{name}: expired by 2027-01-21"
        );
    }
}

#[test]
fn an_unknown_anchor_is_refused() {
    let c = signed_cert(CERTS[0].1);
    let now = key_cert(&c).not_before_ms;
    let none = TrustAnchors::new(vec![]).unwrap();
    assert_eq!(
        check(&none, &c, Scope::Blueprints, now),
        ErrorCode::BlueprintUntrusted
    );

    // 2026-10-22: A stranger root that takes the production root's name certifies a key it
    // holds under the production kid, and that key signs the payload correctly: only the
    // certificate's signature, checked against the compiled key, stands in the way.
    let stranger = Signer::from_seeds(
        Kid::new("kbp-root-2026").unwrap(),
        Suite::MlDsa65Ed25519,
        &[7; 32],
        &[8; 32],
    );
    let leaf = Signer::from_seeds(c.subject.clone(), Suite::MlDsa65Ed25519, &[5; 32], &[6; 32]);
    let forged = SigBlock {
        kid: c.subject.clone(),
        chain: vec![
            certify(
                &stranger,
                &KeyCert {
                    keys: leaf.public(),
                    ..key_cert(&c)
                },
            )
            .unwrap(),
        ],
        sig: leaf.sign(Purpose::KbpBody, b"payload").unwrap(),
    };
    let compiled = TrustAnchors::compiled().unwrap();
    assert_eq!(
        verify(&compiled, &forged, Scope::Blueprints, now),
        Some(ErrorCode::BlueprintSignature),
        "a certificate the real root did not sign"
    );
    let impostor = TrustAnchors::new(vec![Anchor {
        kid: stranger.kid().clone(),
        keys: stranger.public(),
    }])
    .unwrap();
    assert_eq!(
        check(&impostor, &c, Scope::Blueprints, now),
        ErrorCode::BlueprintSignature,
        "the real certificate under an impostor anchor"
    );
}
