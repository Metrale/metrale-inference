// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The trust chain: one positive path and one refusal per step of `verify_signed`,
//! each asserting its specific code. Keys are generated here from fixed test seeds; none is a
//! real key.

use metrale_abstract_io::UnixMs;
use metrale_blueprint_trust::{
    Anchor, KeyCert, Kid, Purpose, Revocations, Revoked, Scope, SigBlock, Signer, Suite,
    TrustAnchors, TrustPolicy, certify, sign_revocations, verify_revocations, verify_signed,
};
use metrale_error::ErrorCode;

const NOW: i64 = 1_760_000_000_000;

struct World {
    root: Signer,
    leaf: Signer,
    anchors: TrustAnchors,
}

fn kid(s: &str) -> Kid {
    Kid::new(s).unwrap()
}

fn world() -> World {
    let root = Signer::from_seeds(kid("test-root"), Suite::MlDsa87Ed25519, &[1; 32], &[2; 32]);
    let leaf = Signer::from_seeds(
        kid("test-blueprints"),
        Suite::MlDsa65Ed25519,
        &[3; 32],
        &[4; 32],
    );
    let anchors = TrustAnchors::new(vec![Anchor {
        kid: root.kid().clone(),
        keys: root.public(),
    }])
    .unwrap();
    World {
        root,
        leaf,
        anchors,
    }
}

fn cert(w: &World, scope: Scope, window: (i64, i64)) -> KeyCert {
    KeyCert {
        kid: w.leaf.kid().clone(),
        keys: w.leaf.public(),
        scope,
        not_before_ms: window.0,
        not_after_ms: window.1,
        serial: 1,
    }
}

fn block(w: &World, c: &KeyCert, msg: &[u8]) -> SigBlock {
    SigBlock {
        kid: w.leaf.kid().clone(),
        chain: vec![certify(&w.root, c).unwrap()],
        sig: w.leaf.sign(Purpose::KbpBody, msg).unwrap(),
    }
}

fn run(w: &World, b: &SigBlock, revoked: &Revoked, msg: &[u8]) -> Result<Kid, ErrorCode> {
    let p = TrustPolicy {
        anchors: &w.anchors,
        revoked,
        now: UnixMs(NOW),
    };
    verify_signed(&p, b, Scope::Blueprints, Purpose::KbpBody, msg, "body")
        .map(|v| v.anchor)
        .map_err(|e| e.code())
}

fn good(w: &World) -> KeyCert {
    cert(w, Scope::Blueprints, (NOW - 1, NOW + 1))
}

#[test]
fn a_certified_signature_verifies_and_names_its_anchor() {
    let w = world();
    let b = block(&w, &good(&w), b"body");
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
        Ok(kid("test-root"))
    );
}

#[test]
fn a_tampered_payload_is_a_signature_error() {
    let w = world();
    let b = block(&w, &good(&w), b"body");
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"bodx"),
        Err(ErrorCode::BlueprintSignature)
    );
}

#[test]
fn an_unknown_kid_is_untrusted() {
    let w = world();
    let mut b = block(&w, &good(&w), b"body");
    b.kid = kid("someone-else");
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
        Err(ErrorCode::BlueprintUntrusted)
    );
}

#[test]
fn a_certificate_from_a_root_that_is_not_an_anchor_is_untrusted() {
    let w = world();
    let rogue = Signer::from_seeds(kid("rogue-root"), Suite::MlDsa87Ed25519, &[9; 32], &[9; 32]);
    let mut b = block(&w, &good(&w), b"body");
    b.chain = vec![certify(&rogue, &good(&w)).unwrap()];
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
        Err(ErrorCode::BlueprintUntrusted)
    );
}

#[test]
fn a_rogue_root_reusing_the_anchor_kid_fails_the_certificate_signature() {
    let w = world();
    let rogue = Signer::from_seeds(kid("test-root"), Suite::MlDsa87Ed25519, &[9; 32], &[9; 32]);
    let mut b = block(&w, &good(&w), b"body");
    b.chain = vec![certify(&rogue, &good(&w)).unwrap()];
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
        Err(ErrorCode::BlueprintSignature)
    );
}

#[test]
fn a_revoked_kid_is_refused() {
    let w = world();
    let b = block(&w, &good(&w), b"body");
    let signed = sign_revocations(
        &w.root,
        &Revocations {
            serial: 3,
            revoked: vec![kid("test-blueprints")],
        },
    )
    .unwrap();
    let revoked = verify_revocations(&w.anchors, &signed, 0).unwrap();
    assert_eq!(
        run(&w, &b, &revoked, b"body"),
        Err(ErrorCode::BlueprintRevoked)
    );
}

#[test]
fn an_expired_and_a_not_yet_valid_certificate_are_refused() {
    let w = world();
    for window in [(NOW - 10, NOW), (NOW + 1, NOW + 10)] {
        let b = block(&w, &cert(&w, Scope::Blueprints, window), b"body");
        assert_eq!(
            run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
            Err(ErrorCode::BlueprintCertExpired)
        );
    }
}

#[test]
fn a_key_of_another_scope_is_refused() {
    let w = world();
    let b = block(&w, &cert(&w, Scope::Artifacts, (NOW - 1, NOW + 1)), b"body");
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
        Err(ErrorCode::BlueprintScope)
    );
}

#[test]
fn a_revocation_list_below_the_floor_is_a_rollback() {
    let w = world();
    let old = sign_revocations(
        &w.root,
        &Revocations {
            serial: 4,
            revoked: vec![],
        },
    )
    .unwrap();
    assert_eq!(
        verify_revocations(&w.anchors, &old, 5)
            .map(|_| ())
            .map_err(|e| e.code()),
        Err(ErrorCode::BlueprintRollback)
    );
    assert!(
        verify_revocations(&w.anchors, &old, 4).is_ok(),
        "the floor itself is accepted"
    );
}

#[test]
fn a_revocation_list_signed_by_a_signing_key_is_untrusted() {
    let w = world();
    let forged = sign_revocations(
        &w.leaf,
        &Revocations {
            serial: 9,
            revoked: vec![],
        },
    )
    .unwrap();
    assert_eq!(
        verify_revocations(&w.anchors, &forged, 0)
            .map(|_| ())
            .map_err(|e| e.code()),
        Err(ErrorCode::BlueprintUntrusted)
    );
}

/// 2026-10-22: Either half alone is not enough: a block whose ML-DSA half is valid but whose
/// Ed25519 half is another key's is refused, and so is the converse.
#[test]
fn both_halves_must_verify() {
    let w = world();
    let other = Signer::from_seeds(
        kid("test-blueprints"),
        Suite::MlDsa65Ed25519,
        &[5; 32],
        &[6; 32],
    );
    let real = w.leaf.sign(Purpose::KbpBody, b"body").unwrap();
    let forged = other.sign(Purpose::KbpBody, b"body").unwrap();
    for (mldsa, ed) in [(&real, &forged), (&forged, &real)] {
        let mut b = block(&w, &good(&w), b"body");
        b.sig.mldsa = mldsa.mldsa.clone();
        b.sig.ed25519 = ed.ed25519.clone();
        assert_eq!(
            run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
            Err(ErrorCode::BlueprintSignature)
        );
    }
}

#[test]
fn a_signature_for_another_purpose_does_not_verify() {
    let w = world();
    let mut b = block(&w, &good(&w), b"body");
    b.sig = w.leaf.sign(Purpose::KernelIndex, b"body").unwrap();
    assert_eq!(
        run(&w, &b, &Revoked::embedded().unwrap(), b"body"),
        Err(ErrorCode::BlueprintSignature)
    );
}

#[test]
fn duplicate_anchor_kids_and_bad_kids_are_refused() {
    let w = world();
    let a = Anchor {
        kid: w.root.kid().clone(),
        keys: w.root.public(),
    };
    assert!(TrustAnchors::new(vec![a.clone(), a]).is_err());
    for bad in ["", "Upper", "has space", &"x".repeat(65)] {
        assert_eq!(
            Kid::new(bad).map_err(|e| e.code()),
            Err(ErrorCode::BlueprintKey),
            "{bad:?}"
        );
    }
}
