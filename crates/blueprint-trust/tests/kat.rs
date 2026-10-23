// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Known-answer tests for both halves of the hybrid signature, and context
//! separation from metrale-auth.
//!
//! The `kat/` files are NIST ACVP ML-DSA vectors (FIPS 204; keyGen tgId 2 / tcId 26, and the
//! sigVer external/pure/with-context pass and "modified signature - z" fail cases), the same
//! files metrale-auth tests against. Each line is one hex field.

use metrale_blueprint_trust::{Kid, Purpose, Signer, Suite, lv_cat};
use ml_dsa::{B32, EncodedVerifyingKey, MlDsa65, Signature, SigningKey, VerifyingKey};

fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn lines(text: &str) -> Vec<Vec<u8>> {
    text.lines().map(unhex).collect()
}

fn signer(seed: u8) -> Signer {
    Signer::from_seeds(
        Kid::new("kat").unwrap(),
        Suite::MlDsa65Ed25519,
        &[seed; 32],
        &[seed; 32],
    )
}

#[test]
fn ml_dsa_65_keygen_matches_the_acvp_vector() {
    let l = lines(include_str!("../kat/mldsa65-keygen-tc.txt"));
    let seed: [u8; 32] = l[0].clone().try_into().unwrap();
    let s = Signer::from_seeds(
        Kid::new("kat").unwrap(),
        Suite::MlDsa65Ed25519,
        &seed,
        &[0; 32],
    );
    assert_eq!(s.public().mldsa, l[1]);
}

#[test]
fn ml_dsa_65_sigver_accepts_the_valid_case_and_refuses_the_modified_one() {
    for (file, expected) in [
        (include_str!("../kat/mldsa65-sigver-pass.txt"), true),
        (include_str!("../kat/mldsa65-sigver-fail.txt"), false),
    ] {
        let l = lines(file);
        let enc = EncodedVerifyingKey::<MlDsa65>::try_from(l[0].as_slice()).unwrap();
        let vk = VerifyingKey::<MlDsa65>::decode(&enc);
        let ok = Signature::<MlDsa65>::try_from(l[3].as_slice())
            .is_ok_and(|s| vk.verify_with_context(&l[1], &l[2], &s));
        assert_eq!(ok, expected);
    }
}

/// 2026-10-22: RFC 8032 §7.1 TEST 1 (empty message): the public key and the signature.
#[test]
fn ed25519_matches_rfc_8032_test_1() {
    use ed25519_dalek::Signer as _;
    let sk = ed25519_dalek::SigningKey::from_bytes(
        &unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        sk.verifying_key().to_bytes().to_vec(),
        unhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    );
    assert_eq!(
        sk.sign(b"").to_bytes().to_vec(),
        unhex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        )
    );
}

#[test]
fn signing_is_deterministic() {
    let s = signer(7);
    let a = s.sign(Purpose::KbpBody, b"same bytes").unwrap();
    let b = s.sign(Purpose::KbpBody, b"same bytes").unwrap();
    assert_eq!(a, b);
    assert_ne!(a, s.sign(Purpose::KbpBody, b"other bytes").unwrap());
}

/// 2026-10-22: metrale-auth signs tokens with seed `[42; 32]` (`token_tests.rs` `pair`) under
/// `authcore/token/v1`. Such a signature must not verify under any blueprint context, and a
/// blueprint body signature must not verify under metrale-auth's context.
#[test]
fn metrale_auth_signatures_and_ours_never_cross() {
    const AUTH: &[u8] = b"authcore/token/v1";
    let key = SigningKey::<MlDsa65>::from_seed(&B32::from([42u8; 32]));
    let vk = {
        use ml_dsa::signature::Keypair;
        key.verifying_key()
    };
    let msg = b"{\"typ\":\"session\"}";
    let auth_sig = key.expanded_key().sign_deterministic(msg, AUTH).unwrap();
    assert!(
        vk.verify_with_context(msg, AUTH, &auth_sig),
        "the control verifies"
    );
    for p in [
        Purpose::KbpBody,
        Purpose::KbpSealed,
        Purpose::KernelArtifact,
        Purpose::KernelIndex,
        Purpose::KeyCert,
        Purpose::Revocations,
    ] {
        assert!(
            !vk.verify_with_context(msg, p.context(), &auth_sig),
            "{p:?}"
        );
    }
    let ours = signer(42).sign(Purpose::KbpBody, msg).unwrap();
    let sig = Signature::<MlDsa65>::try_from(ours.mldsa.as_slice()).unwrap();
    assert!(
        vk.verify_with_context(msg, Purpose::KbpBody.context(), &sig),
        "the control verifies"
    );
    assert!(!vk.verify_with_context(msg, AUTH, &sig));
}

#[test]
fn lv_cat_is_injective_across_boundaries() {
    assert_ne!(lv_cat(&[b"ab", b"c"]), lv_cat(&[b"a", b"bc"]));
    assert_eq!(lv_cat(&[b"ab"]), [0, 0, 0, 2, b'a', b'b']);
}
