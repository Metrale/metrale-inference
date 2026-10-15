// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the hybrid DEK unwrap refuses foreign keys, the wrong kid and
//! any altered wrapped field, and the wrap refuses degenerate public keys.

mod common;

use common::{KID, StreamEntropy};
use metprof_envelope::{
    Dek, EnvelopeError, KeyRingPublic, KeyRingSecret, WrappedDek, unwrap_dek, wrap_dek,
};

fn setup() -> (KeyRingSecret, Dek, WrappedDek) {
    let mut entropy = StreamEntropy::new(b"wrap");
    let keyring = KeyRingSecret::generate(&mut entropy).expect("keyring");
    let dek = Dek::generate(&mut entropy).expect("dek");
    let wrapped = wrap_dek(&dek, KID, &keyring.public(), &mut entropy).expect("wrap");
    (keyring, dek, wrapped)
}

#[test]
fn unwrap_with_the_right_secret_recovers_the_dek() {
    let (keyring, dek, wrapped) = setup();
    assert_eq!(
        unwrap_dek(&wrapped, KID, &keyring)
            .expect("unwrap")
            .as_bytes(),
        dek.as_bytes()
    );
}

#[test]
fn unwrap_with_a_foreign_secret_fails() {
    let (_, _, wrapped) = setup();
    let foreign = KeyRingSecret::generate(&mut StreamEntropy::new(b"foreign")).expect("keyring");
    assert!(matches!(
        unwrap_dek(&wrapped, KID, &foreign),
        Err(EnvelopeError::Unwrap)
    ));
}

#[test]
fn unwrap_with_half_the_right_secret_fails() {
    let (keyring, _, wrapped) = setup();
    let foreign = KeyRingSecret::generate(&mut StreamEntropy::new(b"foreign")).expect("keyring");
    let (good, bad) = (keyring.to_bytes(), foreign.to_bytes());
    let mut x_only = *good;
    x_only[32..].copy_from_slice(&bad[32..]);
    let mut mlkem_only = *good;
    mlkem_only[..32].copy_from_slice(&bad[..32]);
    for mixed in [x_only, mlkem_only] {
        let key = KeyRingSecret::from_bytes(&mixed);
        assert!(matches!(
            unwrap_dek(&wrapped, KID, &key),
            Err(EnvelopeError::Unwrap)
        ));
    }
}

#[test]
fn unwrap_under_another_kid_fails() {
    let (keyring, _, wrapped) = setup();
    assert!(matches!(
        unwrap_dek(&wrapped, "kid-test-02", &keyring),
        Err(EnvelopeError::Unwrap)
    ));
}

#[test]
fn any_altered_wrapped_field_fails() {
    let (keyring, _, wrapped) = setup();
    let mut cases: Vec<(&str, WrappedDek)> = Vec::new();
    for i in [0usize, 31, 47] {
        let mut w = wrapped.clone();
        w.ct[i] ^= 1;
        cases.push(("ct", w));
    }
    for i in [0usize, 543, 1087] {
        let mut w = wrapped.clone();
        w.ct_mlkem[i] ^= 1;
        cases.push(("ct_mlkem", w));
    }
    for i in [0usize, 15, 30] {
        let mut w = wrapped.clone();
        w.epk_x25519[i] ^= 1;
        cases.push(("epk_x25519", w));
    }
    for i in [0usize, 11] {
        let mut w = wrapped.clone();
        w.nonce[i] ^= 1;
        cases.push(("nonce", w));
    }
    for (field, w) in cases {
        assert!(
            matches!(unwrap_dek(&w, KID, &keyring), Err(EnvelopeError::Unwrap)),
            "{field}"
        );
    }
}

#[test]
fn unsupported_scheme_is_named() {
    let (keyring, _, mut wrapped) = setup();
    wrapped.scheme = "x25519-hkdf-a256gcm".to_owned();
    assert!(matches!(
        unwrap_dek(&wrapped, KID, &keyring),
        Err(EnvelopeError::Unsupported {
            field: "wrapped_dek.scheme",
            ..
        })
    ));
}

#[test]
fn low_order_x25519_public_key_is_refused() {
    let (keyring, dek, _) = setup();
    let mut bytes = keyring.public().to_bytes();
    bytes[..32].fill(0);
    let degenerate = KeyRingPublic::from_bytes(&bytes).expect("ek still valid");
    let err = wrap_dek(&dek, KID, &degenerate, &mut StreamEntropy::new(b"e")).expect_err("refused");
    assert_eq!(err, EnvelopeError::NonContributory);
}

#[test]
fn low_order_ephemeral_key_is_refused_on_unwrap() {
    let (keyring, _, mut wrapped) = setup();
    wrapped.epk_x25519 = [0; 32];
    assert!(matches!(
        unwrap_dek(&wrapped, KID, &keyring),
        Err(EnvelopeError::Unwrap)
    ));
}

#[test]
fn invalid_mlkem_encapsulation_key_is_refused() {
    let (keyring, _, _) = setup();
    let mut bytes = keyring.public().to_bytes();
    // First 12-bit coefficient of t_hat set to 0xFFF >= q = 3329 (FIPS 203 §7.2 modulus check).
    bytes[32] = 0xff;
    bytes[33] |= 0x0f;
    assert_eq!(
        KeyRingPublic::from_bytes(&bytes),
        Err(EnvelopeError::InvalidPublicKey(
            "ML-KEM-768 encapsulation key"
        ))
    );
    assert!(matches!(
        KeyRingPublic::from_bytes(&bytes[..100]),
        Err(EnvelopeError::BadLength { .. })
    ));
}

#[test]
fn wrap_rejects_an_invalid_kid() {
    let (keyring, dek, _) = setup();
    let err = wrap_dek(
        &dek,
        "a/b",
        &keyring.public(),
        &mut StreamEntropy::new(b"e"),
    )
    .expect_err("kid");
    assert_eq!(err, EnvelopeError::InvalidKid);
}

/// Recomputes the KEK straight from the module documentation with the
/// RustCrypto crates, so a combiner that silently drops an input (which the
/// public-key binding alone would mask) cannot pass.
#[test]
fn independent_unwrap_from_the_spec_agrees() {
    use aes_gcm::aead::{Aead, KeyInit, Payload};
    use aes_gcm::{Aes256Gcm, Key, Nonce};
    use hkdf::Hkdf;
    use ml_kem::kem::Decapsulate;
    use ml_kem::ml_kem_768::DecapsulationKey;
    use sha2::Sha256;
    use x25519_dalek::{PublicKey, StaticSecret};

    let (keyring, dek, wrapped) = setup();
    let secret = keyring.to_bytes();
    let sk_x = StaticSecret::from(<[u8; 32]>::try_from(&secret[..32]).expect("32"));
    let seed: [u8; 64] = secret[32..].try_into().expect("64");
    let ss_x = sk_x.diffie_hellman(&PublicKey::from(wrapped.epk_x25519));
    let dk = DecapsulationKey::from_seed(seed.into());
    let ss_m = dk.decapsulate(&wrapped.ct_mlkem.into());

    let mut ikm = Vec::new();
    ikm.extend_from_slice(ss_m.as_slice());
    ikm.extend_from_slice(ss_x.as_bytes());
    ikm.extend_from_slice(&wrapped.ct_mlkem);
    ikm.extend_from_slice(&wrapped.epk_x25519);
    ikm.extend_from_slice(PublicKey::from(&sk_x).as_bytes());
    let label = format!("kkb/v1/wrap/{KID}");
    let mut kek = [0u8; 32];
    Hkdf::<Sha256>::new(Some(b"hybrid-x25519-mlkem768-hkdf-a256gcm"), &ikm)
        .expand(label.as_bytes(), &mut kek)
        .expect("hkdf");
    let out = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&kek))
        .decrypt(
            Nonce::from_slice(&wrapped.nonce),
            Payload {
                msg: &wrapped.ct,
                aad: label.as_bytes(),
            },
        )
        .expect("spec unwrap");
    assert_eq!(out.as_slice(), dek.as_bytes());
}
