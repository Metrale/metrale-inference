// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: One sealed artifact: round trip; distinct artifacts, blueprints and CEKs never
//! share a key or a nonce; every AAD component and every ciphertext byte is bound; a ciphertext
//! over its cap is refused before the AEAD; and an independent decrypt written straight from
//! DESIGN-encrypted-kernels.md §2.4 with the RustCrypto crates (not this crate's functions)
//! opens what the crate seals.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use hkdf::Hkdf;
use metrale_error::ErrorCode;
use metrale_seal::{ArtifactBinding, ArtifactKey, Cek, open_artifact, seal_artifact};
use sha2::Sha256;

const BP: [u8; 32] = [0xb1; 32];
const A1: [u8; 32] = [0xa1; 32];
const A2: [u8; 32] = [0xa2; 32];
const CAP: u64 = 1 << 20;

fn binding<'a>(bp: &'a [u8; 32], art: &'a [u8; 32]) -> ArtifactBinding<'a> {
    ArtifactBinding {
        blueprint: bp,
        artifact: art,
        module: "t0__glm_moe",
        arch: "sm_121f",
        codec: 2,
        raw_len: 4096,
    }
}

#[test]
fn a_sealed_artifact_opens_with_its_binding() {
    let cek = Cek::from_bytes([9; 32]);
    let b = binding(&BP, &A1);
    let sealed = seal_artifact(&cek, &b, b"compressed kernel").unwrap();
    assert_eq!(sealed.len(), b"compressed kernel".len() + 16);
    assert_eq!(
        open_artifact(&cek, &b, &sealed, CAP).unwrap().as_slice(),
        b"compressed kernel"
    );
    assert_eq!(
        sealed,
        seal_artifact(&cek, &b, b"compressed kernel").unwrap(),
        "deterministic"
    );
}

/// 2026-10-22: Sabotage target (DESIGN-encrypted-kernels.md §8 E1): with `ArtifactId` dropped
/// from the HKDF info in `ArtifactKey::derive`, the first `assert_ne!` (key across artifacts)
/// goes red: two artifacts of one blueprint would share a GCM key and nonce.
#[test]
fn artifacts_blueprints_and_ceks_never_share_a_key_or_nonce() {
    let cek = Cek::from_bytes([9; 32]);
    let k1 = ArtifactKey::derive(&cek, &BP, &A1);
    let k2 = ArtifactKey::derive(&cek, &BP, &A2);
    assert_ne!(k1.key(), k2.key(), "key across artifacts");
    assert_ne!(k1.nonce(), k2.nonce(), "nonce across artifacts");
    let k3 = ArtifactKey::derive(&cek, &[0xb2; 32], &A1);
    assert_ne!(k1.key(), k3.key(), "key across blueprints");
    assert_ne!(k1.nonce(), k3.nonce(), "nonce across blueprints");
    let k4 = ArtifactKey::derive(&Cek::from_bytes([8; 32]), &BP, &A1);
    assert_ne!(k1.key(), k4.key(), "key across CEKs");
    assert_ne!(k1.nonce(), k4.nonce(), "nonce across CEKs");
}

#[test]
fn every_aad_component_is_bound() {
    let cek = Cek::from_bytes([9; 32]);
    let b = binding(&BP, &A1);
    let sealed = seal_artifact(&cek, &b, b"payload").unwrap();
    let bp2 = [0xb2; 32];
    let cases: Vec<(&str, ArtifactBinding<'_>)> = vec![
        (
            "blueprint",
            ArtifactBinding {
                blueprint: &bp2,
                ..b
            },
        ),
        ("artifact", ArtifactBinding { artifact: &A2, ..b }),
        (
            "module",
            ArtifactBinding {
                module: "t0__glm_moe2",
                ..b
            },
        ),
        (
            "arch",
            ArtifactBinding {
                arch: "sm_120f",
                ..b
            },
        ),
        ("codec", ArtifactBinding { codec: 1, ..b }),
        ("raw_len", ArtifactBinding { raw_len: 4097, ..b }),
    ];
    for (what, other) in cases {
        let err = open_artifact(&cek, &other, &sealed, CAP).unwrap_err();
        assert_eq!(err.code(), ErrorCode::SealedOpen, "{what}");
    }
    // 2026-10-22: The AAD alone (not just the key derivation) binds the fields the HKDF info
    // omits: the same key and nonce with a different module name still fails.
    let mut aad_only = b;
    aad_only.module = "other";
    let k = ArtifactKey::derive(&cek, &BP, &A1);
    assert!(
        metrale_seal::primitives::aes256gcm_open(k.key(), k.nonce(), &aad_only.aad(), &sealed)
            .is_none()
    );
}

#[test]
fn every_flipped_ciphertext_byte_and_another_cek_fail() {
    let cek = Cek::from_bytes([9; 32]);
    let b = binding(&BP, &A1);
    let sealed = seal_artifact(&cek, &b, b"a short kernel").unwrap();
    for i in 0..sealed.len() {
        let mut bad = sealed.clone();
        bad[i] ^= 0x80;
        let err = open_artifact(&cek, &b, &bad, CAP).unwrap_err();
        assert_eq!(err.code(), ErrorCode::SealedOpen, "byte {i}");
        assert!(err.to_string().contains("t0__glm_moe"));
    }
    assert_eq!(
        open_artifact(&Cek::from_bytes([8; 32]), &b, &sealed, CAP)
            .unwrap_err()
            .code(),
        ErrorCode::SealedOpen
    );
    assert_eq!(
        open_artifact(&cek, &b, &sealed[..15], CAP)
            .unwrap_err()
            .code(),
        ErrorCode::SealedOpen
    );
}

/// 2026-10-22: Over the cap is refused as a cap even for garbage bytes: the AEAD never runs.
#[test]
fn a_ciphertext_over_its_cap_is_refused_before_the_aead() {
    let cek = Cek::from_bytes([9; 32]);
    let b = binding(&BP, &A1);
    let sealed = seal_artifact(&cek, &b, &[0u8; 100]).unwrap();
    assert!(open_artifact(&cek, &b, &sealed, 100).is_ok());
    assert_eq!(
        open_artifact(&cek, &b, &sealed, 99).unwrap_err().code(),
        ErrorCode::BlueprintTooLarge
    );
    assert_eq!(
        open_artifact(&cek, &b, &[0u8; 200], 99).unwrap_err().code(),
        ErrorCode::BlueprintTooLarge
    );
}

fn lv(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(&(p.len() as u32).to_be_bytes());
        out.extend_from_slice(p);
    }
    out
}

#[test]
fn an_independent_decrypt_from_the_design_opens_it() {
    let cek = [9u8; 32];
    let b = binding(&BP, &A1);
    let sealed = seal_artifact(&Cek::from_bytes(cek), &b, b"independent").unwrap();
    let mut okm = [0u8; 44];
    Hkdf::<Sha256>::new(Some(b"metrale/kbp-seal/v1"), &cek)
        .expand(&lv(&[&BP, &A1]), &mut okm)
        .unwrap();
    let key: [u8; 32] = okm[..32].try_into().unwrap();
    let nonce: [u8; 12] = okm[32..].try_into().unwrap();
    let aad = lv(&[
        b"metrale/kbp-seal/v1",
        &BP,
        &A1,
        b"t0__glm_moe",
        b"sm_121f",
        &[2],
        &4096u64.to_le_bytes(),
    ]);
    let pt = Aes256Gcm::new(&key.into())
        .decrypt(
            &Nonce::from(nonce),
            Payload {
                msg: &sealed,
                aad: &aad,
            },
        )
        .unwrap();
    assert_eq!(pt, b"independent");
}
