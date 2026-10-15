// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: seal/open and wrap/unwrap round trips, the full publish →
//! fetch flow, the exact canonical header bytes, and an independent decrypt
//! written straight from DESIGN §7.1 with the RustCrypto crates (so a
//! symmetric mistake in seal and open cannot pass unnoticed).

mod common;

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use common::{BUILD, CREATED_AT, ENCODING, FixedClock, KID, PATH, StreamEntropy, request};
use hkdf::Hkdf;
use metprof_envelope::primitives::sha256;
use metprof_envelope::{
    Dek, Expected, KeyRingSecret, OsEntropy, SystemClock, open, parse, seal, unwrap_dek, wrap_dek,
};
use sha2::Sha256;

const EXPECTED: Expected<'static> = Expected {
    kid: KID,
    path: PATH,
};

fn plaintext(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| u8::try_from(i % 251).expect("< 251"))
        .collect()
}

#[test]
fn seal_then_open_returns_the_plaintext_for_many_sizes() {
    let mut entropy = StreamEntropy::new(b"roundtrip");
    let dek = Dek::generate(&mut entropy).expect("dek");
    for len in [0usize, 1, 15, 16, 17, 4096, 100_003] {
        let pt = plaintext(len);
        let env = seal(
            &dek,
            &request(None),
            &pt,
            &mut entropy,
            &FixedClock(CREATED_AT),
        )
        .expect("seal");
        let opened = open(&dek, &EXPECTED, &env).expect("open");
        assert_eq!(opened.plaintext, pt, "len {len}");
        assert_eq!(
            opened.header.plaintext.len,
            u64::try_from(len).expect("len")
        );
        assert_eq!(opened.header.plaintext.sha256, sha256(&pt));
        assert_eq!(opened.header.created_at, CREATED_AT);
    }
}

#[test]
fn production_ports_round_trip() {
    let mut entropy = OsEntropy;
    let dek = Dek::generate(&mut entropy).expect("dek");
    let env = seal(&dek, &request(None), b"hello", &mut entropy, &SystemClock).expect("seal");
    let opened = open(&dek, &EXPECTED, &env).expect("open");
    assert_eq!(opened.plaintext, b"hello");
    assert!(
        opened.header.created_at > CREATED_AT,
        "system clock is after the fixture time"
    );
}

#[test]
fn full_publish_and_fetch_flow_with_wrapped_dek() {
    let mut entropy = StreamEntropy::new(b"flow");
    let keyring = KeyRingSecret::generate(&mut entropy).expect("keyring");
    let pinned = keyring.public();

    let dek = Dek::generate(&mut entropy).expect("dek");
    let wrapped = wrap_dek(&dek, KID, &pinned, &mut entropy).expect("wrap");
    let pt = plaintext(20_000);
    let env = seal(
        &dek,
        &request(Some(wrapped)),
        &pt,
        &mut entropy,
        &FixedClock(CREATED_AT),
    )
    .expect("seal");

    // The key server reads kid + wrapped_dek without any key, unwraps, and the reader opens.
    let parsed = parse(&env).expect("parse");
    let w = parsed
        .header
        .wrapped_dek
        .as_ref()
        .expect("wrapped_dek present");
    let restored = KeyRingSecret::from_bytes(&keyring.to_bytes());
    let dek2 = unwrap_dek(w, &parsed.header.kid, &restored).expect("unwrap");
    assert_eq!(dek2.as_bytes(), dek.as_bytes());
    assert_eq!(open(&dek2, &EXPECTED, &env).expect("open").plaintext, pt);
}

#[test]
fn keyring_public_serialization_round_trips_and_fingerprint_is_its_sha256() {
    let keyring = KeyRingSecret::generate(&mut StreamEntropy::new(b"pub")).expect("keyring");
    let public = keyring.public();
    let bytes = public.to_bytes();
    assert_eq!(bytes.len(), 32 + 1184);
    let back = metprof_envelope::KeyRingPublic::from_bytes(&bytes).expect("parse public");
    assert_eq!(back, public);
    assert_eq!(public.fingerprint(), sha256(&bytes));
    assert!(format!("{public:?}").contains("sha256:"));
    assert_eq!(format!("{keyring:?}"), "KeyRingSecret(<redacted>)");
    assert_eq!(format!("{:?}", Dek::from_bytes([1; 32])), "Dek(<redacted>)");
}

#[test]
fn same_port_inputs_give_identical_bytes() {
    let run = || {
        let mut entropy = StreamEntropy::new(b"determinism");
        let keyring = KeyRingSecret::generate(&mut entropy).expect("keyring");
        let dek = Dek::generate(&mut entropy).expect("dek");
        let wrapped = wrap_dek(&dek, KID, &keyring.public(), &mut entropy).expect("wrap");
        seal(
            &dek,
            &request(Some(wrapped)),
            b"x",
            &mut entropy,
            &FixedClock(CREATED_AT),
        )
        .expect("seal")
    };
    assert_eq!(
        run(),
        run(),
        "all randomness comes through the Entropy port"
    );
}

#[test]
fn canonical_header_bytes_are_exactly_the_specified_json() {
    let dek = Dek::from_bytes([0x11; 32]);
    let mut entropy = common::ConstantEntropy(0xab);
    let env = seal(
        &dek,
        &request(None),
        b"abc",
        &mut entropy,
        &FixedClock(CREATED_AT),
    )
    .expect("seal");
    let parsed = parse(&env).expect("parse");
    let expected = format!(
        concat!(
            r#"{{"alg":"A256GCM","created_at":{},"info":"kkb/v1/{}/{}","kdf":"HKDF-SHA256","kid":"{}","#,
            r#""nonce":"q6urq6urq6urq6ur","plaintext":{{"encoding":"{}","len":3,"#,
            r#""sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}},"#,
            r#""publisher_build":"{}","v":1}}"#
        ),
        CREATED_AT, KID, PATH, KID, ENCODING, BUILD
    );
    assert_eq!(
        std::str::from_utf8(parsed.header_bytes).expect("utf8"),
        expected
    );
    assert_eq!(&env[..4], b"KKB1");
    let hl = u32::from_le_bytes(env[4..8].try_into().expect("4 bytes"));
    assert_eq!(usize::try_from(hl).expect("usize"), expected.len());
    assert_eq!(env.len(), 8 + expected.len() + 3 + 16);
}

#[test]
fn independent_decrypt_from_the_spec_agrees() {
    let mut entropy = StreamEntropy::new(b"spec");
    let dek = Dek::generate(&mut entropy).expect("dek");
    let pt = plaintext(777);
    let env = seal(
        &dek,
        &request(None),
        &pt,
        &mut entropy,
        &FixedClock(CREATED_AT),
    )
    .expect("seal");

    let hl = usize::try_from(u32::from_le_bytes(env[4..8].try_into().expect("4"))).expect("usize");
    let header_bytes = &env[8..8 + hl];
    let header: serde_json::Value = serde_json::from_slice(header_bytes).expect("json");
    let nonce = base64_decode(header["nonce"].as_str().expect("nonce"));
    let info = format!("kkb/v1/{KID}/{PATH}");
    assert_eq!(header["info"], info.as_str());

    let mut k_file = [0u8; 32];
    Hkdf::<Sha256>::new(Some(KID.as_bytes()), dek.as_bytes())
        .expand(info.as_bytes(), &mut k_file)
        .expect("hkdf");
    let out = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&k_file))
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &env[8 + hl..],
                aad: header_bytes,
            },
        )
        .expect("spec decrypt");
    assert_eq!(out, pt);
}

fn base64_decode(text: &str) -> Vec<u8> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .expect("base64")
}
