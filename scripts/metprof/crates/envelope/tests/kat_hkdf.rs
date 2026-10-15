// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: HKDF-SHA256 known answers, run through the crate's own
//! `hkdf_sha256` (the call `derive_file_key` and the wrap KEK make).
//!
//! Source: RFC 5869 Appendix A, test cases A.1, A.2 and A.3 (SHA-256),
//! https://www.rfc-editor.org/rfc/rfc5869#appendix-A.

mod common;

use common::hex;
use metprof_envelope::primitives::{hkdf_sha256, hkdf_sha256_key};

fn range(lo: u8, hi: u8) -> Vec<u8> {
    (lo..=hi).collect()
}

fn derive(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    let mut okm = vec![0u8; len];
    hkdf_sha256(salt, ikm, info, &mut okm).expect("hkdf");
    okm
}

#[test]
fn rfc5869_a1_basic() {
    let okm = derive(
        &hex("000102030405060708090a0b0c"),
        &[0x0b; 22],
        &hex("f0f1f2f3f4f5f6f7f8f9"),
        42,
    );
    assert_eq!(
        okm,
        hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")
    );
}

#[test]
fn rfc5869_a2_long_inputs() {
    let okm = derive(
        &range(0x60, 0xaf),
        &range(0x00, 0x4f),
        &range(0xb0, 0xff),
        82,
    );
    assert_eq!(
        okm,
        hex(
            "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c\
             59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71\
             cc30c58179ec3e87c14c01d5c1f3434f1d87"
        )
    );
}

#[test]
fn rfc5869_a3_empty_salt_and_info() {
    let okm = derive(&[], &[0x0b; 22], &[], 42);
    assert_eq!(
        okm,
        hex("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8")
    );
}

#[test]
fn key_variant_is_the_first_32_bytes_of_the_same_expansion() {
    let key = hkdf_sha256_key(
        &hex("000102030405060708090a0b0c"),
        &[0x0b; 22],
        &hex("f0f1f2f3f4f5f6f7f8f9"),
    )
    .expect("hkdf");
    assert_eq!(
        key.to_vec(),
        hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf")
    );
}

#[test]
fn output_longer_than_255_blocks_is_an_error() {
    let mut okm = vec![0u8; 255 * 32 + 1];
    assert!(hkdf_sha256(b"s", b"k", b"i", &mut okm).is_err());
}
