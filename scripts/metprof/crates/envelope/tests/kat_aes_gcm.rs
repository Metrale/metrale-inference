// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: AES-256-GCM known answers, run through the crate's own
//! `aes256gcm_seal` / `aes256gcm_open` (the calls `seal` and `open` make).
//!
//! Source: NIST CAVP "GCM Test Vectors (SP 800-38D)", gcmtestvectors.zip
//! (https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/mac/gcmtestvectors.zip,
//! sha256 f9fc479e134cde2980b3bb7cddbcb567b2cd96fd753835243ed067699f26a023),
//! file gcmEncryptExtIV256.rsp ("CAVS 14.0", sha256
//! 4448b4998c3f58cd9f5542ddf7c573189a721ff3d3ac24cd5177bb4910a787f8),
//! [Keylen = 256] [IVlen = 96] [Taglen = 128], Count = 0 of each section named below.

mod common;

use common::{hex, hex_array};
use metprof_envelope::EnvelopeError;
use metprof_envelope::primitives::{aes256gcm_open, aes256gcm_seal};

struct Vector {
    section: &'static str,
    key: &'static str,
    iv: &'static str,
    pt: &'static str,
    aad: &'static str,
    ct: &'static str,
    tag: &'static str,
}

const VECTORS: [Vector; 3] = [
    Vector {
        section: "[PTlen = 0] [AADlen = 0]",
        key: "b52c505a37d78eda5dd34f20c22540ea1b58963cf8e5bf8ffa85f9f2492505b4",
        iv: "516c33929df5a3284ff463d7",
        pt: "",
        aad: "",
        ct: "",
        tag: "bdc1ac884d332457a1d2664f168c76f0",
    },
    Vector {
        section: "[PTlen = 128] [AADlen = 160]",
        key: "83688deb4af8007f9b713b47cfa6c73e35ea7a3aa4ecdb414dded03bf7a0fd3a",
        iv: "0b459724904e010a46901cf3",
        pt: "33d893a2114ce06fc15d55e454cf90c3",
        aad: "794a14ccd178c8ebfd1379dc704c5e208f9d8424",
        ct: "cc66bee423e3fcd4c0865715e9586696",
        tag: "0fb291bd3dba94a1dfd8b286cfb97ac5",
    },
    Vector {
        section: "[PTlen = 408] [AADlen = 720]",
        key: "148579a3cbca86d5520d66c0ec71ca5f7e41ba78e56dc6eebd566fed547fe691",
        iv: "b08a5ea1927499c6ecbfd4e0",
        pt: "9d0b15fdf1bd595f91f8b3abc0f7dec927dfd4799935a1795d9ce00c9b879434420fe42c275a7cd7b39d638fb81ca52b49dc41",
        aad: "e4f963f015ffbb99ee3349bbaf7e8e8e6c2a71c230a48f9d59860a29091d2747e01a5ca572347e247d25f56ba7ae8e05cde2be3c97931292c02370208ecd097ef692687fecf2f419d3200162a6480a57dad408a0dfeb492e2c5d",
        ct: "2097e372950a5e9383c675e89eea1c314f999159f5611344b298cda45e62843716f215f82ee663919c64002a5c198d7878fd3f",
        tag: "adbecdb0d5c2224d804d2886ff9a5760",
    },
];

fn expected_body(v: &Vector) -> Vec<u8> {
    let mut body = hex(v.ct);
    body.extend(hex(v.tag));
    body
}

#[test]
fn seal_matches_cavp_ciphertext_and_tag() {
    for v in &VECTORS {
        let body = aes256gcm_seal(&hex_array(v.key), &hex_array(v.iv), &hex(v.aad), &hex(v.pt))
            .expect("seal");
        assert_eq!(body, expected_body(v), "{}", v.section);
    }
}

#[test]
fn open_recovers_cavp_plaintext() {
    for v in &VECTORS {
        let pt = aes256gcm_open(
            &hex_array(v.key),
            &hex_array(v.iv),
            &hex(v.aad),
            &expected_body(v),
        )
        .expect("open");
        assert_eq!(pt, hex(v.pt), "{}", v.section);
    }
}

#[test]
fn open_rejects_cavp_vector_with_any_tag_or_aad_bit_flipped() {
    for v in &VECTORS {
        let key = hex_array(v.key);
        let iv = hex_array(v.iv);
        let body = expected_body(v);
        let aad = hex(v.aad);
        for bit in 0..body.len() * 8 {
            let mut bad = body.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            assert_eq!(
                aes256gcm_open(&key, &iv, &aad, &bad),
                Err(EnvelopeError::Decrypt),
                "{} body bit {bit}",
                v.section
            );
        }
        for bit in 0..aad.len() * 8 {
            let mut bad = aad.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            assert_eq!(
                aes256gcm_open(&key, &iv, &bad, &body),
                Err(EnvelopeError::Decrypt),
                "{} aad bit {bit}",
                v.section
            );
        }
    }
}
