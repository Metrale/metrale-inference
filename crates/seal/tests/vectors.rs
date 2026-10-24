// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Our own deterministic vectors (DESIGN-encrypted-kernels.md §8): from fixed seeds
//! and fixed randomness, the recipient fingerprint, the wrapped CEK and one sealed artifact are
//! pinned byte for byte, so a change of any context string, combiner order or field layout
//! fails here. Values were produced by this crate on 2026-10-22 and are checked independently
//! by `wrap.rs` (unwrap) and `artifact.rs` (a decrypt written from the design).

use metrale_seal::primitives::sha256;
use metrale_seal::{
    ArtifactBinding, Cek, HybridSecret, SealRandom, WrapLabel, hex, seal_artifact, unwrap_key,
    wrap_key,
};

const RECIPIENT_SEED: [u8; 32] = [0x11; 32];
const BLUEPRINT: [u8; 32] = [0xb1; 32];

/// 2026-10-22: Run randomness: bytes 0, 1, 2, … (CEK = 00..1f, then esk, m, nonce).
fn run() -> SealRandom {
    let bytes: Vec<u8> = (0..SealRandom::len(1)).map(|i| i as u8).collect();
    SealRandom::from_bytes(&bytes, 1).unwrap()
}

#[test]
fn the_recipient_key_of_a_fixed_seed() {
    let sk = HybridSecret::from_seed(&RECIPIENT_SEED);
    assert_eq!(
        hex(&sk.to_bytes()[..32]),
        "dcac894c8d4b0b983c7bfbcf8f1486f39639048caf92805892d69747a73f3136",
        "X25519 half of HKDF(metrale/kbp-recipient/v1, 0x11*32)"
    );
    let mut okm = [0u8; 96];
    hkdf::Hkdf::<sha2::Sha256>::new(Some(b"metrale/kbp-recipient/v1"), &RECIPIENT_SEED)
        .expand(b"hybrid-x25519-mlkem768", &mut okm)
        .unwrap();
    assert_eq!(*sk.to_bytes(), okm, "independent expansion of the seed");
    assert_eq!(
        sk.public().fingerprint().hex(),
        "94d0f381991e28dbb689dc3452be0d116c38f01d34d659b1b9a3ba7b5b70a6f9"
    );
}

#[test]
fn the_wrap_of_a_fixed_cek_with_fixed_randomness() {
    let sk = HybridSecret::from_seed(&RECIPIENT_SEED);
    let label = WrapLabel::kbp_cek(&BLUEPRINT, sk.public().fingerprint());
    let r = run();
    let w = wrap_key(&label, &r.cek, &sk.public(), &r.wraps[0]).unwrap();
    assert_eq!(
        hex(&w.epk),
        "358072d6365880d1aeea329adf9121383851ed21a28e3b75e965d0d2cd166254"
    );
    assert_eq!(
        hex(&sha256(&w.ct_mlkem)),
        "716f1032a930c272b5f7376d7cdabcabb6dce118c4fd1f7fd5c6e52e0404843b"
    );
    assert_eq!(
        hex(&w.ct),
        "cf5184942ddc03a929cf71add85300707385f91a33cdafa43b87fca4915105792457b088bd3f17274e37a321fd79ec15"
    );
    assert!(unwrap_key(&label, &w, &sk).unwrap().same_as(&r.cek));
}

#[test]
fn a_sealed_artifact_with_a_fixed_cek() {
    let b = ArtifactBinding {
        blueprint: &BLUEPRINT,
        artifact: &[0xa1; 32],
        module: "m",
        arch: "sm_121f",
        codec: 0,
        raw_len: 5,
    };
    let sealed = seal_artifact(&Cek::from_bytes(*run().cek.as_bytes()), &b, b"hello").unwrap();
    assert_eq!(hex(&sealed), "c29c873bc7d22072e204e58d4728dd0d2e912a35ae");
}
