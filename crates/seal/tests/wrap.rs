// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The recipient key and the hybrid CEK wrap: round trip; every way a wrap can be
//! wrong surfaces as the one opaque `SealedUnwrap` (foreign key, X25519-only and ML-KEM-only
//! half keys, each field altered, another label); a low-order X25519 point is refused on both
//! sides; public keys parse strictly; secrets never print; the run's randomness is one draw.
//! Seeds are fixed test values.

mod common;

use common::CountingEntropy;
use metrale_error::ErrorCode;
use metrale_seal::primitives::MLKEM768_EK_LEN;
use metrale_seal::{
    Cek, HybridPublic, HybridSecret, PUBLIC_LEN, SealRandom, WrapLabel, WrapRandom, unwrap_key,
    wrap_key,
};
use zeroize::Zeroizing;

const BLUEPRINT: [u8; 32] = [0xb1; 32];

fn rnd(b: u8) -> WrapRandom {
    WrapRandom {
        esk: Zeroizing::new([b; 32]),
        m: Zeroizing::new([b.wrapping_add(1); 32]),
        nonce: [b.wrapping_add(2); 12],
    }
}

fn label_for(sk: &HybridSecret) -> WrapLabel {
    WrapLabel::kbp_cek(&BLUEPRINT, sk.public().fingerprint())
}

#[test]
fn a_wrapped_key_unwraps_with_the_recipient_secret() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let cek = Cek::from_bytes([0x5a; 32]);
    let label = label_for(&sk);
    let w = wrap_key(&label, &cek, &sk.public(), &rnd(1)).unwrap();
    let back = unwrap_key(&label, &w, &sk).unwrap();
    assert!(back.same_as(&cek));
    assert_eq!(back.as_bytes(), &[0x5a; 32]);
}

fn assert_unwrap_refused(
    label: &WrapLabel,
    w: &metrale_seal::WrappedCek,
    sk: &HybridSecret,
    what: &str,
) {
    let err = unwrap_key(label, w, sk).unwrap_err();
    assert_eq!(err.code(), ErrorCode::SealedUnwrap, "{what}: {err}");
}

#[test]
fn a_foreign_key_and_each_half_key_are_refused_opaquely() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let label = label_for(&sk);
    let w = wrap_key(&label, &Cek::from_bytes([1; 32]), &sk.public(), &rnd(1)).unwrap();
    let foreign = HybridSecret::from_seed(&[8; 32]);
    assert_unwrap_refused(&label, &w, &foreign, "foreign key");

    let mine = sk.to_bytes();
    let other = foreign.to_bytes();
    let mut x_only = *mine;
    x_only[32..].copy_from_slice(&other[32..]);
    assert_unwrap_refused(
        &label,
        &w,
        &HybridSecret::from_bytes(&x_only),
        "X25519 half only",
    );
    let mut m_only = *mine;
    m_only[..32].copy_from_slice(&other[..32]);
    assert_unwrap_refused(
        &label,
        &w,
        &HybridSecret::from_bytes(&m_only),
        "ML-KEM half only",
    );
}

#[test]
fn every_wrap_field_altered_is_refused() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let label = label_for(&sk);
    let w = wrap_key(&label, &Cek::from_bytes([1; 32]), &sk.public(), &rnd(1)).unwrap();
    let mut cases = Vec::new();
    let mut a = w.clone();
    a.epk[5] ^= 1;
    cases.push(("epk", a));
    let mut a = w.clone();
    a.ct_mlkem[600] ^= 1;
    cases.push(("ct_mlkem", a));
    let mut a = w.clone();
    a.nonce[0] ^= 1;
    cases.push(("nonce", a));
    let mut a = w.clone();
    a.ct[3] ^= 1;
    cases.push(("ct (key)", a));
    let mut a = w.clone();
    a.ct[40] ^= 1;
    cases.push(("ct (tag)", a));
    for (what, altered) in cases {
        assert_unwrap_refused(&label, &altered, &sk, what);
    }
}

#[test]
fn a_wrap_for_another_blueprint_or_fingerprint_is_refused() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let fp = sk.public().fingerprint();
    let w = wrap_key(
        &label_for(&sk),
        &Cek::from_bytes([1; 32]),
        &sk.public(),
        &rnd(1),
    )
    .unwrap();
    assert_unwrap_refused(
        &WrapLabel::kbp_cek(&[0xb2; 32], fp),
        &w,
        &sk,
        "other blueprint",
    );
    let other_fp = HybridSecret::from_seed(&[9; 32]).public().fingerprint();
    assert_unwrap_refused(
        &WrapLabel::kbp_cek(&BLUEPRINT, other_fp),
        &w,
        &sk,
        "other fingerprint",
    );
}

fn public_with_x25519(sk: &HybridSecret, x: [u8; 32]) -> HybridPublic {
    let mut bytes = sk.public().to_bytes();
    bytes[..32].copy_from_slice(&x);
    HybridPublic::from_bytes(&bytes).unwrap()
}

/// 2026-10-22: Sabotage target (DESIGN-encrypted-kernels.md §8 E1): with the contributory check
/// in `x25519_shared` forced true, the `wrap_key(..).unwrap_err()` below panics on an `Ok`.
#[test]
fn a_low_order_x25519_point_is_refused_on_both_sides() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let label = label_for(&sk);
    let mut one = [0u8; 32];
    one[0] = 1;
    for x in [[0u8; 32], one] {
        let to = public_with_x25519(&sk, x);
        let err = wrap_key(&label, &Cek::from_bytes([1; 32]), &to, &rnd(1)).unwrap_err();
        assert_eq!(err.code(), ErrorCode::BlueprintKey, "{err}");
    }
    let mut w = wrap_key(&label, &Cek::from_bytes([1; 32]), &sk.public(), &rnd(1)).unwrap();
    w.epk = [0; 32];
    assert_unwrap_refused(&label, &w, &sk, "low-order epk");
}

#[test]
fn public_keys_parse_strictly() {
    let pk = HybridSecret::from_seed(&[7; 32]).public();
    let bytes = pk.to_bytes();
    assert_eq!(bytes.len(), PUBLIC_LEN);
    assert_eq!(HybridPublic::from_bytes(&bytes).unwrap(), pk);
    for bad in [
        &bytes[..PUBLIC_LEN - 1],
        &[bytes.as_slice(), &[0]].concat()[..],
    ] {
        assert_eq!(
            HybridPublic::from_bytes(bad).unwrap_err().code(),
            ErrorCode::BlueprintKey
        );
    }
    // 2026-10-22: Every 12-bit ek coefficient 0xFFF ≥ q = 3329: fails the FIPS 203 §7.2 check.
    let mut over = bytes.clone();
    over[32..32 + MLKEM768_EK_LEN - 32].fill(0xff);
    assert_eq!(
        HybridPublic::from_bytes(&over).unwrap_err().code(),
        ErrorCode::BlueprintKey
    );
}

#[test]
fn seeds_expand_deterministically_and_distinctly() {
    let a = HybridSecret::from_seed(&[7; 32]);
    assert_eq!(*a.to_bytes(), *HybridSecret::from_seed(&[7; 32]).to_bytes());
    assert_ne!(*a.to_bytes(), *HybridSecret::from_seed(&[8; 32]).to_bytes());
    assert_ne!(
        a.public().fingerprint(),
        HybridSecret::from_seed(&[8; 32]).public().fingerprint()
    );
    let restored = HybridSecret::from_bytes(&a.to_bytes());
    assert_eq!(restored.public(), a.public());
}

#[test]
fn secrets_never_print() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let cek = Cek::from_bytes([0xab; 32]);
    let printed = format!("{sk:?} {cek:?} {:?}", sk.public());
    assert!(printed.contains("<redacted>"));
    for secret in [
        metrale_seal::hex(&*sk.to_bytes()),
        metrale_seal::hex(cek.as_bytes()),
    ] {
        assert!(!printed.contains(&secret[..16]), "{printed}");
    }
    assert!(printed.contains(&sk.public().fingerprint().hex()));
}

#[test]
fn a_run_draws_its_randomness_in_one_fill() {
    let e = CountingEntropy::new(0);
    let r = SealRandom::draw(&e, 3).unwrap();
    assert_eq!(e.calls.get(), 1);
    assert_eq!(e.drawn.get(), SealRandom::len(3));
    assert_eq!(r.wraps.len(), 3);
    assert_eq!(r.cek.as_bytes()[0], 0);
    assert_eq!(r.wraps[0].esk[0], 32);
    assert_eq!(r.wraps[1].esk[0], 32 + 76);
    assert_ne!(*r.wraps[0].m, *r.wraps[1].m);
    assert!(SealRandom::from_bytes(&[0; 32], 1).is_err());
}

#[test]
fn equal_and_unequal_keys_compare_as_such() {
    assert!(Cek::from_bytes([3; 32]).same_as(&Cek::from_bytes([3; 32])));
    let mut b = [3; 32];
    b[31] = 4;
    assert!(!Cek::from_bytes([3; 32]).same_as(&Cek::from_bytes(b)));
}

#[test]
fn a_wrap_round_trips_through_postcard_and_refuses_other_lengths() {
    let sk = HybridSecret::from_seed(&[7; 32]);
    let w = wrap_key(
        &label_for(&sk),
        &Cek::from_bytes([1; 32]),
        &sk.public(),
        &rnd(1),
    )
    .unwrap();
    let bytes = postcard::to_allocvec(&w).unwrap();
    assert_eq!(
        postcard::from_bytes::<metrale_seal::WrappedCek>(&bytes).unwrap(),
        w
    );
    assert_eq!(
        bytes.len(),
        32 + 2 + 1088 + 12 + 1 + 48,
        "fixed arrays, two varint lengths"
    );
    // 2026-10-22: A 1087-byte ML-KEM ciphertext: rewrite the varint length (1088 = 0xC0 0x08).
    let mut short = bytes.clone();
    assert_eq!(&short[32..34], &[0xc0, 0x08]);
    short[32] = 0xbf;
    short.remove(34);
    assert!(postcard::from_bytes::<metrale_seal::WrappedCek>(&short).is_err());
}
