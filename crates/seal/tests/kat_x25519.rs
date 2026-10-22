// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: X25519 known answers through the crate's own `x25519_shared` / `x25519_public`
//! (the calls the wrap makes). Source: RFC 7748 §5.2 (the two single-step vectors and the
//! 1 and 1000 iteration results) and §6.1 (the Diffie-Hellman example),
//! https://www.rfc-editor.org/rfc/rfc7748.

mod common;

use common::hex_array;
use metrale_seal::primitives::{x25519_public, x25519_shared};

fn x25519(k: &[u8; 32], u: &[u8; 32]) -> [u8; 32] {
    *x25519_shared(k, u).expect("contributory")
}

#[test]
fn rfc7748_5_2_single_steps() {
    for (k, u, out) in [
        (
            "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
            "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
            "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552",
        ),
        (
            "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
            "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493",
            "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957",
        ),
    ] {
        assert_eq!(x25519(&hex_array(k), &hex_array(u)), hex_array::<32>(out));
    }
}

#[test]
fn rfc7748_5_2_iterations_1_and_1000() {
    let mut k = [0u8; 32];
    k[0] = 9;
    let mut u = k;
    for i in 1..=1000 {
        let next = x25519(&k, &u);
        u = k;
        k = next;
        if i == 1 {
            assert_eq!(
                k,
                hex_array::<32>("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")
            );
        }
    }
    assert_eq!(
        k,
        hex_array::<32>("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51")
    );
}

#[test]
fn rfc7748_6_1_diffie_hellman() {
    let a: [u8; 32] = hex_array("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
    let b: [u8; 32] = hex_array("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
    let pa = x25519_public(&a);
    let pb = x25519_public(&b);
    assert_eq!(
        pa,
        hex_array::<32>("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
    );
    assert_eq!(
        pb,
        hex_array::<32>("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
    );
    let shared =
        hex_array::<32>("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
    assert_eq!(x25519(&a, &pb), shared);
    assert_eq!(x25519(&b, &pa), shared);
}

/// 2026-10-22: Low-order points give the all-zero secret, which the exchange refuses.
#[test]
fn low_order_points_are_not_contributory() {
    let k = [0x42u8; 32];
    let mut one = [0u8; 32];
    one[0] = 1;
    for u in [[0u8; 32], one] {
        assert!(x25519_shared(&k, &u).is_none());
    }
}
