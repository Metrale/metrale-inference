// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Encodings round-trip exactly, refuse values they cannot hold, and pack E2M1 with
//! the even index in the low nibble (the layout every NVFP4 kernel here reads).

use super::*;

#[test]
fn encodings_round_trip_and_refuse_unrepresentable_values() {
    let vals = [0.0, -0.5, 1.5, 6.0, -6.0, 3.0, 0.5];
    let t = Tensor::encode(Enc::E2m1x2, vec![7], &vals).unwrap();
    assert_eq!(t.bytes.len(), 4);
    assert_eq!(t.values(), vals.to_vec());
    assert_eq!(t.bytes[0] & 0xf, 0x0, "index 0 in the low nibble");
    assert_eq!(t.bytes[0] >> 4, 0x9, "index 1 (-0.5) in the high nibble");
    assert!(Tensor::encode(Enc::E2m1x2, vec![1], &[2.5]).is_err());
    assert!(Tensor::encode(Enc::Bf16, vec![1], &[1.0 + 1.0 / 1024.0]).is_err());
    assert!(Tensor::encode(Enc::Ue4m3, vec![1], &[-1.0]).is_err());
    assert!(
        Tensor::encode(Enc::F32, vec![2], &[1.0]).is_err(),
        "length mismatch"
    );
    let big = crate::elem::BF16.round(3.0e38).unwrap();
    let b = Tensor::encode(Enc::Bf16, vec![2, 2], &[1.0, -2.0, 0.15625, big]).unwrap();
    assert_eq!(b.values(), vec![1.0, -2.0, 0.15625, big]);
    let u = Tensor::encode(Enc::Ue8m0, vec![2], &[0.25, 8.0]).unwrap();
    assert_eq!(u.bytes, vec![125, 130]);
    assert!(Tensor::encode(Enc::Ue8m0, vec![1], &[3.0]).is_err());
}
