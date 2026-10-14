// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: How far apart two BF16 outputs are, in units in the last place, for the GPU class
//! tests that compare one kernel's bytes with another's.
//!
//! Owner: model-layers ops (test support).
//! Invariants: a non-finite element on either side reads as `i32::MAX` ulps, never as a match.

/// 2026-10-14: A BF16 as an integer that orders like the value (sign-magnitude to two's
/// complement), so the distance of two finite BF16s in ulps is a difference.
fn bf16_ordinal(b: [u8; 2]) -> i32 {
    let v = u16::from_le_bytes(b) as i32;
    if v & 0x8000 != 0 { -(v & 0x7FFF) } else { v }
}

/// 2026-10-14: Elements of `got` that differ from `want` (little-endian BF16 bytes), and the
/// largest distance in BF16 ulps (`i32::MAX` when either side is not finite).
pub(crate) fn ulp_gap(want: &[u8], got: &[u8]) -> (usize, i32) {
    let finite = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]) & 0x7F80 != 0x7F80;
    want.chunks_exact(2)
        .zip(got.chunks_exact(2))
        .filter(|(w, g)| w != g)
        .fold((0, 0), |(n, max), (w, g)| {
            let gap = if finite(w) && finite(g) {
                (bf16_ordinal([w[0], w[1]]) - bf16_ordinal([g[0], g[1]])).abs()
            } else {
                i32::MAX
            };
            (n + 1, max.max(gap))
        })
}

/// 2026-10-14: The ordinal is what makes the gap a count of representable values: across zero
/// (-0 and +0 are 0 apart, the smallest subnormals 2 apart), and up the exponent boundary.
#[test]
fn the_gap_counts_representable_bf16_values() {
    let le = |v: &[u16]| v.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>();
    let one = 0x3F80;
    assert_eq!(ulp_gap(&le(&[one]), &le(&[one])), (0, 0));
    assert_eq!(ulp_gap(&le(&[one, one]), &le(&[one, one + 3])), (1, 3));
    assert_eq!(ulp_gap(&le(&[0x0000]), &le(&[0x8000])), (1, 0));
    assert_eq!(ulp_gap(&le(&[0x0001]), &le(&[0x8001])), (1, 2));
    assert_eq!(ulp_gap(&le(&[0x3F7F]), &le(&[0x3F80])), (1, 1));
    assert_eq!(ulp_gap(&le(&[one]), &le(&[0x7FC0])), (1, i32::MAX));
    assert_eq!(ulp_gap(&le(&[0x7F80]), &le(&[0xFF80])), (1, i32::MAX));
}
