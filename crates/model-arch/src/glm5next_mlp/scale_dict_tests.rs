// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `scale_dict`: exact round trips at the serving shapes (whole chunks and a
//! K % 128 == 64 tail), the byte layout the kernel decodes, and the refusals.

use super::*;

fn codes(rows: usize, groups: usize, base: u8, span: u8, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..rows * groups)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            base + ((s >> 33) % u64::from(span)) as u8
        })
        .collect()
}

#[test]
fn every_code_round_trips_at_whole_and_half_chunk_widths() {
    // 2026-10-21: gate/up per rank [704, 256 groups]; down per rank [4096, 44 groups].
    for (rows, groups) in [(704, 256), (64, 44), (3, 4)] {
        let s = codes(rows, groups, 90, 32, rows as u64);
        let c = compact(&s, rows, groups).expect("span 32 fits");
        let cells = rows * groups.div_ceil(8);
        assert_eq!(c.bytes.len(), cells * 4 + cells.next_multiple_of(4));
        assert_eq!(expand(&c, rows, groups), s);
        assert_eq!(c.base, *s.iter().min().unwrap());
    }
}

/// 2026-10-21: The word and hi byte of one chunk, decoded by hand as the kernel does: byte b of
/// the word is (S0[b] low, S1[b] high) nibbles; hi bits b / 4 + b their fifth bits.
#[test]
fn one_chunk_has_the_kernel_layout() {
    let base = 90u8;
    let offs: [u8; 8] = [0, 31, 16, 1, 17, 2, 30, 15];
    let s: Vec<u8> = offs.iter().map(|o| base + o).collect();
    let c = compact(&s, 1, 8).unwrap();
    let w = u32::from_le_bytes(c.bytes[..4].try_into().unwrap());
    let h = c.bytes[4];
    let word = |order: [usize; 4], shift: u32, hshift: u32| -> u32 {
        (0..4)
            .map(|b| {
                let o = u32::from(offs[order[b]]);
                let lo = (w >> (8 * b as u32 + shift)) & 0xF;
                let fifth = u32::from((h >> (hshift + b as u32)) & 1);
                assert_eq!(lo | (fifth << 4), o, "group {}", order[b]);
                (u32::from(base) + o) << (8 * b)
            })
            .sum()
    };
    // 2026-10-21: The scale words the plain path's byte_perm builds from the natural bytes.
    let x = u32::from_le_bytes(s[..4].try_into().unwrap());
    let y = u32::from_le_bytes(s[4..].try_into().unwrap());
    let perm = |sel: [u32; 4]| -> u32 {
        let b = |i: u32| {
            if i < 4 {
                (x >> (8 * i)) & 0xFF
            } else {
                (y >> (8 * (i - 4))) & 0xFF
            }
        };
        (0..4).map(|k| b(sel[k as usize]) << (8 * k)).sum()
    };
    assert_eq!(word(S0, 0, 0), perm([0, 4, 2, 6]));
    assert_eq!(word(S1, 4, 4), perm([1, 5, 3, 7]));
}

#[test]
fn a_span_over_32_codes_or_a_malformed_tensor_keeps_its_plain_scales() {
    let wide = codes(16, 256, 80, 33, 1);
    assert!(wide.iter().max().unwrap() - wide.iter().min().unwrap() >= 32);
    assert_eq!(compact(&wide, 16, 256), None);
    assert_eq!(compact(&codes(2, 6, 80, 8, 1), 2, 6), None, "groups % 4");
    assert_eq!(compact(&[], 0, 8), None);
    assert_eq!(compact(&[1, 2, 3], 1, 4), None, "length");
    assert_eq!(compact(&[0x7F, 0x7F, 0x7F, 0x7F], 1, 4), None, "NaN code");
}
