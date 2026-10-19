// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::attn::act_quant_inplace;
use super::super::to_bf16_rne;
use super::*;

/// 2026-10-19: Deterministic values in roughly [-2, 2), not bf16- or fp8-representable in
/// general, so a skipped rounding shows up.
fn vals(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| ((to_bf16_rne(x).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// 2026-10-19: The plain definition: every output is a sequential f32 dot of one input row
/// with one weight row.
fn naive(w: &[f32], x: &[f32], n: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * out_dim];
    for i in 0..n {
        for o in 0..out_dim {
            y[i * out_dim + o] = x[i * in_dim..(i + 1) * in_dim]
                .iter()
                .zip(&w[o * in_dim..(o + 1) * in_dim])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    y
}

fn decode_all<W: Weight + ?Sized>(w: &W, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for (r, o) in out.chunks_mut(cols).enumerate() {
        w.decode_row(r, cols, o);
    }
    out
}

#[test]
fn e8m0_is_two_to_the_byte_minus_127_including_the_subnormal_zero_code() {
    assert_eq!(e8m0(127), 1.0);
    assert_eq!(e8m0(128), 2.0);
    assert_eq!(e8m0(120), 2f32.powi(-7));
    assert_eq!(e8m0(254), 2f32.powi(127));
    assert_eq!(e8m0(1), 2f32.powi(-126));
    assert_eq!(e8m0(0), 2f32.powi(-127));
}

#[test]
#[should_panic(expected = "E8M0 NaN scale")]
fn e8m0_refuses_the_nan_code() {
    e8m0(0xFF);
}

#[test]
fn bf16_storage_decodes_little_endian_high_halves() {
    // 2026-10-19: 1.0 = 0x3F80, -2.5 = 0xC020, the smallest normal 2^-126 = 0x0080.
    let bytes = [0x80, 0x3F, 0x20, 0xC0, 0x80, 0x00];
    let m = Mat::new(Store::Bf16(&bytes), 1, 3, InputQuant::Exact).unwrap();
    assert_eq!(decode_all(&m, 1, 3), vec![1.0, -2.5, 2f32.powi(-126)]);
}

/// 2026-10-19: A `[40, 40]` e4m3 weight with 32x32 tiles has 2x2 scale tiles, the last ones
/// partial. Every element is code 1.0 times its tile's scale, so a transposed or
/// off-by-one tile index puts the wrong power of two somewhere.
#[test]
fn e4m3_tiles_index_scales_by_row_block_then_column_block() {
    let (rows, cols) = (40, 40);
    let codes = vec![0x38u8; rows * cols];
    let scale = [127u8, 128, 129, 130];
    let m = Mat::new(
        Store::E4m3 {
            w: &codes,
            scale: &scale,
            block_rows: 32,
        },
        rows,
        cols,
        InputQuant::Fp8Block,
    )
    .unwrap();
    let got = decode_all(&m, rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let want = [1.0, 2.0, 4.0, 8.0][(r / 32) * 2 + c / 32];
            assert_eq!(got[r * cols + c], want, "({r}, {c})");
        }
    }
}

/// 2026-10-19: The Engram table layout: one scale per row per 32 values (`block_rows = 1`),
/// and codes decoded through the e4m3 table, sign and subnormals included.
#[test]
fn e4m3_row_blocks_of_one_and_codes_follow_the_e4m3_table() {
    let (rows, cols) = (3, 64);
    let codes: Vec<u8> = (0..rows * cols).map(|i| (i * 37 % 256) as u8).collect();
    let codes: Vec<u8> = codes
        .into_iter()
        .map(|c| if c & 0x7F == 0x7F { 0x01 } else { c })
        .collect();
    let scale = [120u8, 121, 122, 123, 124, 125];
    let m = Mat::new(
        Store::E4m3 {
            w: &codes,
            scale: &scale,
            block_rows: 1,
        },
        rows,
        cols,
        InputQuant::Fp8Block,
    )
    .unwrap();
    let got = decode_all(&m, rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let want = e4m3(codes[r * cols + c]) * e8m0(scale[r * 2 + c / 32]);
            assert_eq!(got[r * cols + c].to_bits(), want.to_bits(), "({r}, {c})");
        }
    }
    assert_eq!(e4m3(0x01), 2f32.powi(-9), "smallest subnormal");
    assert_eq!(e4m3(0x7E), 448.0);
    assert_eq!(e4m3(0xB8), -1.0);
    assert!(e4m3(0x7F).is_nan() && e4m3(0xFF).is_nan());
}

/// 2026-10-19: Byte `0x71` holds code 1 (0.5) in its low nibble and code 7 (6.0) in its high
/// nibble; the low one comes first. The second 32-value block takes the second scale.
#[test]
fn e2m1_decodes_the_low_nibble_first_with_one_scale_per_32() {
    let (rows, cols) = (2, 64);
    let mut w = vec![0u8; rows * cols / 2];
    w[0] = 0x71;
    w[16] = 0x9A; // 2026-10-19: codes 0xA (-1.0) then 0x9 (-0.5), block 1 of row 0
    w[32] = 0x0F; // 2026-10-19: code 0xF (-6.0) then 0x0, row 1
    let scale = [128u8, 126, 127, 127];
    let m = Mat::new(
        Store::E2m1 {
            w: &w,
            scale: &scale,
        },
        rows,
        cols,
        InputQuant::Fp8Block,
    )
    .unwrap();
    let got = decode_all(&m, rows, cols);
    assert_eq!(&got[..2], &[1.0, 12.0]);
    assert_eq!(&got[32..34], &[-0.5, -0.25]);
    assert_eq!(&got[64..66], &[-6.0, 0.0]);
    assert_eq!(got.iter().filter(|v| **v != 0.0).count(), 5);
}

#[test]
fn storage_that_does_not_match_the_shape_is_refused() {
    let b = vec![0u8; 64 * 2];
    let refused = |r: Result<Mat<'_>, Error>| {
        let e = r.expect_err("must refuse").to_string();
        assert!(e.contains("invalid argument weight storage"), "{e}");
        e
    };
    refused(Mat::new(Store::Bf16(&b[..127]), 2, 32, InputQuant::Exact));
    let e = refused(Mat::new(
        Store::E4m3 {
            w: &b[..64],
            scale: &b[..2],
            block_rows: 32,
        },
        2,
        32,
        InputQuant::Fp8Block,
    ));
    assert!(e.contains("e4m3 scale"), "{e}");
    assert!(
        refused(Mat::new(
            Store::E4m3 {
                w: &b[..64],
                scale: &b[..2],
                block_rows: 0,
            },
            2,
            32,
            InputQuant::Fp8Block,
        ))
        .contains("block_rows is 0")
    );
    assert!(
        refused(Mat::new(
            Store::E2m1 {
                w: &b[..24],
                scale: &b[..1],
            },
            2,
            24,
            InputQuant::Fp8Block,
        ))
        .contains("not a multiple of 32")
    );
    assert!(
        refused(Mat::new(
            Store::E2m1 {
                w: &b[..32],
                scale: &b[..1],
            },
            2,
            32,
            InputQuant::Fp8Block,
        ))
        .contains("e2m1 scale")
    );
    // 2026-10-19: The control: the same shapes with the right lengths are accepted.
    Mat::new(Store::Bf16(&b), 2, 32, InputQuant::Exact).unwrap();
    Mat::new(
        Store::E4m3 {
            w: &b[..64],
            scale: &b[..1],
            block_rows: 32,
        },
        2,
        32,
        InputQuant::Fp8Block,
    )
    .unwrap();
    Mat::new(
        Store::E2m1 {
            w: &b[..32],
            scale: &b[..2],
        },
        2,
        32,
        InputQuant::Fp8Block,
    )
    .unwrap();
}

/// 2026-10-19: A weight that declares `Fp8Block` projects exactly what an exact weight does
/// on the act-quantized input; the control (no QDQ) differs, so the declaration is live.
#[test]
fn fp8_block_input_is_the_act_quant_of_the_activation() {
    let (n, k, o) = (3, 64, 5);
    let wv: Vec<f32> = vals(o * k, 1).into_iter().map(to_bf16_rne).collect();
    let bytes = bf16_bytes(&wv);
    let x = vals(n * k, 2);
    let m = Mat::new(Store::Bf16(&bytes), o, k, InputQuant::Fp8Block).unwrap();
    let got = linear(&m, &x, n, k, o, OutPrec::F32);
    let mut xq = x.clone();
    act_quant_inplace(&mut xq);
    assert_eq!(got, naive(&wv, &xq, n, k, o));
    assert_ne!(
        got,
        naive(&wv, &x, n, k, o),
        "control: QDQ must change the result"
    );
    let exact = Mat::new(Store::Bf16(&bytes), o, k, InputQuant::Exact).unwrap();
    assert_eq!(
        linear(&exact, &x, n, k, o, OutPrec::F32),
        naive(&wv, &x, n, k, o)
    );
}

/// 2026-10-19: More output rows than one parallel tile and several tokens: the scatter back
/// to `[n, out]` must put every dot in its place, bit for bit.
#[test]
fn linear_matches_the_definition_across_row_tiles() {
    let (n, k, o) = (3, 40, 2 * ROW_TILE + 7);
    let w = vals(o * k, 3);
    let x = vals(n * k, 4);
    let want = naive(&w, &x, n, k, o);
    assert_eq!(linear(w.as_slice(), &x, n, k, o, OutPrec::F32), want);
    let b = linear(w.as_slice(), &x, n, k, o, OutPrec::Bf16);
    assert_eq!(b, want.iter().map(|&v| to_bf16_rne(v)).collect::<Vec<_>>());
    assert!(
        want.iter().zip(&b).any(|(a, c)| a != c),
        "control: F32 keeps bits bf16 drops"
    );
}

/// 2026-10-19: The grouped projection is one plain linear per group on that group's slice of
/// the input and that group's rows of the weight.
#[test]
fn grouped_linear_is_one_linear_per_group() {
    let (n, groups, in_g, out_g) = (2, 3, 32, 70);
    let w = vals(groups * out_g * in_g, 5);
    let x = vals(n * groups * in_g, 6);
    let got = linear_grouped(w.as_slice(), &x, n, groups, in_g, out_g, OutPrec::Bf16);
    for g in 0..groups {
        let wg = &w[g * out_g * in_g..(g + 1) * out_g * in_g];
        let xg: Vec<f32> = (0..n)
            .flat_map(|i| x[(i * groups + g) * in_g..(i * groups + g + 1) * in_g].to_vec())
            .collect();
        let want = linear(wg, &xg, n, in_g, out_g, OutPrec::Bf16);
        for i in 0..n {
            let row = &got[(i * groups + g) * out_g..(i * groups + g + 1) * out_g];
            assert_eq!(
                row,
                &want[i * out_g..(i + 1) * out_g],
                "token {i} group {g}"
            );
        }
    }
}

/// 2026-10-19: `gather_rows` returns the decoded rows in id order, repeats included.
#[test]
fn gather_rows_decodes_the_requested_rows_in_order() {
    let (rows, cols) = (4, 32);
    let codes: Vec<u8> = (0..rows * cols).map(|i| (i % 120) as u8).collect();
    let scale = [127u8, 128, 129, 130];
    let m = Mat::new(
        Store::E4m3 {
            w: &codes,
            scale: &scale,
            block_rows: 1,
        },
        rows,
        cols,
        InputQuant::Fp8Block,
    )
    .unwrap();
    let all = decode_all(&m, rows, cols);
    let got = gather_rows(&m, cols, &[3, 0, 3]);
    assert_eq!(&got[..cols], &all[3 * cols..]);
    assert_eq!(&got[cols..2 * cols], &all[..cols]);
    assert_eq!(&got[2 * cols..], &all[3 * cols..]);
}
