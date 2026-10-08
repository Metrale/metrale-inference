// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Known answers for the packed-int layout, written as literal words so they do
//! not depend on a packer, plus controls: a decoder with the wrong field order or the wrong
//! sign convention must fail the same known answers.

use metrale_config::precision_plan::packed_int::PackedIntScheme;

use super::{PackedIntMatrix, check_tensor_pair, decode_code};

const INT4: PackedIntScheme = PackedIntScheme::INT4_G128;
const INT8: PackedIntScheme = PackedIntScheme::INT8_G128;

/// 2026-10-07: A decoder under test: (scheme, row words, k) -> signed code.
type Decode = fn(PackedIntScheme, &[u32], usize) -> i32;

/// 2026-10-07: Known-bad control: fields read most-significant first.
fn msb_first(s: PackedIntScheme, words: &[u32], k: usize) -> i32 {
    let (per, bits) = (s.codes_per_word(), s.bits as usize);
    let u = (words[k / per] >> (bits * (per - 1 - k % per))) & ((1 << bits) - 1);
    u as i32 - s.code_offset()
}

/// 2026-10-07: Known-bad control: fields read as two's complement.
fn twos_complement(s: PackedIntScheme, words: &[u32], k: usize) -> i32 {
    let (per, bits) = (s.codes_per_word(), s.bits as usize);
    let u = ((words[k / per] >> (bits * (k % per))) & ((1 << bits) - 1)) as i32;
    if u >= s.code_offset() {
        u - 2 * s.code_offset()
    } else {
        u
    }
}

/// 2026-10-07: The hand-derived answers. `0x76543210` stores u = 0..=7 at k = 0..=7, so
/// q = -8..=-1; `0xFEDCBA98` stores u = 8..=15, q = 0..=7; `0x8888888F` is q = 7 then
/// seven zeros; INT8 `0x80FF0001` stores bytes 01 00 FF 80 at k = 0..=3, so q = -127, -128,
/// 127, 0.
fn known_answers(decode: Decode) -> Result<(), String> {
    let mut failures = Vec::new();
    let mut check = |what: &str, got: i32, want: i32| {
        if got != want {
            failures.push(format!("{what}: got {got}, want {want}"));
        }
    };
    let int4 = [0x7654_3210u32, 0xFEDC_BA98, 0x8888_888F];
    for k in 0..16 {
        check(&format!("int4 k={k}"), decode(INT4, &int4, k), k as i32 - 8);
    }
    check("int4 k=16", decode(INT4, &int4, 16), 7);
    for k in 17..24 {
        check(&format!("int4 k={k}"), decode(INT4, &int4, k), 0);
    }
    let int8 = [0x80FF_0001u32];
    for (k, want) in [-127, -128, 127, 0].into_iter().enumerate() {
        check(&format!("int8 k={k}"), decode(INT8, &int8, k), want);
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[test]
fn decoder_matches_hand_derived_codes() {
    known_answers(decode_code).expect("layout decoder");
}

#[test]
fn wrong_field_order_fails_the_known_answers() {
    let e = known_answers(msb_first).expect_err("MSB-first must fail");
    assert!(e.contains("int4 k=0: got -1, want -8"), "{e}");
    assert!(e.contains("int8 k=0: got 0, want -127"), "{e}");
}

#[test]
fn twos_complement_fails_the_known_answers() {
    let e = known_answers(twos_complement).expect_err("two's complement must fail");
    assert!(e.contains("int4 k=0: got 0, want -8"), "{e}");
    assert!(e.contains("int8 k=1: got 0, want -128"), "{e}");
}

/// 2026-10-07: A two-row INT4 `[2, 256]` GEMV with hand-computed outputs. Row 0 repeats
/// q = -8..=7 over both groups (group sums -8 * 8 = -64 each) at scales 0.25 and 2.0 with
/// x = 1, so y0 = -64 * 0.25 + -64 * 2.0 = -144. Row 1 is zero except q = 7 at k = 0 and
/// k = 128, scales 0.5 and 0.125, x[0] = 3 and x[128] = -4: y1 = 10.5 - 3.5 = 7.
#[test]
fn int4_gemv_known_answer() {
    let mut words = Vec::new();
    for _ in 0..16 {
        words.extend([0x7654_3210u32, 0xFEDC_BA98]);
    }
    let mut row1 = vec![0x8888_8888u32; 32];
    row1[0] = 0x8888_888F;
    row1[16] = 0x8888_888F;
    words.extend(row1);
    let scales = [0.25, 2.0, 0.5, 0.125];
    let m = PackedIntMatrix::new(INT4, 2, 256, &words, &scales).expect("shape");
    let mut x = vec![1.0f32; 256];
    let y0 = m.gemv(&x)[0];
    x.iter_mut().for_each(|v| *v = 0.0);
    x[0] = 3.0;
    x[128] = -4.0;
    let y1 = m.gemv(&x)[1];
    assert_eq!((y0, y1), (-144.0, 7.0));
    assert_eq!(m.weight(0, 128), -16.0);
    assert_eq!(m.weight(0, 255), 14.0);
}

/// 2026-10-07: INT8 `[1, 128]`: every word `0x80FF0001` (q = -127, -128, 127, 0), scale
/// 0.5, x = 1: 32 * (-128) * 0.5 = -2048.
#[test]
fn int8_gemv_known_answer() {
    let words = vec![0x80FF_0001u32; 32];
    let m = PackedIntMatrix::new(INT8, 1, 128, &words, &[0.5]).expect("shape");
    assert_eq!(m.gemv(&[1.0; 128]), [-2048.0]);
}

#[test]
fn matrix_refuses_mismatched_buffers() {
    let words = vec![0u32; 32];
    assert!(PackedIntMatrix::new(INT4, 2, 128, &words, &[1.0, 1.0]).is_ok());
    assert!(PackedIntMatrix::new(INT4, 2, 128, &words[..31], &[1.0, 1.0]).is_err());
    assert!(PackedIntMatrix::new(INT4, 2, 128, &words, &[1.0]).is_err());
    assert!(PackedIntMatrix::new(INT4, 1, 192, &words[..24], &[1.0, 1.0]).is_err());
    assert!(PackedIntMatrix::new(INT8, 2, 128, &words, &[1.0, 1.0]).is_err());
}

/// 2026-10-07: The tensor shapes of Laguna-XS-2.1-INT4 layer 1 (INT4) and layer 35 (INT8)
/// expert projections, read from the checkpoint's safetensors headers.
#[test]
fn laguna_checkpoint_tensor_shapes_pass_and_transposes_fail() {
    let gate4 = check_tensor_pair(INT4, 512, 2048, ("I32", &[512, 256]), ("BF16", &[512, 16]));
    let down4 = check_tensor_pair(INT4, 2048, 512, ("I32", &[2048, 64]), ("BF16", &[2048, 4]));
    let gate8 = check_tensor_pair(INT8, 512, 2048, ("I32", &[512, 512]), ("BF16", &[512, 16]));
    let down8 = check_tensor_pair(INT8, 2048, 512, ("I32", &[2048, 128]), ("BF16", &[2048, 4]));
    for ok in [gate4, down4, gate8, down8] {
        ok.expect("checkpoint shape");
    }
    let refused = [
        check_tensor_pair(INT4, 2048, 512, ("I32", &[512, 256]), ("BF16", &[512, 16])),
        check_tensor_pair(INT8, 512, 2048, ("I32", &[512, 256]), ("BF16", &[512, 16])),
        check_tensor_pair(INT4, 512, 2048, ("U32", &[512, 256]), ("BF16", &[512, 16])),
        check_tensor_pair(
            INT4,
            512,
            2048,
            ("I32", &[512, 256]),
            ("F8_E4M3", &[512, 16]),
        ),
        check_tensor_pair(INT4, 512, 2048, ("I32", &[512, 256]), ("BF16", &[512, 32])),
        check_tensor_pair(INT4, 512, 2000, ("I32", &[512, 250]), ("BF16", &[512, 16])),
    ];
    for (i, r) in refused.into_iter().enumerate() {
        assert!(r.is_err(), "case {i} accepted");
    }
}

/// 2026-10-07: Round trip through symmetric min-max group quantization (scale =
/// absmax / (2^(bits-1) - 0.5), q = round-half-even(w / scale) clamped), packed per the
/// documented layout: every dequantized value is within half a step of the original.
#[test]
fn symmetric_round_trip_is_within_half_a_step() {
    for scheme in [INT4, INT8] {
        let (n, k) = (3usize, 256usize);
        let qmax = (scheme.code_offset() - 1) as f32;
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919 % 1009) as f32 - 504.0) / 97.0)
            .collect();
        let mut words = vec![0u32; n * k / scheme.codes_per_word()];
        let mut scales = Vec::new();
        for row in 0..n {
            for g in 0..k / 128 {
                let group = &w[row * k + g * 128..row * k + (g + 1) * 128];
                let absmax = group.iter().fold(0f32, |a, v| a.max(v.abs()));
                let scale = absmax / (qmax + 0.5);
                scales.push(scale);
                for (j, v) in group.iter().enumerate() {
                    let q = (v / scale).round_ties_even().clamp(-qmax - 1.0, qmax) as i32;
                    let kk = g * 128 + j;
                    let u = (q + scheme.code_offset()) as u32;
                    let per = scheme.codes_per_word();
                    words[(row * k + kk) / per] |= u << (scheme.bits as usize * (kk % per));
                }
            }
        }
        let m = PackedIntMatrix::new(scheme, n, k, &words, &scales).expect("shape");
        for row in 0..n {
            for kk in 0..k {
                let scale = scales[row * 2 + kk / 128];
                let err = (m.weight(row, kk) - w[row * k + kk]).abs();
                assert!(
                    err <= scale * 0.5 + 1e-6,
                    "{scheme:?} ({row},{kk}) err {err}"
                );
            }
        }
    }
}
