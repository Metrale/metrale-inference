// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: Weights in their stored precision behind one `linear`, with the publisher's
//! `inference/model.py` semantics, so the reference runs the real checkpoint without widening
//! it. A quantized weight (fp8 e4m3 with an E8M0 scale per tile, or fp4 e2m1 with an E8M0
//! scale per 32) is decoded one row at a time, exactly (a code times a power of two is
//! representable in f32), and only the rows a call touches are read.
//!
//! The fixture weights stay `[f32]` bf16 values; [`Weight`] is implemented for them and for
//! [`Mat`], so every projection goes through [`linear_grouped`] whichever the storage.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants:
//! - Decoding is exact; the only roundings are the input QDQ a weight declares
//!   ([`InputQuant`]) and the output precision a call asks for ([`OutPrec`]).
//! - The input QDQ is declared by whoever builds the weight (the publisher quantizes the
//!   activation for an fp8/fp4 weight, but not for `wo_a`, which its converter widens to
//!   bf16); it is never inferred from the storage.
//! - The reference performs no I/O: quantized storage is borrowed bytes.

use metrale_error::{Error, ErrorCode, error};
use rayon::prelude::*;

use super::attn::{FP8_BLOCK, act_quant_inplace};
use super::to_bf16_rne;

/// 2026-10-19: The scale group along a row for e2m1 weights, and the column tile of e4m3
/// weights (the publisher's `fp8_block_size` and its fp4 block).
pub const MX_BLOCK: usize = 32;

/// 2026-10-19: Output rows decoded per parallel work item.
const ROW_TILE: usize = 64;

/// 2026-10-19: What happens to the activation before the dot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputQuant {
    /// The activation enters as it is (bf16 and f32 weights).
    Exact,
    /// `act_quant(x, 32, ue8m0)` quantize-dequantize first: the publisher's `linear()` on an
    /// fp8 or fp4 weight.
    Fp8Block,
}

/// 2026-10-19: The precision of a projection's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutPrec {
    /// f32 accumulate, rounded to bf16 (the publisher's default dtype).
    Bf16,
    /// f32 accumulate kept (the router gate, the head, the ratio > 1 compressor).
    F32,
}

/// 2026-10-19: A row-major `[rows, cols]` weight the reference can project with.
pub trait Weight: Sync {
    /// The activation treatment this weight declares.
    fn input_quant(&self) -> InputQuant;
    /// Row `r` of the matrix, `cols` wide, as exact f32 values into `out`.
    fn decode_row(&self, r: usize, cols: usize, out: &mut [f32]);
}

impl Weight for [f32] {
    fn input_quant(&self) -> InputQuant {
        InputQuant::Exact
    }

    fn decode_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        out.copy_from_slice(&self[r * cols..(r + 1) * cols]);
    }
}

impl Weight for Vec<f32> {
    fn input_quant(&self) -> InputQuant {
        InputQuant::Exact
    }

    fn decode_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        self.as_slice().decode_row(r, cols, out);
    }
}

/// 2026-10-19: fp8 e4m3fn code -> value (sign, 4-bit exponent with bias 7, 3-bit mantissa;
/// exponent 0 is subnormal, `S.1111.111` is NaN), built at compile time.
const E4M3: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let exp = (b >> 3) & 0xF;
        let man = (b & 7) as u32;
        let mag = if exp == 0xF && man == 7 {
            f32::NAN
        } else if exp == 0 {
            man as f32 * (1.0 / 512.0)
        } else {
            f32::from_bits((((exp as u32) + 120) << 23) | (man << 20))
        };
        t[b] = if b & 0x80 != 0 { -mag } else { mag };
        b += 1;
    }
    t
};

/// 2026-10-19: fp4 e2m1 code -> value, the publisher's `FP4_TABLE`.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-10-19: The value of an E8M0 scale byte, `2^(b - 127)`; 255 is the NaN code, which no
/// checkpoint scale may hold.
pub fn e8m0(b: u8) -> f32 {
    assert_ne!(b, 0xFF, "E8M0 NaN scale");
    if b == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits(u32::from(b) << 23)
    }
}

/// 2026-10-19: The fp8 e4m3fn value of `code`.
pub fn e4m3(code: u8) -> f32 {
    E4M3[usize::from(code)]
}

/// 2026-10-19: Borrowed storage of a quantized or bf16 checkpoint tensor.
#[derive(Clone, Copy, Debug)]
pub enum Store<'a> {
    /// Little-endian bf16, two bytes per value.
    Bf16(&'a [u8]),
    /// One fp8 e4m3fn code per value; one E8M0 scale per `block_rows x 32` tile, row-major
    /// over tiles.
    E4m3 {
        w: &'a [u8],
        scale: &'a [u8],
        block_rows: usize,
    },
    /// Two fp4 e2m1 codes per byte, the low nibble first; one E8M0 scale per 32 values.
    E2m1 { w: &'a [u8], scale: &'a [u8] },
}

/// 2026-10-19: A checkpoint weight `[rows, cols]` in its stored precision.
#[derive(Clone, Copy, Debug)]
pub struct Mat<'a> {
    store: Store<'a>,
    rows: usize,
    cols: usize,
    input: InputQuant,
}

fn refuse(what: &'static str, why: String) -> Error {
    error!(ErrorCode::InvalidArgument, what, why)
}

impl<'a> Mat<'a> {
    /// 2026-10-19: Validate `store` against `[rows, cols]` and bind the input treatment.
    pub fn new(
        store: Store<'a>,
        rows: usize,
        cols: usize,
        input: InputQuant,
    ) -> Result<Self, Error> {
        let n = rows * cols;
        let check = |got: usize, want: usize, part: &str| {
            if got == want {
                Ok(())
            } else {
                Err(refuse(
                    "weight storage",
                    format!("{part} holds {got} bytes, [{rows} x {cols}] needs {want}"),
                ))
            }
        };
        match store {
            Store::Bf16(w) => check(w.len(), n * 2, "bf16 data")?,
            Store::E4m3 {
                w,
                scale,
                block_rows,
            } => {
                if block_rows == 0 {
                    return Err(refuse("weight storage", "e4m3 block_rows is 0".into()));
                }
                check(w.len(), n, "e4m3 data")?;
                let tiles = rows.div_ceil(block_rows) * cols.div_ceil(MX_BLOCK);
                check(scale.len(), tiles, "e4m3 scale")?;
            }
            Store::E2m1 { w, scale } => {
                if !cols.is_multiple_of(MX_BLOCK) {
                    return Err(refuse(
                        "weight storage",
                        format!("e2m1 cols {cols} is not a multiple of {MX_BLOCK}"),
                    ));
                }
                check(w.len(), n / 2, "e2m1 data")?;
                check(scale.len(), n / MX_BLOCK, "e2m1 scale")?;
            }
        }
        Ok(Mat {
            store,
            rows,
            cols,
            input,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl Weight for Mat<'_> {
    fn input_quant(&self) -> InputQuant {
        self.input
    }

    fn decode_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        assert!(
            cols == self.cols && r < self.rows,
            "row {r} x {cols} of [{} x {}]",
            self.rows,
            self.cols
        );
        match self.store {
            Store::Bf16(w) => {
                for (o, b) in out
                    .iter_mut()
                    .zip(w[r * cols * 2..(r + 1) * cols * 2].chunks_exact(2))
                {
                    *o = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
                }
            }
            Store::E4m3 {
                w,
                scale,
                block_rows,
            } => {
                let tiles = cols.div_ceil(MX_BLOCK);
                let srow = &scale[(r / block_rows) * tiles..(r / block_rows + 1) * tiles];
                let codes = &w[r * cols..(r + 1) * cols];
                for (j, (o, c)) in out
                    .chunks_mut(MX_BLOCK)
                    .zip(codes.chunks(MX_BLOCK))
                    .enumerate()
                {
                    let s = e8m0(srow[j]);
                    for (v, &code) in o.iter_mut().zip(c) {
                        *v = E4M3[usize::from(code)] * s;
                    }
                }
            }
            Store::E2m1 { w, scale } => {
                let srow = &scale[r * cols / MX_BLOCK..(r + 1) * cols / MX_BLOCK];
                let bytes = &w[r * cols / 2..(r + 1) * cols / 2];
                for (j, (o, bb)) in out
                    .chunks_mut(MX_BLOCK)
                    .zip(bytes.chunks(MX_BLOCK / 2))
                    .enumerate()
                {
                    let s = e8m0(srow[j]);
                    for (pair, &b) in o.chunks_mut(2).zip(bb) {
                        pair[0] = E2M1[usize::from(b & 0xF)] * s;
                        pair[1] = E2M1[usize::from(b >> 4)] * s;
                    }
                }
            }
        }
    }
}

/// 2026-10-19: `y = x · w^T`: `x` `[n, in_dim]`, `w` `[out_dim, in_dim]`, result
/// `[n, out_dim]`.
pub fn linear<W: Weight + ?Sized>(
    w: &W,
    x: &[f32],
    n: usize,
    in_dim: usize,
    out_dim: usize,
    prec: OutPrec,
) -> Vec<f32> {
    linear_grouped(w, x, n, 1, in_dim, out_dim, prec)
}

/// 2026-10-19: The block-diagonal projection (the publisher's `einsum("bsgd,grd->bsgr")`):
/// `x` `[n, groups, in_g]`, `w` `[groups * out_g, in_g]` where group `g` owns rows
/// `g*out_g..(g+1)*out_g`; result `[n, groups, out_g]`. `groups = 1` is a plain linear.
/// Each weight row is decoded once and dotted with every token, in sequential f32 order.
pub fn linear_grouped<W: Weight + ?Sized>(
    w: &W,
    x: &[f32],
    n: usize,
    groups: usize,
    in_g: usize,
    out_g: usize,
    prec: OutPrec,
) -> Vec<f32> {
    assert_eq!(x.len(), n * groups * in_g, "activation shape");
    let quantized;
    let x = match w.input_quant() {
        InputQuant::Exact => x,
        InputQuant::Fp8Block => {
            assert!(
                in_g.is_multiple_of(FP8_BLOCK),
                "QDQ blocks must not straddle rows"
            );
            let mut q = x.to_vec();
            act_quant_inplace(&mut q);
            quantized = q;
            &quantized
        }
    };
    let out_dim = groups * out_g;
    let tiles: Vec<Vec<f32>> = (0..out_dim.div_ceil(ROW_TILE))
        .into_par_iter()
        .map(|t| {
            let (r0, r1) = (t * ROW_TILE, ((t + 1) * ROW_TILE).min(out_dim));
            let mut row = vec![0f32; in_g];
            let mut acc = Vec::with_capacity((r1 - r0) * n);
            for r in r0..r1 {
                w.decode_row(r, in_g, &mut row);
                let g = r / out_g;
                for i in 0..n {
                    let xi = &x[(i * groups + g) * in_g..(i * groups + g + 1) * in_g];
                    acc.push(xi.iter().zip(&row).map(|(a, b)| a * b).sum::<f32>());
                }
            }
            acc
        })
        .collect();
    let mut y = vec![0f32; n * out_dim];
    for (t, acc) in tiles.iter().enumerate() {
        for (k, &v) in acc.iter().enumerate() {
            let (r, i) = (t * ROW_TILE + k / n, k % n);
            y[i * out_dim + r] = match prec {
                OutPrec::Bf16 => to_bf16_rne(v),
                OutPrec::F32 => v,
            };
        }
    }
    y
}

/// 2026-10-19: Gather rows of a `[_, cols]` table by id (the Engram embedding lookup).
/// Output `[ids.len(), cols]`.
pub fn gather_rows<W: Weight + ?Sized>(w: &W, cols: usize, ids: &[i64]) -> Vec<f32> {
    let mut out = vec![0f32; ids.len() * cols];
    for (o, &id) in out.chunks_mut(cols).zip(ids) {
        let r = usize::try_from(id).expect("non-negative row id");
        w.decode_row(r, cols, o);
    }
    out
}

#[cfg(test)]
#[path = "mat_tests.rs"]
mod tests;
