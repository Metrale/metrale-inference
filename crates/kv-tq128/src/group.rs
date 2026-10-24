// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Encode and decode of one 128-group and of whole head rows.
//!
//! Encode: `norm = sqrt(sum x^2)`; scale by `1/norm` (0 when `norm <= 1e-10`); rotate;
//! quantize (Viterbi or nearest level); store `norm / |reconstruction|` as fp16, so the decoded
//! group keeps the input's norm (arXiv:2504.19874 — TurboQuant rotation and norm). Decode:
//! `codebook[index] * fp16(norm) * decode_scale(side)` in the rotated domain; dequant also
//! un-rotates and rounds to bf16. Format after spiritbuun/buun-llama-cpp (MIT; credit
//! spiritbuun), encode-side norm alpha fixed at 1.0 (upstream's default).
//!
//! Owner: kv-tq128.
//! Invariants: every sum of 128 terms runs in the kernels' lane-then-butterfly order
//! ([`group_sum_squares`]), so encode output is byte-identical to the GPU's.

use crate::Tq128Error;
use crate::codebooks::TURBO4_MID;
use crate::codec::{GROUP, KvSide, Tq128Codec};
use crate::numeric::{
    f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, group_sum_squares, sqrt_f32,
};
use crate::pack::{nibble_at, pack_nibbles, pack_trellis, trellis_state_at};
use crate::rotation::{rotate, unrotate};
use crate::trellis::viterbi;

const TINY: f32 = 1e-10;

fn as_group(x: &[f32]) -> Result<&[f32; GROUP], Tq128Error> {
    <&[f32; GROUP]>::try_from(x).map_err(|_| Tq128Error::GroupLength { found: x.len() })
}

fn check_block(codec: Tq128Codec, block: &[u8]) -> Result<(), Tq128Error> {
    let expected = codec.block_bytes();
    if block.len() == expected {
        Ok(())
    } else {
        Err(Tq128Error::BlockLength {
            expected,
            found: block.len(),
        })
    }
}

/// 2026-10-18: The scalar codec's index: how many decision midpoints lie at or below `v`
/// (a value exactly on a midpoint takes the upper level, as upstream's search does).
fn turbo4_index(v: f32) -> u8 {
    let mut i = 0_u8;
    for &m in &TURBO4_MID {
        if v >= m {
            i += 1;
        }
    }
    i
}

/// 2026-10-18: Encode one 128-group into `out` (`codec.block_bytes()` bytes).
pub fn encode_group(
    codec: Tq128Codec,
    side: KvSide,
    x: &[f32],
    out: &mut [u8],
) -> Result<(), Tq128Error> {
    let x = as_group(x)?;
    check_block(codec, out)?;
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(Tq128Error::NonFinite { index });
    }
    let book = codec.codebook(side);
    let norm = sqrt_f32(group_sum_squares(x));
    let inv = if norm > TINY { 1.0 / norm } else { 0.0 };
    let mut v = *x;
    for e in &mut v {
        *e *= inv;
    }
    rotate(&mut v);
    out.fill(0);
    let mut recon = [0.0_f32; GROUP];
    let payload = &mut out[2..2 + codec.payload_bytes()];
    match codec.trellis() {
        Some(t) => {
            let path = viterbi(t, book, &v)?;
            for (r, &s) in recon.iter_mut().zip(&path.states) {
                *r = book[usize::from(s)];
            }
            pack_trellis(t, &path.states, payload)?;
        }
        None => {
            let mut idx = [0_u8; GROUP];
            for ((i, r), &e) in idx.iter_mut().zip(recon.iter_mut()).zip(&v) {
                *i = turbo4_index(e);
                *r = book[usize::from(*i)];
            }
            pack_nibbles(&idx, payload)?;
        }
    }
    let recon_norm = sqrt_f32(group_sum_squares(&recon));
    let corrected = if recon_norm > TINY {
        norm / recon_norm
    } else {
        norm
    };
    out[..2].copy_from_slice(&f32_to_f16_bits(corrected).to_le_bytes());
    Ok(())
}

/// 2026-10-18: The 128 codebook indices a block selects (trellis states or nibbles).
pub fn group_indices(codec: Tq128Codec, block: &[u8]) -> Result<[u16; GROUP], Tq128Error> {
    check_block(codec, block)?;
    let payload = &block[2..];
    let mut idx = [0_u16; GROUP];
    for (t, i) in idx.iter_mut().enumerate() {
        *i = match codec.trellis() {
            Some(tr) => trellis_state_at(tr, payload, t),
            None => u16::from(nibble_at(payload, t)),
        };
    }
    Ok(idx)
}

/// 2026-10-18: Decode a block in the rotated domain: `codebook[index] * scale`, with
/// `scale = fp16(norm) * decode_scale(side)` formed first.
pub fn decode_group_rotated(
    codec: Tq128Codec,
    side: KvSide,
    block: &[u8],
) -> Result<[f32; GROUP], Tq128Error> {
    let idx = group_indices(codec, block)?;
    let book = codec.codebook(side);
    let scale =
        f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]])) * codec.decode_scale(side);
    let mut y = [0.0_f32; GROUP];
    for (o, &i) in y.iter_mut().zip(&idx) {
        *o = book[usize::from(i)] * scale;
    }
    Ok(y)
}

/// 2026-10-18: Decode a block back to the input domain (f32).
pub fn dequant_group(
    codec: Tq128Codec,
    side: KvSide,
    block: &[u8],
) -> Result<[f32; GROUP], Tq128Error> {
    let mut y = decode_group_rotated(codec, side, block)?;
    unrotate(&mut y);
    Ok(y)
}

/// 2026-10-18: Decode a block back to the input domain, rounded to bf16 bits.
pub fn dequant_group_bf16(
    codec: Tq128Codec,
    side: KvSide,
    block: &[u8],
) -> Result<[u16; GROUP], Tq128Error> {
    let y = dequant_group(codec, side, block)?;
    let mut out = [0_u16; GROUP];
    for (o, v) in out.iter_mut().zip(&y) {
        *o = f32_to_bf16_bits(*v);
    }
    Ok(out)
}

fn groups_of(head_dim: usize) -> Result<usize, Tq128Error> {
    if head_dim == 0 || !head_dim.is_multiple_of(GROUP) {
        return Err(Tq128Error::HeadDim { head_dim });
    }
    Ok(head_dim / GROUP)
}

/// 2026-10-18: Encode one head row (`head_dim` a positive multiple of 128) group by group into
/// `out` (`head_dim / 128` blocks).
pub fn encode_head(
    codec: Tq128Codec,
    side: KvSide,
    row: &[f32],
    out: &mut [u8],
) -> Result<(), Tq128Error> {
    let groups = groups_of(row.len())?;
    let expected = groups * codec.block_bytes();
    if out.len() != expected {
        return Err(Tq128Error::BlockLength {
            expected,
            found: out.len(),
        });
    }
    for (x, o) in row
        .chunks_exact(GROUP)
        .zip(out.chunks_exact_mut(codec.block_bytes()))
    {
        encode_group(codec, side, x, o)?;
    }
    Ok(())
}

/// 2026-10-18: Decode one head row's blocks to bf16 bits (`out.len()` is the head_dim).
pub fn dequant_head_bf16(
    codec: Tq128Codec,
    side: KvSide,
    blocks: &[u8],
    out: &mut [u16],
) -> Result<(), Tq128Error> {
    let groups = groups_of(out.len())?;
    let expected = groups * codec.block_bytes();
    if blocks.len() != expected {
        return Err(Tq128Error::BlockLength {
            expected,
            found: blocks.len(),
        });
    }
    for (b, o) in blocks
        .chunks_exact(codec.block_bytes())
        .zip(out.chunks_exact_mut(GROUP))
    {
        o.copy_from_slice(&dequant_group_bf16(codec, side, b)?);
    }
    Ok(())
}
