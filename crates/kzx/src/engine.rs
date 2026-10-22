// SPDX-License-Identifier: MIT OR Apache-2.0

//! The block coder: one byte loop written once over [`BitCoder`], so compression and
//! decompression run the same model steps in the same order. Each model byte is either flagged
//! (a long match predicted it: one binary decision) or coded as eight bits through the
//! predictor. For the PTX profile the bytes the model sees are the transformed skeleton
//! ([`crate::ptx`]); the numbers are coded inline after each marker.
//!
//! Loop bounds: the generic loop runs exactly `raw_len` iterations; the PTX decode loop emits at
//! least one output byte per iteration and stops at `raw_len`, so both are bounded by the
//! declared length before the first bit is read.

use crate::coder::{BitCoder, Decoder, Encoder};
use crate::format::Profile;
use crate::predictor::Predictor;
use crate::ptx::{ESCAPE, Event, MARKER, NumberModel, next_event, push_decimal};
use alloc::vec::Vec;

/// Code one model byte (`Some` when encoding) and append it to `hist`. Returns the byte.
#[inline]
pub fn code_byte(
    pred: &mut Predictor,
    io: &mut impl BitCoder,
    byte: Option<u8>,
    hist: &mut Vec<u8>,
) -> u8 {
    if let Some(e) = pred.lzp_candidate() {
        let hit = io.code(u32::from(byte == Some(e)), pred.lzp_p()) == 1;
        pred.lzp_update(hit);
        if hit {
            hist.push(e);
            pred.byte_done(hist);
            return e;
        }
        pred.set_excluded(e);
    }
    let want = u32::from(byte.unwrap_or(0));
    let mut c = 1u32;
    for bitpos in 0..8u32 {
        let bit = (want >> (7 - bitpos)) & 1;
        let p = pred.p();
        let got = io.code(bit, p);
        pred.update(got);
        c = (c << 1) | got;
    }
    let b = u8::try_from(c & 0xff).unwrap_or(0);
    hist.push(b);
    pred.byte_done(hist);
    b
}

/// Compress one block with `profile` and `mem_log2`; the caller frames it.
pub fn compress_block(raw: &[u8], profile: Profile, mem_log2: u8) -> Vec<u8> {
    let mut pred = Predictor::new(profile, mem_log2);
    let mut enc = Encoder::with_capacity(raw.len() / 4 + 16);
    let mut hist: Vec<u8> = Vec::with_capacity(raw.len() + 16);
    match profile {
        Profile::Stored | Profile::Generic | Profile::Elf => {
            for &b in raw {
                code_byte(&mut pred, &mut enc, Some(b), &mut hist);
            }
        }
        Profile::Ptx => {
            let mut numbers = NumberModel::new();
            let mut pos = 0usize;
            while pos < raw.len() {
                match next_event(raw, pos) {
                    Event::Literal(b) => {
                        if b == MARKER || b == ESCAPE {
                            code_byte(&mut pred, &mut enc, Some(ESCAPE), &mut hist);
                        }
                        code_byte(&mut pred, &mut enc, Some(b), &mut hist);
                        pos += 1;
                    }
                    Event::Number { digits, value } => {
                        let (class, line) = pred.number_context();
                        code_byte(&mut pred, &mut enc, Some(MARKER), &mut hist);
                        numbers.code(&mut enc, class, line, Some(value));
                        pos += digits.max(1);
                    }
                }
            }
        }
    }
    enc.finish()
}

/// Decompress one block of exactly `raw_len` bytes into `out` (appended). Never fails: a
/// corrupt stream yields wrong bytes of the declared length, which the caller's hash check
/// catches; bounds are enforced by the caller before this runs.
pub fn decompress_block(
    packed: &[u8],
    raw_len: usize,
    profile: Profile,
    mem_log2: u8,
    out: &mut Vec<u8>,
) {
    let mut pred = Predictor::new(profile, mem_log2);
    let mut dec = Decoder::new(packed);
    let start = out.len();
    match profile {
        Profile::Stored | Profile::Generic | Profile::Elf => {
            let mut hist = Vec::with_capacity(raw_len);
            for _ in 0..raw_len {
                code_byte(&mut pred, &mut dec, None, &mut hist);
            }
            out.extend_from_slice(&hist);
        }
        Profile::Ptx => {
            let mut numbers = NumberModel::new();
            let mut hist: Vec<u8> = Vec::with_capacity(raw_len);
            while out.len() - start < raw_len {
                let (class, line) = pred.number_context();
                let b = code_byte(&mut pred, &mut dec, None, &mut hist);
                if b == ESCAPE {
                    let lit = code_byte(&mut pred, &mut dec, None, &mut hist);
                    out.push(lit);
                } else if b == MARKER {
                    let v = numbers.code(&mut dec, class, line, None);
                    push_decimal(out, v);
                } else {
                    out.push(b);
                }
            }
            out.truncate(start + raw_len);
        }
    }
}
