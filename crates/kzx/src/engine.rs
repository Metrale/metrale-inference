// SPDX-License-Identifier: MIT OR Apache-2.0

//! The block coder: the byte loop shared, step for step, by compression and decompression.
//! Each byte is either flagged (a long match predicted it: one binary decision) or coded as
//! eight bits through the predictor. The loop runs exactly `raw_len` iterations, so a block's
//! work is bounded by its declared length before the first bit is read.

use crate::coder::{Decoder, Encoder};
use crate::format::Profile;
use crate::predictor::Predictor;
use alloc::vec::Vec;

/// Compress one block with `profile` and `mem_log2`; the caller frames it.
pub fn compress_block(raw: &[u8], profile: Profile, mem_log2: u8) -> Vec<u8> {
    let mut pred = Predictor::new(profile, mem_log2);
    let mut enc = Encoder::with_capacity(raw.len() / 4 + 16);
    #[cfg(kzx_stats)]
    let (mut flagged, mut hits, mut flag_bytes_before) = (0u64, 0u64, 0usize);
    for i in 0..raw.len() {
        let byte = raw[i];
        if let Some(e) = pred.lzp_candidate() {
            let hit = e == byte;
            #[cfg(kzx_stats)]
            { flagged += 1; hits += u64::from(hit); }
            enc.encode(u32::from(hit), pred.lzp_p());
            pred.lzp_update(hit);
            if hit {
                pred.byte_done(&raw[..=i]);
                continue;
            }
            pred.set_excluded(e);
        }
        for bitpos in 0..8u32 {
            let bit = (u32::from(byte) >> (7 - bitpos)) & 1;
            let p = pred.p();
            enc.encode(bit, p);
            pred.update(bit);
        }
        pred.byte_done(&raw[..=i]);
    }
    #[cfg(kzx_stats)]
    {
        extern crate std;
        let _ = flag_bytes_before;
        std::eprintln!("stats: bytes {} flagged {} hits {} packed {}", raw.len(), flagged, hits, enc.len());
    }
    enc.finish()
}

/// Decompress one block of exactly `raw_len` bytes into `out` (appended). Never fails: a
/// corrupt stream yields wrong bytes of the declared length, which the caller's hash check
/// catches; bounds are enforced by the caller before this runs.
pub fn decompress_block(packed: &[u8], raw_len: usize, profile: Profile, mem_log2: u8, out: &mut Vec<u8>) {
    let mut pred = Predictor::new(profile, mem_log2);
    let mut dec = Decoder::new(packed);
    let start = out.len();
    for _ in 0..raw_len {
        if let Some(e) = pred.lzp_candidate() {
            let hit = dec.decode(pred.lzp_p()) == 1;
            pred.lzp_update(hit);
            if hit {
                out.push(e);
                pred.byte_done(&out[start..]);
                continue;
            }
            pred.set_excluded(e);
        }
        let mut c = 1u32;
        for _ in 0..8 {
            let p = pred.p();
            let bit = dec.decode(p);
            pred.update(bit);
            c = (c << 1) | bit;
        }
        out.push(u8::try_from(c & 0xff).unwrap_or(0));
        pred.byte_done(&out[start..]);
    }
}
