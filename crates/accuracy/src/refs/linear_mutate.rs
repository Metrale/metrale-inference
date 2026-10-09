// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The linear reference's shard layout and its data mutations: which bytes a
//! mutation edits and which output columns it can reach (the comparison always includes them).
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A mutation that cannot apply to the case (no block scale to corrupt, one shard to shift) is
//!   an error naming why, never a silent no-op.

use crate::case::{Case, Enc, Shard};
use crate::contract::Split;
use crate::inputs::SplitMix64;
use crate::mutation::Mutation;

/// 2026-10-09: The production shard layout of `n` output rows over `split.world` ranks: equal
/// shards rounded up to `split.align`, the last taking the remainder.
pub fn shards(n: usize, split: Split) -> Vec<Shard> {
    let (world, align) = (split.world.max(1) as usize, split.align.max(1) as usize);
    let size = n.div_ceil(world).div_ceil(align) * align;
    (0..world)
        .map(|j| (j * size).min(n))
        .map(|lo| Shard {
            lo,
            hi: (lo + size).min(n),
            out_at: lo,
        })
        .filter(|s| s.hi > s.lo)
        .collect()
}

fn flip_scale(enc: Enc, b: &mut [u8], i: usize) -> Result<(), String> {
    match enc {
        Enc::Ue4m3 | Enc::E4m3 => {
            let flipped = b[i] ^ 0x08;
            // 2026-10-09: Never make a NaN code (a NaN is caught trivially; the mutation must be
            // a plausible wrong scale).
            b[i] = if flipped & 0x7f == 0x7f {
                b[i] ^ 0x10
            } else {
                flipped
            };
        }
        Enc::F32 => {
            let mut v = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
            v ^= 1 << 23;
            b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        Enc::Ue8m0 => b[i] ^= 0x01,
        other => return Err(format!("a {other:?} scale")),
    }
    Ok(())
}

/// 2026-10-09: Apply `m` to a linear case; returns the output columns it reaches that a sample
/// might miss (empty when it reaches every column, which every sample sees).
pub fn mutate(case: &mut Case, m: &Mutation, rng: &mut SplitMix64) -> Result<Vec<usize>, String> {
    let n = case.tensor("w")?.dims[0];
    match m {
        Mutation::CorruptBlockScale => {
            let t = case
                .tensors
                .get_mut("w_block")
                .ok_or("the weight has no block scales")?;
            let (rows, groups) = (t.dims[0], t.dims[1]);
            let (rb, g) = (
                rng.below(rows as u64) as usize,
                rng.below(groups as u64) as usize,
            );
            let block_rows = case.scalars.get("w_block_rows").map_or(1, |v| *v as usize);
            flip_scale(t.enc, &mut t.bytes, rb * groups + g)?;
            Ok((rb * block_rows..((rb + 1) * block_rows).min(n)).collect())
        }
        Mutation::SwapScaleGranularity => {
            let t = case
                .tensors
                .get_mut("w_block")
                .ok_or("the weight has no block scales")?;
            let (rows, groups) = (t.dims[0], t.dims[1]);
            if groups < 2 {
                return Err("one scale group along K: no coarser granularity to swap to".into());
            }
            let w = t.enc.bytes_for(1);
            for r in 0..rows {
                for g in (1..groups).step_by(2) {
                    let (dst, src) = ((r * groups + g) * w, (r * groups + g - 1) * w);
                    let tmp: Vec<u8> = t.bytes[src..src + w].to_vec();
                    t.bytes[dst..dst + w].copy_from_slice(&tmp);
                }
            }
            Ok(Vec::new())
        }
        Mutation::SplitOffByOne => {
            if case.split.len() < 2 {
                return Err("the case is not split across shards".into());
            }
            let mut reach = Vec::new();
            for s in case.split.iter_mut().skip(1) {
                s.out_at += 1;
                if s.out_at + (s.hi - s.lo) > n {
                    s.hi -= 1;
                }
                reach.extend(s.out_at.saturating_sub(1)..(s.out_at + 2).min(n));
            }
            Ok(reach)
        }
        other => Err(format!("`{}` does not apply to a projection", other.name())),
    }
}
