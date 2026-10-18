// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The sharded form of [`crate::feed_argmax`]: each rank turns its slice of a
//! logits row into two `u64` keys, the keys combine by `max` (one all-reduce of two words
//! instead of the row), and the combined keys resolve to the token [`crate::feed_argmax`]
//! returns for the whole row, ties included. The device twins are `argmax_key_shard_bf16` and
//! `argmax_key_resolve` (`kernels/gb10/common/argmax_key.cu`).
//!
//! Owner: metrale-sampling.
//! Invariants:
//! - For every split of a row into contiguous slices,
//!   `resolve(max over slices of keys(slice)) == feed_argmax(row)` (NaN included: it never
//!   wins either reduction).
//!
//! The `p` key (the plain argmax, `argmax_bf16_batch`'s scan and tree) orders, highest first:
//! the value, then the smaller bit-reversed thread id `i mod BLOCK` (the tree keeps the lower
//! thread at each halving, so bit 0 of the thread id decides first), then the smaller `i`
//! within a thread (a thread keeps its first strict maximum). A value at or below the
//! `-1e30` floor, or NaN, gives key 0: no candidate, and a row without one answers 0.
//!
//! The `q` key (the masked re-pick) orders the value with the two mask ids at `-inf`, then the
//! larger index. NaN gives key 0 (the scan starts at `(-inf, 0)`, so an all-NaN row answers 0
//! there too). `-0.0` keys as `+0.0` in both, since the scans compare them equal.

use crate::feed_argmax::{BLOCK, PLAIN_FLOOR};

/// 2026-10-10: Bits of the thread id in the `p` key (`BLOCK` = 1024).
const TID_BITS: u32 = 10;
/// 2026-10-10: Bits of `i / BLOCK` in the `p` key; rows up to 2^32 entries.
const STRIDE_BITS: u32 = 22;
const _: () = assert!(1 << TID_BITS == BLOCK);

/// 2026-10-10: An order-preserving map of f32 to u32 (`-0.0` folded into `+0.0`). Not for NaN.
#[inline]
pub fn order_bits(v: f32) -> u32 {
    let v = if v == 0.0 { 0.0 } else { v };
    let b = v.to_bits();
    if b & 0x8000_0000 == 0 {
        b | 0x8000_0000
    } else {
        !b
    }
}

#[inline]
fn bitrev_tid(t: u32) -> u32 {
    t.reverse_bits() >> (32 - TID_BITS)
}

/// 2026-10-10: The `p` key of entry `i` (global index) holding `v`; 0 when it cannot win.
#[inline]
pub fn plain_key(v: f32, i: u32) -> u64 {
    if v.is_nan() || v <= PLAIN_FLOOR {
        return 0;
    }
    let t = i % BLOCK as u32;
    let k = i / BLOCK as u32;
    let tie =
        ((!bitrev_tid(t) & ((1 << TID_BITS) - 1)) << STRIDE_BITS) | (!k & ((1 << STRIDE_BITS) - 1));
    (u64::from(order_bits(v)) << 32) | u64::from(tie)
}

/// 2026-10-10: The `q` key of entry `i` holding `v` under `mask`; 0 for NaN.
#[inline]
pub fn masked_key(v: f32, i: u32, mask: [u32; 2]) -> u64 {
    let w = if i == mask[0] || i == mask[1] {
        f32::NEG_INFINITY
    } else {
        v
    };
    if w.is_nan() {
        return 0;
    }
    (u64::from(order_bits(w)) << 32) | u64::from(i)
}

/// 2026-10-10: `(p, q)` keys of a slice of a row whose first entry has global index `base`.
pub fn shard_keys(slice: &[f32], base: u32, mask: [u32; 2]) -> (u64, u64) {
    slice.iter().enumerate().fold((0, 0), |(p, q), (j, &v)| {
        let i = base + j as u32;
        (p.max(plain_key(v, i)), q.max(masked_key(v, i, mask)))
    })
}

/// 2026-10-10: The index a `p` key names; 0 for the empty key.
pub fn plain_index(key: u64) -> u32 {
    if key == 0 {
        return 0;
    }
    let tie = key as u32;
    let t = bitrev_tid(!(tie >> STRIDE_BITS) & ((1 << TID_BITS) - 1));
    let k = !tie & ((1 << STRIDE_BITS) - 1);
    k * BLOCK as u32 + t
}

/// 2026-10-10: The token for the combined keys: `q`'s index when `p` lands on a mask id,
/// else `p`'s ([`crate::feed_argmax::feed_argmax_f32`]'s rule).
pub fn resolve(p_key: u64, q_key: u64, mask: [u32; 2]) -> u32 {
    let p = plain_index(p_key);
    if p == mask[0] || p == mask[1] {
        q_key as u32
    } else {
        p
    }
}
