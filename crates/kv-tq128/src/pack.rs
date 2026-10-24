// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Index payloads. A trellis path is stored as the bitstream whose `L`-bit window
//! at bit `k * t` (LSB first) is state `t`: the first state's low `L - k` bits, then each
//! state's top `k` bits (its symbol). turbo4_n128 stores two 4-bit indices per byte, the even
//! element in the low nibble. Layout as in spiritbuun/buun-llama-cpp (MIT; credit spiritbuun).
//!
//! Owner: kv-tq128.
//! Invariants: unused tail bits are zero; [`trellis_state_at`] reads at most two bytes, the
//! read the kernels do (`(qs[b] | qs[b + 1] << 8) >> (bit % 8)`).

use crate::Tq128Error;
use crate::trellis::Trellis;

fn set_bits(out: &mut [u8], bit: usize, width: u32, value: usize) {
    for i in 0..width {
        if (value >> i) & 1 == 1 {
            let b = bit + usize::try_from(i).unwrap_or(0);
            out[b / 8] |= 1 << (b % 8);
        }
    }
}

/// 2026-10-18: Write `states` (a walk of `trellis`) as its bitstream into `out`, which must be
/// exactly `stream_bits(states.len())` bits rounded up to bytes.
pub fn pack_trellis(trellis: Trellis, states: &[u16], out: &mut [u8]) -> Result<(), Tq128Error> {
    let expected = trellis.stream_bits(states.len()).div_ceil(8);
    if out.len() != expected {
        return Err(Tq128Error::BlockLength {
            expected,
            found: out.len(),
        });
    }
    out.fill(0);
    let Some(&first) = states.first() else {
        return Ok(());
    };
    let prefix = trellis.l() - trellis.k();
    set_bits(out, 0, prefix, usize::from(first) & ((1 << prefix) - 1));
    let k = usize::try_from(trellis.k()).unwrap_or(0);
    let start = usize::try_from(prefix).unwrap_or(0);
    for (t, &s) in states.iter().enumerate() {
        set_bits(
            out,
            start + k * t,
            trellis.k(),
            trellis.symbol(usize::from(s)),
        );
    }
    Ok(())
}

/// 2026-10-18: State `t` of a packed stream: the `L`-bit window at bit `k * t`. A window that
/// runs past the end reads the missing byte as zero.
pub fn trellis_state_at(trellis: Trellis, stream: &[u8], t: usize) -> u16 {
    let bit = usize::try_from(trellis.k()).unwrap_or(0) * t;
    let lo = u16::from(stream.get(bit / 8).copied().unwrap_or(0));
    let hi = u16::from(stream.get(bit / 8 + 1).copied().unwrap_or(0));
    let window = (lo | (hi << 8)) >> (bit % 8);
    window & ((1 << trellis.l()) - 1)
}

/// 2026-10-18: Pack 4-bit indices two per byte, even element low. `out` must hold
/// `idx.len() / 2` bytes and every index must be below 16.
pub fn pack_nibbles(idx: &[u8], out: &mut [u8]) -> Result<(), Tq128Error> {
    let expected = idx.len().div_ceil(2);
    if out.len() != expected {
        return Err(Tq128Error::BlockLength {
            expected,
            found: out.len(),
        });
    }
    for (o, pair) in out.iter_mut().zip(idx.chunks(2)) {
        let hi = pair.get(1).copied().unwrap_or(0);
        *o = (pair[0] & 0xF) | ((hi & 0xF) << 4);
    }
    Ok(())
}

/// 2026-10-18: Index `t` of a nibble payload.
pub fn nibble_at(payload: &[u8], t: usize) -> u8 {
    (payload[t / 2] >> (4 * (t % 2))) & 0xF
}
