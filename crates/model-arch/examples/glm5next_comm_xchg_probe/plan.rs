// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The pure core of the `glm5next_comm_xchg_probe` example: the measured patterns,
//! their byte counts, the bandwidth each reports, the fill bytes every rank sends and the
//! checks of what arrived. No I/O; `main.rs` measures and prints.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: a pattern moves whole hidden-state rows; a received slot is accepted only when
//! every byte equals the sending rank's fill byte.

use anyhow::{Result, ensure};

/// 2026-10-15: One GLM-5.3 hidden-state row in BF16 (4096 x 2 bytes).
pub const ROW_BYTES: usize = 4096 * 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 2026-10-15: A grouped send/recv of `rows` rows to and from every peer, as the SP
    /// reduce-scatter / all-gather of `glm5next_sp_mhc` issues it.
    Xchg,
    /// 2026-10-15: An in-place BF16 sum all-reduce of `rows` rows through the serve's
    /// `NcclBackend::all_reduce` (the one-shot path off).
    AllReduce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pattern {
    pub name: &'static str,
    pub kind: Kind,
    pub rows: usize,
    /// 2026-10-15: Operations issued back to back on the stream per timed repetition.
    pub times: usize,
    /// 2026-10-15: Whether large device-to-device copies run on a second stream meanwhile,
    /// standing in for the prefill's DRAM traffic.
    pub contended: bool,
}

const fn p(name: &'static str, kind: Kind, rows: usize, times: usize, contended: bool) -> Pattern {
    Pattern {
        name,
        kind,
        rows,
        times,
        contended,
    }
}

/// 2026-10-15: The 32k prefill's SP message sizes per peer: the overlap piece (8208 rows / 3
/// ranks / 4 pieces = 684), one section's four pieces back to back, the whole shard (2736) and
/// the tail pass (246); the section again under DRAM traffic; the all-reduce at the one-shot
/// ceiling (1 MiB = 128 rows) and at a shard.
pub const PATTERNS: [Pattern; 7] = [
    p("piece", Kind::Xchg, 684, 1, false),
    p("sect4", Kind::Xchg, 684, 4, false),
    p("shard", Kind::Xchg, 2736, 1, false),
    p("tail", Kind::Xchg, 246, 1, false),
    p("sect4mc", Kind::Xchg, 684, 4, true),
    p("ar1m", Kind::AllReduce, 128, 1, false),
    p("arshard", Kind::AllReduce, 2736, 1, false),
];

impl Pattern {
    /// 2026-10-15: Bytes per operation: per peer for an exchange, the whole buffer for an
    /// all-reduce.
    pub fn bytes(&self) -> usize {
        self.rows * ROW_BYTES
    }

    /// 2026-10-15: GB/s from the milliseconds of one timed repetition: per peer per direction
    /// for an exchange; the ring bus bandwidth `2 (w - 1) / w * bytes / t` for an all-reduce.
    pub fn gbps(&self, world: usize, ms: f64) -> f64 {
        let moved = (self.bytes() * self.times) as f64;
        let factor = match self.kind {
            Kind::Xchg => 1.0,
            Kind::AllReduce => 2.0 * (world - 1) as f64 / world as f64,
        };
        factor * moved / (ms * 1e6)
    }
}

/// 2026-10-15: Bytes of the send (and of the receive) buffer: one slot per rank at the largest
/// exchange.
pub fn xchg_buffer_bytes(world: usize) -> usize {
    let rows = PATTERNS
        .iter()
        .filter(|p| p.kind == Kind::Xchg)
        .map(|p| p.rows)
        .max()
        .unwrap_or(0);
    rows * ROW_BYTES * world
}

/// 2026-10-15: Bytes of the all-reduce buffer.
pub fn all_reduce_buffer_bytes() -> usize {
    PATTERNS
        .iter()
        .filter(|p| p.kind == Kind::AllReduce)
        .map(Pattern::bytes)
        .max()
        .unwrap_or(0)
}

/// 2026-10-15: The byte rank `rank` fills its send buffer with; distinct per rank and never 0,
/// the receive buffer's initial value.
pub fn fill(rank: usize) -> Result<u8> {
    ensure!(rank < 15, "rank {rank}: the fill bytes cover ranks 0..15");
    Ok(0x11 * (rank as u8 + 1))
}

/// 2026-10-15: Check that slot `q` of `recv` (slots of `bytes`) holds peer `q`'s fill for every
/// peer `q != rank`.
pub fn check_recv(recv: &[u8], world: usize, rank: usize, bytes: usize) -> Result<()> {
    ensure!(
        recv.len() >= world * bytes,
        "receive buffer shorter than {world} slots"
    );
    for q in (0..world).filter(|&q| q != rank) {
        let want = fill(q)?;
        let slot = &recv[q * bytes..(q + 1) * bytes];
        let bad = slot.iter().filter(|&&b| b != want).count();
        ensure!(
            bad == 0,
            "slot from rank {q}: {bad} of {bytes} bytes differ from {want:#04x}"
        );
    }
    Ok(())
}

/// 2026-10-15: The BF16 bits of `x`, for small integers (exact in BF16).
pub fn bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// 2026-10-15: Check an all-reduce of BF16 ones across `world` ranks: every element is `world`.
pub fn check_all_reduce_ones(out: &[u8], world: usize) -> Result<()> {
    let want = bf16_bits(world as f32);
    let bad = out
        .chunks_exact(2)
        .filter(|c| u16::from_le_bytes([c[0], c[1]]) != want)
        .count();
    ensure!(
        bad == 0,
        "all-reduce of ones: {bad} elements differ from {world}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_bandwidth_is_per_peer_per_direction() {
        let sect4 = PATTERNS[1];
        // 2026-10-15: 4 x 5,603,328 bytes in 2.4 ms.
        let g = sect4.gbps(3, 2.4);
        assert!((g - 4.0 * 5_603_328.0 / 2.4e6).abs() < 1e-9);
    }

    #[test]
    fn all_reduce_bandwidth_is_ring_bus_bandwidth() {
        let ar = PATTERNS[5];
        assert_eq!(ar.bytes(), 1 << 20);
        let g = ar.gbps(3, 1.0);
        assert!((g - (4.0 / 3.0) * (1u64 << 20) as f64 / 1e6).abs() < 1e-9);
    }

    #[test]
    fn buffers_cover_the_largest_patterns() {
        assert_eq!(xchg_buffer_bytes(3), 2736 * ROW_BYTES * 3);
        assert_eq!(all_reduce_buffer_bytes(), 2736 * ROW_BYTES);
    }

    #[test]
    fn fills_are_distinct_and_never_zero() {
        let f: Vec<u8> = (0..15).map(|r| fill(r).unwrap()).collect();
        assert!(f.iter().all(|&b| b != 0));
        let mut s = f.clone();
        s.dedup();
        assert_eq!(s.len(), f.len());
        assert!(fill(15).is_err());
    }

    #[test]
    fn recv_check_accepts_peer_fills_and_ignores_own_slot() {
        let bytes = 4;
        let mut r = vec![0u8; 3 * bytes];
        r[..bytes].fill(fill(0).unwrap());
        r[2 * bytes..].fill(fill(2).unwrap());
        check_recv(&r, 3, 1, bytes).unwrap();
    }

    #[test]
    fn recv_check_rejects_one_wrong_byte_and_a_swapped_slot() {
        let bytes = 4;
        let mut r = vec![0u8; 3 * bytes];
        r[..bytes].fill(fill(0).unwrap());
        r[2 * bytes..].fill(fill(2).unwrap());
        r[2 * bytes + 3] = fill(1).unwrap();
        assert!(check_recv(&r, 3, 1, bytes).is_err());
        let mut s = vec![0u8; 3 * bytes];
        s[..bytes].fill(fill(2).unwrap());
        s[2 * bytes..].fill(fill(0).unwrap());
        assert!(check_recv(&s, 3, 1, bytes).is_err());
        assert!(check_recv(&r[..2 * bytes], 3, 1, bytes).is_err());
    }

    #[test]
    fn all_reduce_check_wants_world_in_every_element() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(3.0), 0x4040);
        let good: Vec<u8> = [0x4040u16; 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        check_all_reduce_ones(&good, 3).unwrap();
        let mut bad = good.clone();
        bad[15] = 0x3f;
        assert!(check_all_reduce_ones(&bad, 3).is_err());
        assert!(check_all_reduce_ones(&good, 2).is_err());
    }
}
