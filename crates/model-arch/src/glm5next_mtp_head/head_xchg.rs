// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The vocab-sharded draft head's cross-rank `(max, argmax)` exchange, as pure
//! functions over the exchange buffer's bytes.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - The buffer holds `world` BF16 logit lanes, then three BF16 base-256 digit lanes per rank:
//!   `4 * world` lanes, [`xchg_bytes`] bytes. At `world == 2` this is the 16-byte layout the
//!   two-rank exchange used before 2026-10-11.
//! - A rank writes only its own lanes and leaves the rest zero, so a BF16 SUM all-reduce
//!   delivers every rank's lanes untouched (`x + 0.0` is exact; a digit is an integer <= 255,
//!   which BF16 holds exactly).
//! - The pick is the highest logit; a tie goes to the lowest rank, identically on every rank.

use anyhow::{Result, bail};

/// 2026-10-11: The most ranks the exchange buffer is allocated for. [`super::init`] shards the
/// draft head only for worlds of `2..=MAX_RANKS`.
pub const MAX_RANKS: usize = 4;

/// 2026-10-11: Bytes of the exchange buffer `alloc_state` allocates (and the per-sequence
/// reserve charges): room for [`MAX_RANKS`] ranks of the log-probability layout
/// ([`xchg_lp_bytes`]), the larger one.
pub const BUFFER_BYTES: usize = 12 * MAX_RANKS;

/// 2026-10-11: Base-256 digits per token id: 24 bits, enough for any vocab below 16,777,216.
const DIGITS: usize = 3;

/// 2026-10-11: The largest token id the digits can carry.
const MAX_ID: usize = (1 << (8 * DIGITS)) - 1;

/// 2026-10-11: Bytes of the exchange at `world` ranks: what the all-reduce sums.
pub fn xchg_bytes(world: usize) -> usize {
    2 * (1 + DIGITS) * world
}

/// 2026-10-11: Bytes of the exchange with the draft's log-probability: the [`xchg_bytes`] lanes,
/// then two lanes per rank for its log-sum-exp (high part, then remainder).
pub fn xchg_lp_bytes(world: usize) -> usize {
    xchg_bytes(world) + 4 * world
}

fn bf16_of(x: f32) -> [u8; 2] {
    ((x.to_bits() >> 16) as u16).to_le_bytes()
}

fn f32_of(lane: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(lane) as u32) << 16)
}

/// 2026-10-11: Rank `rank`'s contribution: its max logit (the BF16 bytes the head wrote) in
/// lane `rank`, the digits of its global argmax `global_idx` in lanes
/// `world + 3 * rank ..`, and zeros elsewhere.
pub fn pack(world: usize, rank: usize, logit: [u8; 2], global_idx: usize) -> Result<Vec<u8>> {
    if world == 0 || world > MAX_RANKS || rank >= world {
        bail!("draft head exchange: rank {rank} of {world} (at most {MAX_RANKS} ranks)");
    }
    if global_idx > MAX_ID {
        bail!("draft head exchange: token id {global_idx} exceeds {MAX_ID}");
    }
    let mut buf = vec![0u8; xchg_bytes(world)];
    buf[rank * 2..][..2].copy_from_slice(&logit);
    for d in 0..DIGITS {
        let digit = ((global_idx >> (8 * d)) & 0xFF) as f32;
        let lane = world + rank * DIGITS + d;
        buf[lane * 2..][..2].copy_from_slice(&bf16_of(digit));
    }
    Ok(buf)
}

/// 2026-10-11: The winning global token id from the summed buffer.
pub fn pick(world: usize, summed: &[u8]) -> Result<u32> {
    if world == 0 || world > MAX_RANKS || summed.len() < xchg_bytes(world) {
        bail!(
            "draft head exchange: {} bytes for {world} ranks (need {})",
            summed.len(),
            xchg_bytes(world)
        );
    }
    let lane = |i: usize| f32_of([summed[i * 2], summed[i * 2 + 1]]);
    // 2026-10-11: Strictly greater, so the lowest rank keeps a tie (the two-rank exchange's
    // `lane(0) >= lane(1)`).
    let mut win = 0;
    for r in 1..world {
        if lane(r) > lane(win) {
            win = r;
        }
    }
    let idx = (0..DIGITS).fold(0usize, |a, d| {
        a + ((lane(world + win * DIGITS + d) as usize) << (8 * d))
    });
    Ok(idx as u32)
}

/// 2026-10-11: [`pack`] plus this rank's log-sum-exp over its vocab shard, `lse`, as two BF16
/// lanes: its top 16 bits, then the top 16 bits of the remainder (the device's
/// `glm5next_mtp_xchg_pack_lp`).
pub fn pack_lp(
    world: usize,
    rank: usize,
    logit: [u8; 2],
    global_idx: usize,
    lse: f32,
) -> Result<Vec<u8>> {
    let mut buf = pack(world, rank, logit, global_idx)?;
    buf.resize(xchg_lp_bytes(world), 0);
    let hi = bf16_of(lse);
    let lo = bf16_of(lse - f32_of(hi));
    let lane = 4 * world + 2 * rank;
    buf[lane * 2..][..2].copy_from_slice(&hi);
    buf[(lane + 1) * 2..][..2].copy_from_slice(&lo);
    Ok(buf)
}

/// 2026-10-11: [`pick`]'s winner and its global log-probability: the winner's logit minus the
/// log-sum-exp of every rank's (the device's `glm5next_mtp_xchg_pick_lp`).
pub fn pick_lp(world: usize, summed: &[u8]) -> Result<(u32, f32)> {
    if summed.len() < xchg_lp_bytes(world) {
        bail!(
            "draft head exchange: {} bytes for {world} ranks with log-probabilities (need {})",
            summed.len(),
            xchg_lp_bytes(world)
        );
    }
    let id = pick(world, summed)?;
    let lane = |i: usize| f32_of([summed[i * 2], summed[i * 2 + 1]]);
    let mut win = 0;
    for r in 1..world {
        if lane(r) > lane(win) {
            win = r;
        }
    }
    let lses: Vec<f32> = (0..world)
        .map(|r| lane(4 * world + 2 * r) + lane(4 * world + 2 * r + 1))
        .collect();
    let m = lses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = lses.iter().map(|l| (l - m).exp()).sum();
    Ok((id, lane(win) - (m + s.ln())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: The BF16 SUM all-reduce, lane by lane, as the device computes it: each lane's
    /// contributions added in f32 and rounded back to BF16 (exact here, since at most one
    /// contribution per lane is non-zero).
    fn all_reduce(packs: &[Vec<u8>]) -> Vec<u8> {
        let n = packs[0].len();
        let mut out = vec![0u8; n];
        for lane in 0..n / 2 {
            let s: f32 = packs
                .iter()
                .map(|p| f32_of([p[lane * 2], p[lane * 2 + 1]]))
                .sum();
            out[lane * 2..][..2].copy_from_slice(&bf16_of(s));
        }
        out
    }

    fn exchange(world: usize, picks: &[(f32, usize)]) -> u32 {
        let packs: Vec<Vec<u8>> = picks
            .iter()
            .enumerate()
            .map(|(r, &(v, idx))| pack(world, r, bf16_of(v), idx).unwrap())
            .collect();
        pick(world, &all_reduce(&packs)).unwrap()
    }

    /// 2026-10-11: The highest logit wins at every world size, wherever it sits.
    #[test]
    fn the_highest_logit_wins_at_every_world() {
        for world in 1..=MAX_RANKS {
            for best in 0..world {
                let picks: Vec<(f32, usize)> = (0..world)
                    .map(|r| {
                        (
                            if r == best { 9.5 } else { -3.0 + r as f32 },
                            51_627 * r + 17 * r,
                        )
                    })
                    .collect();
                assert_eq!(
                    exchange(world, &picks) as usize,
                    picks[best].1,
                    "world {world} best {best}"
                );
            }
        }
    }

    /// 2026-10-11: A tie goes to the lowest rank, as the two-rank exchange decided it.
    #[test]
    fn a_tie_goes_to_the_lowest_rank() {
        assert_eq!(
            exchange(3, &[(1.0, 5), (2.0, 60_000), (2.0, 120_000)]),
            60_000
        );
        assert_eq!(exchange(3, &[(2.0, 5), (2.0, 60_000), (2.0, 120_000)]), 5);
        assert_eq!(exchange(2, &[(-1.0, 7), (-1.0, 80_000)]), 7);
    }

    /// 2026-10-11: Every id up to the 24-bit limit survives the BF16 digits, including GLM-5.3's
    /// last row; one past the limit is refused rather than truncated.
    #[test]
    fn ids_round_trip_through_the_digits() {
        for id in [
            0usize, 1, 255, 256, 65_535, 65_536, 154_855, 154_879, MAX_ID,
        ] {
            assert_eq!(
                exchange(3, &[(0.5, id), (0.25, 3), (0.125, 4)]) as usize,
                id
            );
        }
        assert!(pack(3, 0, bf16_of(0.0), MAX_ID + 1).is_err());
    }

    /// 2026-10-11: Negative and very large logits compare correctly (sign and exponent survive
    /// the sum of one non-zero contribution with zeros).
    #[test]
    fn negative_and_large_logits_compare_correctly() {
        assert_eq!(exchange(3, &[(-40.0, 1), (-2.5, 2), (-7.0, 3)]), 2);
        assert_eq!(exchange(3, &[(1.0e30, 1), (3.0e38, 2), (-3.0e38, 3)]), 2);
    }

    /// 2026-10-11: At two ranks the bytes are the 16-byte layout the exchange had before it took
    /// any world: logit lanes 0-1, then rank 0's digits, then rank 1's.
    #[test]
    fn two_ranks_keep_the_previous_layout() {
        assert_eq!(xchg_bytes(2), 16);
        let p = pack(2, 1, bf16_of(3.0), 0x01_02_03).unwrap();
        assert_eq!(&p[2..4], &bf16_of(3.0));
        assert_eq!(&p[4 + 3 * 2..][..2], &bf16_of(3.0)); // digit 0 of rank 1 = 0x03
        assert_eq!(&p[4 + 4 * 2..][..2], &bf16_of(2.0));
        assert_eq!(&p[4 + 5 * 2..][..2], &bf16_of(1.0));
        assert!(p[..2].iter().chain(&p[4..10]).all(|&b| b == 0));
    }

    /// 2026-10-11: The log-probability of the winner over the whole vocab, from each rank's
    /// shard log-sum-exp, matches a direct log-softmax over the concatenated shards; the id is
    /// the plain exchange's.
    #[test]
    fn the_log_probability_is_the_whole_vocab_log_softmax() {
        let shards: [&[f32]; 3] = [&[1.0, 3.5, -2.0], &[0.25, 4.0, 2.0], &[-1.0, 0.0, 3.75]];
        let lse = |xs: &[f32]| {
            let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            m + xs.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
        };
        let packs: Vec<Vec<u8>> = shards
            .iter()
            .enumerate()
            .map(|(r, xs)| {
                let (j, &v) = xs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                pack_lp(3, r, bf16_of(v), 3 * r + j, lse(xs)).unwrap()
            })
            .collect();
        let (id, lp) = pick_lp(3, &all_reduce(&packs)).unwrap();
        let all: Vec<f32> = shards.iter().flat_map(|s| s.iter().copied()).collect();
        assert_eq!(id, 4);
        assert!((lp - (4.0 - lse(&all))).abs() < 1e-3, "lp {lp}");
        // 2026-10-11: The plain exchange of the same lanes picks the same id.
        assert_eq!(pick(3, &all_reduce(&packs)).unwrap(), 4);
    }

    /// 2026-10-11: The high and low lanes keep the log-sum-exp to 16 mantissa bits, where one
    /// BF16 lane keeps 8.
    #[test]
    fn the_split_lanes_keep_the_log_sum_exp() {
        for lse in [12.345_678f32, -0.003_21, 31.0, 1.0e-3] {
            let p = pack_lp(2, 1, bf16_of(0.0), 9, lse).unwrap();
            let got = f32_of([p[10 * 2], p[10 * 2 + 1]]) + f32_of([p[11 * 2], p[11 * 2 + 1]]);
            assert!(
                (got - lse).abs() <= lse.abs() * 2.0f32.powi(-15),
                "{lse} -> {got}"
            );
        }
        assert_eq!(xchg_lp_bytes(2), 24);
        assert!(pick_lp(2, &[0u8; 16]).is_err());
    }

    /// 2026-10-11: The buffer `alloc_state` allocates holds every world the shard admits, and a
    /// rank or world outside it is refused.
    #[test]
    fn the_buffer_holds_every_admitted_world_and_refuses_others() {
        for world in 1..=MAX_RANKS {
            assert!(xchg_bytes(world) <= BUFFER_BYTES);
            assert!(xchg_lp_bytes(world) <= BUFFER_BYTES);
        }
        assert!(pack(MAX_RANKS + 1, 0, [0, 0], 1).is_err());
        assert!(pack(3, 3, [0, 0], 1).is_err());
        assert!(pick(3, &[0u8; 16]).is_err());
    }
}
