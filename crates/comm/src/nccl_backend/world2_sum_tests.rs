// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Why the one-shot sum of two partials has the 2-rank exchange's bytes. The 2-rank
//! exchange writes `__hadd(a, b)`: the exact sum rounded once to BF16 (round to nearest, ties to
//! even). The one-shot's consumers (`bf16_add_rank_sum` and the fused kernels that copy its
//! arithmetic) write `bf16_rn(fp32(fp32(+0 + a) + b))`: the exact sum rounded to FP32, then to
//! BF16. Rounding a sum twice gives the once-rounded result whenever the wider format has at
//! least `2p + 1` significant bits for a `p`-bit target: FP32 has 24, BF16 needs 17. The one
//! exception is the sign of a zero: `(-0) + (-0)` is -0 for `__hadd` and +0 from an accumulator
//! started at +0.
//!
//! The reference here is exact: each BF16 value is an integer times a power of two, the sum is
//! formed in `i128`, and one rounding routine rounds an exact value to any precision. The FP32 arm
//! uses the host's IEEE `f32` addition, which keeps subnormals as the GB10 build does (no FTZ).
//!
//! Owner: metrale-comm.
//! Invariants: none beyond the types.

/// 2026-10-16: The BF16 exponent range (FP32's): smallest normal exponent and largest exponent.
const EMIN: i32 = -126;
const EMAX: i32 = 127;

/// 2026-10-16: A finite value `(-1)^neg * sig * 2^exp`, exactly.
#[derive(Clone, Copy, Debug)]
struct Exact {
    neg: bool,
    sig: u128,
    exp: i32,
}

/// 2026-10-16: A rounded value, or infinity after overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rounded {
    Finite { neg: bool, sig: u128, exp: i32 },
    Inf { neg: bool },
}

/// 2026-10-16: Round `v` to `p` significant bits, subnormals below 2^EMIN, ties to even,
/// overflow to infinity.
fn round(v: Exact, p: u32) -> Rounded {
    if v.sig == 0 {
        return Rounded::Finite {
            neg: v.neg,
            sig: 0,
            exp: 0,
        };
    }
    let top = v.exp + (127 - v.sig.leading_zeros() as i32);
    let quantum = (top - (p as i32 - 1)).max(EMIN - (p as i32 - 1));
    let (sig, exp) = if quantum <= v.exp {
        (v.sig << (v.exp - quantum) as u32, quantum)
    } else {
        let shift = (quantum - v.exp) as u32;
        if shift >= 127 {
            // 2026-10-16: The whole value sits under half a quantum (sig < 2^126 here).
            (0, quantum)
        } else {
            let kept = v.sig >> shift;
            let rem = v.sig & ((1u128 << shift) - 1);
            let half = 1u128 << (shift - 1);
            let up = rem > half || (rem == half && kept & 1 == 1);
            (kept + u128::from(up), quantum)
        }
    };
    normalize(v.neg, sig, exp)
}

/// 2026-10-16: Canonical form (odd significand) with the overflow check, so equal values compare
/// equal however they were reached.
fn normalize(neg: bool, mut sig: u128, mut exp: i32) -> Rounded {
    if sig == 0 {
        return Rounded::Finite {
            neg,
            sig: 0,
            exp: 0,
        };
    }
    while sig & 1 == 0 {
        sig >>= 1;
        exp += 1;
    }
    let top = exp + (127 - sig.leading_zeros() as i32);
    if top > EMAX {
        return Rounded::Inf { neg };
    }
    Rounded::Finite { neg, sig, exp }
}

/// 2026-10-16: A finite BF16 bit pattern as an exact value.
fn bf16_exact(bits: u16) -> Exact {
    let neg = bits >> 15 == 1;
    let e = i32::from((bits >> 7) & 0xff);
    let m = u128::from(bits & 0x7f);
    if e == 0 {
        Exact {
            neg,
            sig: m,
            exp: EMIN - 7,
        }
    } else {
        Exact {
            neg,
            sig: 0x80 | m,
            exp: e - 127 - 7,
        }
    }
}

/// 2026-10-16: A finite `f32` as an exact value.
fn f32_exact(x: f32) -> Exact {
    let bits = x.to_bits();
    let neg = bits >> 31 == 1;
    let e = ((bits >> 23) & 0xff) as i32;
    let m = u128::from(bits & 0x7f_ffff);
    if e == 0 {
        Exact {
            neg,
            sig: m,
            exp: EMIN - 23,
        }
    } else {
        Exact {
            neg,
            sig: 0x80_0000 | m,
            exp: e - 127 - 23,
        }
    }
}

/// 2026-10-16: The exact sum of two finite BF16 values, with IEEE's zero sign under round to
/// nearest: -0 only for (-0) + (-0). A gap over 70 binades replaces the smaller term by a
/// sticky unit 62 binades under the larger one's last bit, which no rounding to 24 or fewer bits
/// can tell from the true value (both lie strictly inside the same rounding interval).
fn exact_sum(a: u16, b: u16) -> Exact {
    let (x, y) = (bf16_exact(a), bf16_exact(b));
    if x.sig == 0 && y.sig == 0 {
        return Exact {
            neg: x.neg && y.neg,
            sig: 0,
            exp: 0,
        };
    }
    let (big, small) = if y.sig == 0 || (x.sig != 0 && x.exp >= y.exp) {
        (x, y)
    } else {
        (y, x)
    };
    // 2026-10-16: A zero term adds nothing and sets no alignment (its `exp` is a placeholder).
    let (small_sig, gap) = if small.sig == 0 {
        (0, 0)
    } else if big.exp - small.exp > 70 {
        (1u128, 62)
    } else {
        (small.sig, (big.exp - small.exp).max(0) as u32)
    };
    let base = big.exp - gap as i32;
    let bs = big.sig << gap;
    let (neg, sig) = if small_sig == 0 || big.neg == small.neg {
        (big.neg, bs + small_sig)
    } else if bs >= small_sig {
        (big.neg, bs - small_sig)
    } else {
        (small.neg, small_sig - bs)
    };
    Exact {
        neg: if sig == 0 { false } else { neg },
        sig,
        exp: base,
    }
}

/// 2026-10-16: The 2-rank exchange's result for a finite pair: the exact sum rounded once to BF16.
fn hadd(a: u16, b: u16) -> Rounded {
    round(exact_sum(a, b), 8)
}

/// 2026-10-16: The one-shot result for a finite pair: `acc = +0; acc += a; acc += b` in FP32,
/// then BF16 round to nearest even.
fn rank_sum(a: u16, b: u16) -> Rounded {
    let f = |h: u16| f32::from_bits(u32::from(h) << 16);
    let mut acc = 0.0f32;
    acc += f(a);
    acc += f(b);
    if acc.is_infinite() {
        return Rounded::Inf {
            neg: acc.is_sign_negative(),
        };
    }
    round(f32_exact(acc), 8)
}

/// 2026-10-16: The control: an intermediate of `mid` bits between the exact sum and BF16.
fn twice(a: u16, b: u16, mid: u32) -> Rounded {
    match round(exact_sum(a, b), mid) {
        Rounded::Finite { neg, sig, exp } => round(Exact { neg, sig, exp }, 8),
        inf => inf,
    }
}

fn finite(h: u16) -> bool {
    (h >> 7) & 0xff != 0xff
}

/// 2026-10-16: Every bit pattern with every exponent, both signs, and the mantissas at the edges
/// of a binade, a tie bit and an odd/even neighbour: the partners that reach every alignment gap
/// and every tie the double rounding could get wrong.
fn partners() -> Vec<u16> {
    let mut v = Vec::new();
    for sign in [0u16, 0x8000] {
        for e in 0u16..0xff {
            for m in [0u16, 1, 2, 0x3f, 0x40, 0x41, 0x7e, 0x7f] {
                v.push(sign | e << 7 | m);
            }
        }
    }
    v
}

/// 2026-10-16: The first terms the default (debug) run sweeps: every 61st pattern, plus both zeros,
/// the 16-bit witness, the largest finite and the smallest subnormals. About 1,080 first terms
/// against [`partners`] keep the default run near four million pairs; the exhaustive sweep is
/// the ignored test.
fn sampled_firsts() -> Vec<u16> {
    let mut v: Vec<u16> = (0..=u16::MAX)
        .step_by(61)
        .chain([0x0000, 0x8000, 0x3f81, 0x7f7f, 0xff7f, 0x0001, 0x8001])
        .filter(|&a| finite(a))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn the_reference_rounds_known_values() {
    // 2026-10-16: 1 + 2^-8 is a tie between 1 and 1 + 2^-7 and goes to the even 1; 1 + 3 * 2^-8
    // is a tie that goes up to 1 + 2^-6; 1 + 2^-8 + 2^-30 is past the tie and goes up.
    let one = Exact {
        neg: false,
        sig: 1 << 30,
        exp: -30,
    };
    let at = |extra: u128| {
        round(
            Exact {
                sig: one.sig + extra,
                ..one
            },
            8,
        )
    };
    assert_eq!(at(0), round(one, 8));
    assert_eq!(at(1 << 22), round(one, 8), "tie to even (down)");
    assert_eq!(
        at(3 << 22),
        round(
            Exact {
                neg: false,
                sig: 0x81 + 1,
                exp: -7
            },
            8
        ),
        "tie to even (up)"
    );
    assert_eq!(
        at((1 << 22) + 1),
        round(
            Exact {
                neg: false,
                sig: 0x81,
                exp: -7
            },
            8
        ),
        "past the tie"
    );
    // 2026-10-16: The largest BF16 plus half its ULP overflows; plus less stays finite.
    let max = bf16_exact(0x7f7f);
    assert_eq!(
        hadd(0x7f7f, 0x7b00),
        Rounded::Inf { neg: false },
        "0x7b00 is half an ULP of max: a tie to the even 2^128"
    );
    assert_eq!(
        round(max, 8),
        hadd(0x7f7f, 0x7a80),
        "a quarter ULP rounds back to max"
    );
    // 2026-10-16: The smallest subnormal plus itself is exact; half of it would be a tie to 0.
    assert_eq!(hadd(0x0001, 0x0001), round(bf16_exact(0x0002), 8));
}

#[test]
fn the_one_shot_sum_of_two_partials_matches_the_2rank_add() {
    let partners = partners();
    let mut zeros = 0usize;
    for a in sampled_firsts() {
        for &b in &partners {
            let (want, got) = (hadd(a, b), rank_sum(a, b));
            if want == got {
                continue;
            }
            assert!(
                a == 0x8000 && b == 0x8000,
                "{a:#06x} + {b:#06x}: 2-rank {want:?}, one-shot {got:?}"
            );
            zeros += 1;
        }
    }
    // 2026-10-16: The one difference is reached, so the comparison does see a sign of zero.
    assert_eq!(zeros, 1, "(-0) + (-0) is the one pair whose bytes differ");
}

#[test]
fn a_narrower_intermediate_does_move_bytes() {
    // 2026-10-16: The positive control at the bound: 17 bits (2p + 1) never moves a result, 16
    // does, so the comparison above is able to see a double-rounding error. The 16-bit witness:
    // 1 + 2^-7 plus 2^-8 - 2^-16 sits just under a tie, rounds up to the tie at 16 bits, and the
    // tie then goes to the even 1 + 2^-6 instead of down to 1 + 2^-7. The sampled first
    // terms keep the sweep quick; the exhaustive comparison is the ignored test below.
    assert_eq!(hadd(0x3f81, 0x3b7f), round(bf16_exact(0x3f81), 8));
    assert_ne!(twice(0x3f81, 0x3b7f, 16), hadd(0x3f81, 0x3b7f));
    assert_eq!(twice(0x3f81, 0x3b7f, 17), hadd(0x3f81, 0x3b7f));
    let partners = partners();
    let count = |mid: u32| {
        sampled_firsts()
            .into_iter()
            .map(|a| {
                partners
                    .iter()
                    .filter(|&&b| twice(a, b, mid) != hadd(a, b))
                    .count()
            })
            .sum::<usize>()
    };
    assert_eq!(count(17), 0, "17 bits is enough");
    assert!(count(16) > 0, "16 bits double-rounds some ties wrong");
    assert!(count(12) > 0, "12 bits double-rounds some ties wrong");
}

/// 2026-10-16: All 2^32 ordered pairs (finite ones) once in release:
/// `cargo test -p metrale-comm --release -- --ignored every_bf16_pair`.
#[test]
#[ignore = "exhaustive: about four billion pairs; run in release"]
fn every_bf16_pair_matches_except_two_negative_zeros() {
    let mut zeros = 0usize;
    for a in (0..=u16::MAX).filter(|&a| finite(a)) {
        for b in (0..=u16::MAX).filter(|&b| finite(b)) {
            if hadd(a, b) != rank_sum(a, b) {
                assert!(a == 0x8000 && b == 0x8000, "{a:#06x} + {b:#06x}");
                zeros += 1;
            }
        }
    }
    assert_eq!(zeros, 1);
}
