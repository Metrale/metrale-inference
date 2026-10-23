// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The normal and Student t distributions: CDFs and quantiles. The t CDF is the
//! regularized incomplete beta function (Lentz continued fraction, Numerical Recipes §6.4), so
//! non-integer degrees of freedom (Satterthwaite, Welch) are exact; the quantiles invert the CDFs
//! (Acklam's rational start refined by Newton for the normal, bracketed bisection for t).
//!
//! Owner: metrale-stats.
//! Invariants: `df = f64::INFINITY` is the normal distribution; quantiles of p outside (0, 1)
//! are ∓∞ at the ends and NaN beyond them.

const SQRT2: f64 = core::f64::consts::SQRT_2;

/// 2026-10-22: Φ(x).
#[must_use]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT2)
}

/// 2026-10-22: Φ⁻¹(p): Acklam's approximation (rel. error 1.15e-9) and one Halley step.
#[must_use]
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let lo = 0.02425;
    let x = if p < lo {
        let q = libm::sqrt(-2.0 * libm::log(p));
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - lo {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = libm::sqrt(-2.0 * libm::log(1.0 - p));
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let e = norm_cdf(x) - p;
    let u = e * libm::sqrt(2.0 * core::f64::consts::PI) * libm::exp(x * x / 2.0);
    x - u / (1.0 + x * u / 2.0)
}

/// 2026-10-22: The continued fraction of the incomplete beta (modified Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    let mut m = 1.0;
    while m < 400.0 {
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
        m += 1.0;
    }
    h
}

/// 2026-10-22: The regularized incomplete beta I_x(a, b).
#[must_use]
pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b)
        + a * libm::log(x)
        + b * libm::log1p(-x);
    let front = libm::exp(ln_front);
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// 2026-10-22: The Student t CDF with `df` degrees of freedom (`INFINITY`: normal).
#[must_use]
pub fn t_cdf(x: f64, df: f64) -> f64 {
    if x.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if df.is_infinite() || df > 1e7 {
        return norm_cdf(x);
    }
    let tail = 0.5 * inc_beta(df / 2.0, 0.5, df / (df + x * x));
    if x > 0.0 { 1.0 - tail } else { tail }
}

/// 2026-10-22: The Student t quantile: bisection on [`t_cdf`] over a bracket grown from the
/// normal quantile, to 1e-12 relative.
#[must_use]
pub fn t_ppf(p: f64, df: f64) -> f64 {
    if df.is_infinite() || df > 1e7 {
        return norm_ppf(p);
    }
    if p.is_nan() || df.is_nan() || df <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p < 0.5 {
        return -t_ppf(1.0 - p, df);
    }
    let mut lo = 0.0;
    let mut hi = norm_ppf(p).max(1.0);
    while t_cdf(hi, df) < p {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * hi.abs().max(1e-300) {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
#[path = "dist_tests.rs"]
mod tests;
