// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The go rule of the P-WD bench (`pwd.rs`): the serve measurements it is read
//! against, the rungs the wide verify must win, and the `SUMMARY` / `VERDICT` lines.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

/// 2026-10-15: The C8 MTP serve's 16-row union sweep per layer, gate+up 1050.1 us plus down
/// 519.5 us, at a measured union of about 80 experts.
const T16_SERVE_US: f64 = 1050.1 + 519.5;
/// 2026-10-15: The default `GLM_PWD_U16` the fitted models match: the serve's measured U(16).
pub const U16_SERVE: f64 = 80.0;
/// 2026-10-15: Largest relative t(16) difference from the serve at which the absolute
/// thresholds still read; past it only the ratio form does.
const FIDELITY: f64 = 0.10;
const MOE_LAYERS: f64 = 42.0;
/// 2026-10-15: Sustained DRAM bandwidth of the union sweeps on GB10, GB/s.
const DRAM_GBPS: f64 = 249.0;
const U32_PLAN_MAX: f64 = 135.0;

/// 2026-10-15: A rung the wide verify must win: `c` sequences of two rows, tokens per step,
/// the projected serial-path rate with the same levers, and the non-MoE step time at width.
struct Rung {
    c: usize,
    tokens_per_step: f64,
    serial_tok_s: f64,
    non_moe_ms: f64,
}

const RUNGS: [Rung; 2] = [
    Rung {
        c: 12,
        tokens_per_step: 19.44,
        serial_tok_s: 143.7,
        non_moe_ms: 37.9,
    },
    Rung {
        c: 16,
        tokens_per_step: 25.97,
        serial_tok_s: 161.0,
        non_moe_ms: 44.6,
    },
];

impl Rung {
    /// 2026-10-15: The per-layer MoE time (us) at which this rung's step only ties the serial
    /// rate: `(tokens / serial_tok_s - non_moe) / layers`.
    fn budget_us(&self) -> f64 {
        (self.tokens_per_step / self.serial_tok_s * 1e3 - self.non_moe_ms) / MOE_LAYERS * 1e3
    }
    /// 2026-10-15: The union that streams for [`Self::budget_us`] at [`DRAM_GBPS`].
    fn union_max(&self, expert_bytes: usize) -> f64 {
        self.budget_us() * DRAM_GBPS * 1e3 / expert_bytes as f64
    }
}

/// 2026-10-15: Per draw: `u`, the union sizes of rows 0..16, 16..32, 0..24 and 0..32; per
/// candidate (16, 24, 32, 2x16 rows) the median us of gate+up, down and the whole pipeline.
pub struct Draw {
    pub u: [usize; 4],
    pub gu: [f64; 4],
    pub dn: [f64; 4],
    pub full: [f64; 4],
}

const NAMES: [&str; 4] = ["16", "24", "32", "2x16"];

pub fn row(v: &[f64; 4]) -> String {
    NAMES
        .iter()
        .zip(v)
        .map(|(n, t)| format!("{n}={t:.1}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// 2026-10-15: The `SUMMARY` and `VERDICT` lines of one routing model; returns whether the
/// plan rule, the C12 and the C16 break-even rules pass.
pub fn verdict(name: &str, draws: &[Draw], expert_bytes: usize) -> [bool; 3] {
    let n = draws.len() as f64;
    let mean = |f: &dyn Fn(&Draw) -> f64| draws.iter().map(f).sum::<f64>() / n;
    let u = |i: usize| mean(&|d| d.u[i] as f64);
    let t = |i: usize| mean(&|d| d.gu[i] + d.dn[i]);
    let (u16, u24, u32_) = (u(0), u(2), u(3));
    let (t16, t24, t32, t2) = (t(0), t(1), t(2), t(3));
    println!(
        "SUMMARY route={name} draws={} U16={u16:.1} U24={u24:.1} U32={u32_:.1} \
         U2x16={:.1} | sweep_us 16={t16:.1} 24={t24:.1} 32={t32:.1} 2x16={t2:.1} | full_us 16={:.1} \
         24={:.1} 32={:.1} 2x16={:.1} | uniform_U 16={:.1} 32={:.1}",
        draws.len(),
        u(0) + u(1),
        mean(&|d| d.full[0]),
        mean(&|d| d.full[1]),
        mean(&|d| d.full[2]),
        mean(&|d| d.full[3]),
        super::pwd_route::uniform_union(16),
        super::pwd_route::uniform_union(32),
    );
    let pf = |b: bool| if b { "PASS" } else { "FAIL" };
    let fid = t16 / T16_SERVE_US - 1.0;
    let plan = u32_ <= U32_PLAN_MAX && t32 < t2;
    println!(
        "VERDICT route={name} fidelity t16/{T16_SERVE_US:.1}us = {:+.1}% ({}), U16 {u16:.1} vs \
         {U16_SERVE}",
        fid * 100.0,
        if fid.abs() <= FIDELITY {
            "absolute and ratio thresholds read"
        } else {
            "ratio form only"
        }
    );
    println!(
        "VERDICT route={name} plan: U32 {u32_:.1} <= {U32_PLAN_MAX} {} and t32 {t32:.1} < t2x16 \
         {t2:.1} {} => {}",
        pf(u32_ <= U32_PLAN_MAX),
        pf(t32 < t2),
        if plan { "GO" } else { "NO-GO" }
    );
    let mut out = [plan, false, false];
    for (i, (r, (tw, uw))) in RUNGS.iter().zip([(t24, u24), (t32, u32_)]).enumerate() {
        let ratio = r.budget_us() / T16_SERVE_US;
        // 2026-10-15: Time decides; U against the bandwidth model is printed as a cross-check.
        // At C16 the wide launch must also beat the serial 2x16 path it replaces.
        let ok = tw <= t16 * ratio && (r.c != 16 || t32 < t2);
        println!(
            "VERDICT route={name} C{} break-even: budget ({:.2}/{:.1}*1e3 - {:.1}) / {MOE_LAYERS} = \
             {:.1} us/layer = {ratio:.3} x t16_serve; t{} {tw:.1} <= t16 {t16:.1} x {ratio:.3} = \
             {:.1} {}; U {uw:.1} <= {:.1} (bandwidth model) {} => {}",
            r.c,
            r.tokens_per_step,
            r.serial_tok_s,
            r.non_moe_ms,
            r.budget_us(),
            2 * r.c,
            t16 * ratio,
            pf(tw <= t16 * ratio),
            r.union_max(expert_bytes),
            pf(uw <= r.union_max(expert_bytes)),
            if ok { "GO" } else { "NO-GO" }
        );
        out[i + 1] = ok;
    }
    out
}
