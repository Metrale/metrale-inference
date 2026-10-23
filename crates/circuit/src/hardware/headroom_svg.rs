// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Two SVG figures of a headroom (`met circuit headroom --format svg-roofline` /
//! `svg-waterfall`): the roofline (arithmetic intensity against achieved throughput per launch
//! site, under the class's bandwidth and per-MMA-class compute ceilings) and the step waterfall
//! (measured step, minus the time no kernel runs, Σ kernels, minus the time above the floors,
//! the step ceiling).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: deterministic text; colors are CSS custom properties with a light and a dark set
//! (`prefers-color-scheme` and a `data-theme` override); every mark carries a `<title>` tooltip;
//! a projection (unmeasured site) is a hollow mark, never drawn as a measurement.

use crate::prelude::*;

use core::fmt::Write as _;

use super::headroom::Headroom;

const W: f64 = 960.0;
const H: f64 = 600.0;
const ML: f64 = 80.0;
const MR: f64 = 200.0;
const MT: f64 = 56.0;
const MB: f64 = 84.0;

const STYLE: &str = r#"<style>
.viz text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px}
.viz .t{font-size:15px;font-weight:600}
.viz .g{stroke-width:1}.viz .c{stroke-width:2;fill:none;stroke-dasharray:6 4}
.viz .m{stroke-width:2}.viz .p{stroke-width:2}.viz .d{stroke-width:2;opacity:.45}
.viz .bg{fill:#fcfcfb}.viz text{fill:#52514e}.viz .t{fill:#0b0b0b}.viz .g{stroke:#e4e3df}.viz .c{stroke:#52514e}.viz .m{fill:#2a78d6;stroke:#fcfcfb}.viz .p{fill:#fcfcfb;stroke:#eb6834}.viz .d{stroke:#2a78d6}.viz .b1{fill:#2a78d6}.viz .b2{fill:#eb6834}.viz .b3{fill:#1baf7a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .viz .bg{fill:#1a1a19}:root:not([data-theme="light"]) .viz text{fill:#c3c2b7}:root:not([data-theme="light"]) .viz .t{fill:#f0efec}:root:not([data-theme="light"]) .viz .g{stroke:#3a3936}:root:not([data-theme="light"]) .viz .c{stroke:#c3c2b7}:root:not([data-theme="light"]) .viz .m{fill:#3987e5;stroke:#1a1a19}:root:not([data-theme="light"]) .viz .p{fill:#1a1a19;stroke:#d95926}:root:not([data-theme="light"]) .viz .d{stroke:#3987e5}:root:not([data-theme="light"]) .viz .b1{fill:#3987e5}:root:not([data-theme="light"]) .viz .b2{fill:#d95926}:root:not([data-theme="light"]) .viz .b3{fill:#199e70}}
[data-theme="dark"] .viz .bg{fill:#1a1a19}[data-theme="dark"] .viz text{fill:#c3c2b7}[data-theme="dark"] .viz .t{fill:#f0efec}[data-theme="dark"] .viz .g{stroke:#3a3936}[data-theme="dark"] .viz .c{stroke:#c3c2b7}[data-theme="dark"] .viz .m{fill:#3987e5;stroke:#1a1a19}[data-theme="dark"] .viz .p{fill:#1a1a19;stroke:#d95926}[data-theme="dark"] .viz .d{stroke:#3987e5}[data-theme="dark"] .viz .b1{fill:#3987e5}[data-theme="dark"] .viz .b2{fill:#d95926}[data-theme="dark"] .viz .b3{fill:#199e70}
</style>"#;

fn head(title: &str, desc: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" class=\"viz\" viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\" role=\"img\" aria-label=\"{}\">\n<title>{}</title>\n<desc>{}</desc>\n{STYLE}\n<rect class=\"bg\" width=\"{W}\" height=\"{H}\"/>\n",
        esc(title),
        esc(title),
        esc(desc)
    )
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn log10(x: f64) -> f64 {
    libm::log10(x)
}

fn decade(x: f64, up: bool) -> f64 {
    let l = log10(x);
    libm::pow(10.0, if up { libm::ceil(l) } else { libm::floor(l) })
}

fn sci(x: f64) -> String {
    if x >= 1.0 {
        format!("{x:.0}")
    } else {
        format!("{x}")
    }
}

/// 2026-10-22: Peak compute ceilings, TFLOPS, by MMA class, and DRAM bandwidth, GB/s.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Ceilings {
    /// 2026-10-22: GB/s.
    pub dram_gbps: f64,
    /// 2026-10-22: BF16 TFLOPS.
    pub bf16_tflops: f64,
    /// 2026-10-22: FP8 TFLOPS.
    pub fp8_tflops: f64,
    /// 2026-10-22: NVFP4 TFLOPS.
    pub nvfp4_tflops: f64,
}

/// 2026-10-22: The roofline figure.
pub fn roofline_svg(h: &Headroom) -> String {
    let c = h.ceilings;
    let peaks = [
        ("NVFP4", c.nvfp4_tflops),
        ("FP8", c.fp8_tflops),
        ("BF16", c.bf16_tflops),
    ];
    let pts: Vec<_> = h
        .sites
        .iter()
        .filter(|r| r.flops > 0.0 && r.bytes > 0.0 && r.floor_us > 0.0)
        .map(|r| {
            let ai = r.flops / r.bytes;
            let at_floor = r.flops / r.floor_us / 1e6;
            let got = r
                .measured
                .as_ref()
                .filter(|m| m.total_us > 0.0)
                .map(|m| r.flops / m.total_us / 1e6);
            (r, ai, at_floor, got)
        })
        .collect();
    let skipped = h.sites.len() - pts.len();
    let bw_tf = c.dram_gbps / 1e3;
    let max_peak = peaks.iter().map(|p| p.1).fold(0.0, f64::max);
    let (mut xmin, mut xmax, mut ymin) = (1.0f64, max_peak / bw_tf * 4.0, max_peak / 1e3);
    for (_, ai, f, g) in &pts {
        xmin = xmin.min(*ai);
        xmax = xmax.max(*ai);
        ymin = ymin.min(g.unwrap_or(*f)).min(*f);
    }
    let (xmin, xmax) = (decade(xmin, false), decade(xmax, true));
    let (ymin, ymax) = (decade(ymin, false), decade(max_peak, true));
    let pw = W - ML - MR;
    let ph = H - MT - MB;
    let x = |v: f64| ML + pw * (log10(v) - log10(xmin)) / (log10(xmax) - log10(xmin));
    let y = |v: f64| MT + ph * (1.0 - (log10(v) - log10(ymin)) / (log10(ymax) - log10(ymin)));
    let mut s = head(
        &format!("Roofline: {} on {}", h.model, h.device),
        "Arithmetic intensity (FLOP/byte) against throughput (TFLOP/s), log-log. Filled: measured; hollow: unmeasured projection at the floor; the faint line joins a measured site to its floor.",
    );
    let _ = writeln!(
        s,
        "<text class=\"t\" x=\"{ML}\" y=\"28\">Roofline · {} on {} · {}</text>",
        esc(&h.model),
        esc(&h.device),
        esc(&h.runs.join(" + "))
    );
    let mut d = xmin;
    while d <= xmax * 1.0001 {
        let _ = writeln!(
            s,
            "<line class=\"g\" x1=\"{0:.1}\" y1=\"{MT}\" x2=\"{0:.1}\" y2=\"{1:.1}\"/><text x=\"{0:.1}\" y=\"{2:.1}\" text-anchor=\"middle\">{3}</text>",
            x(d),
            MT + ph,
            MT + ph + 18.0,
            sci(d)
        );
        d *= 10.0;
    }
    let mut d = ymin;
    while d <= ymax * 1.0001 {
        let _ = writeln!(
            s,
            "<line class=\"g\" x1=\"{ML}\" y1=\"{0:.1}\" x2=\"{1:.1}\" y2=\"{0:.1}\"/><text x=\"{2:.1}\" y=\"{3:.1}\" text-anchor=\"end\">{4}</text>",
            y(d),
            ML + pw,
            ML - 8.0,
            y(d) + 4.0,
            sci(d)
        );
        d *= 10.0;
    }
    let _ = writeln!(
        s,
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">arithmetic intensity (FLOP/byte)</text>\n<text transform=\"translate(22 {:.1}) rotate(-90)\" text-anchor=\"middle\">throughput (TFLOP/s)</text>",
        ML + pw / 2.0,
        MT + ph + 40.0,
        MT + ph / 2.0
    );
    for (name, peak) in peaks {
        let ridge = peak / bw_tf;
        let x0 = xmin.max(ymin / bw_tf);
        let _ = writeln!(
            s,
            "<polyline class=\"c\" points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\"><title>{name} ceiling {peak} TFLOP/s; DRAM {} GB/s; ridge {ridge:.0} FLOP/byte</title></polyline><text x=\"{:.1}\" y=\"{:.1}\">{name} {peak} TFLOP/s</text>",
            x(x0),
            y(x0 * bw_tf),
            x(ridge),
            y(peak),
            x(xmax),
            y(peak),
            c.dram_gbps,
            ML + pw + 6.0,
            y(peak) + 4.0
        );
    }
    let _ = writeln!(
        s,
        "<text x=\"{:.1}\" y=\"{:.1}\" transform=\"rotate(-38 {:.1} {:.1})\">DRAM {} GB/s</text>",
        x(xmin * 3.0),
        y(xmin * 3.0 * bw_tf) - 8.0,
        x(xmin * 3.0),
        y(xmin * 3.0 * bw_tf) - 8.0,
        c.dram_gbps
    );
    for (r, ai, f, g) in &pts {
        let tip = |what: &str| {
            format!(
                "{} · {} · {what} · AI {:.2} FLOP/byte · floor {:.2} us ({})",
                r.run,
                r.site,
                ai,
                r.floor_us,
                r.bound.name()
            )
        };
        match g {
            Some(g) => {
                let _ = writeln!(
                    s,
                    "<line class=\"d\" x1=\"{0:.1}\" y1=\"{1:.1}\" x2=\"{0:.1}\" y2=\"{2:.1}\"/><circle class=\"m\" cx=\"{0:.1}\" cy=\"{1:.1}\" r=\"5\"><title>{3}</title></circle>",
                    x(*ai),
                    y(*g),
                    y(*f),
                    esc(&tip(&format!(
                        "measured {:.3} TFLOP/s, {:.1}% of floor",
                        g,
                        r.pct_of_floor.unwrap_or(0.0)
                    )))
                );
            }
            None => {
                let _ = writeln!(
                    s,
                    "<circle class=\"p\" cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\"><title>{}</title></circle>",
                    x(*ai),
                    y(*f),
                    esc(&tip("unmeasured: projection at the floor"))
                );
            }
        }
    }
    for (r, ai, _, g) in pts
        .iter()
        .filter(|p| p.3.is_some() && p.0.upside_full.is_some_and(|u| u.saved_us > 0.0))
        .take(5)
    {
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\">{}</text>",
            x(*ai) + 8.0,
            y(g.unwrap_or(1.0)) + 4.0,
            esc(&r.site)
        );
    }
    let ly = H - 16.0;
    let _ = writeln!(
        s,
        "<circle class=\"m\" cx=\"{0:.1}\" cy=\"{ly:.1}\" r=\"5\"/><text x=\"{1:.1}\" y=\"{2:.1}\">measured</text><circle class=\"p\" cx=\"{3:.1}\" cy=\"{ly:.1}\" r=\"4\"/><text x=\"{4:.1}\" y=\"{2:.1}\">unmeasured (projection at floor)</text><text x=\"{5:.1}\" y=\"{2:.1}\">{6} site(s) with no FLOPs not drawn</text>",
        ML + 6.0,
        ML + 16.0,
        ly + 4.0,
        ML + 106.0,
        ML + 116.0,
        ML + 340.0,
        skipped
    );
    s.push_str("</svg>\n");
    s
}

/// 2026-10-22: The step waterfall; without a capture it shows the ceiling alone.
pub fn waterfall_svg(h: &Headroom) -> String {
    let t = &h.totals;
    let mut bars: Vec<(String, f64, f64, &str, String)> = Vec::new();
    if let (Some(step), Some(res), Some(out), Some(sites)) = (
        t.step_us,
        t.residual_us,
        t.outside_exclusive_us,
        t.sites_exclusive_us,
    ) {
        let kernels = sites + out;
        bars.push(("measured step".into(), 0.0, step, "b1", format!("{:.0} us, {:.1} tok/s", step, t.step_tok_s.unwrap_or(0.0))));
        bars.push(("− no kernel running".into(), kernels, step, "b2", format!("residual {:.0} us (host, syncs, waits, gaps)", res)));
        bars.push(("= Σ kernels, exclusive".into(), 0.0, kernels, "b1", format!("{:.0} us: launch sites {:.0} + outside the plan {:.0}", kernels, sites, out)));
        let lo = t.ceiling_us.min(kernels);
        bars.push(("− above the floors".into(), lo, kernels, "b2", format!("{:.0} us (outside-the-plan kernels have no floor)", kernels - lo)));
    }
    bars.push((
        "= step ceiling: Σ floors".into(),
        0.0,
        t.ceiling_us,
        "b3",
        format!(
            "{:.0} us, {:.1} tok/s; unmeasured sites' projections {:.0} us",
            t.ceiling_us, t.ceiling_tok_s, t.unmeasured_projection_us
        ),
    ));
    let max = bars.iter().map(|b| b.2).fold(1.0, f64::max);
    let mut s = head(
        &format!("Step waterfall: {} on {}", h.model, h.device),
        "Microseconds per step: the measured step, the time no kernel runs, the kernels' exclusive time, the time above the floors, and the step ceiling.",
    );
    let _ = writeln!(
        s,
        "<text class=\"t\" x=\"{ML}\" y=\"28\">Step waterfall · {} on {} · {}</text>",
        esc(&h.model),
        esc(&h.device),
        esc(&h.runs.join(" + "))
    );
    let left = 200.0;
    let pw = W - left - 40.0;
    let x = |v: f64| left + pw * v / max;
    let row = (H - MT - MB) / bars.len() as f64;
    for (i, (name, a, b, cls, tip)) in bars.iter().enumerate() {
        let y0 = MT + row * i as f64 + row * 0.2;
        let hgt = row * 0.6;
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text><rect class=\"{cls}\" x=\"{:.1}\" y=\"{y0:.1}\" width=\"{:.1}\" height=\"{hgt:.1}\" rx=\"4\"><title>{}</title></rect><text x=\"{:.1}\" y=\"{:.1}\">{}</text>",
            left - 10.0,
            y0 + hgt / 2.0 + 4.0,
            esc(name),
            x(*a),
            (x(*b) - x(*a)).max(1.0),
            esc(tip),
            (x(*b) + 6.0).min(W - 160.0),
            y0 - 4.0,
            esc(&format!("{:.2} ms", (b - a) / 1e3))
        );
    }
    s.push_str("</svg>\n");
    s
}
