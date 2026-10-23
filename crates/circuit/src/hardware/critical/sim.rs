// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Discrete-event simulation of the event graph with two shared resources: SM
//! residency and DRAM bandwidth. Every non-kernel event follows the max-plus rule; a kernel,
//! once started, is resident (holds SM capacity, also while it waits on a programmatic
//! predecessor) and progresses at a rate set by what it is granted.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - SM capacity is fluid, in SM units: a kernel wants `min(ctas, sms * per_sm) / per_sm`
//!   ([`cap_of`]), where `per_sm` is how many of its CTAs fit one SM (registers at the 256-per-warp
//!   allocation, threads, shared memory, the CTA limit). Resident kernels are granted capacity
//!   oldest start first, as the block scheduler drains the older grid first.
//! - A progressing kernel's rate is its granted share of the capacity it wants, scaled down
//!   (for kernels that read DRAM) when the progressing kernels' demand exceeds the shared
//!   bandwidth: `bytes / solo work` each, work-conserving.
//! - A kernel's solo work is calibrated so that, on the measured schedule, the same rate law
//!   reproduces its measured in-context work ([`solo_work`]); the simulation's baseline error is
//!   therefore the error of its schedule, and a rewrite moves rates only through what it
//!   changes about co-residency and bandwidth.
//! - Events of nodes that started before the window (the context) keep their measured times.

use crate::prelude::*;

use alloc::collections::BinaryHeap;
use core::cmp::{Ordering, Reverse};

use super::graph::{EdgeKind, Graph, NodeKind, f_of, ix, node_of};

/// 2026-10-23: One SM's limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmLimits {
    /// 2026-10-23: 32-bit registers.
    pub regs: u32,
    /// 2026-10-23: Resident threads.
    pub threads: u32,
    /// 2026-10-23: Shared memory, bytes.
    pub smem: u32,
    /// 2026-10-23: Resident CTAs.
    pub ctas: u32,
}

/// 2026-10-23: The shared resources.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resources {
    /// 2026-10-23: SMs.
    pub sms: f64,
    /// 2026-10-23: Per SM.
    pub sm: SmLimits,
    /// 2026-10-23: Shared DRAM bandwidth, bytes per microsecond (GB/s x 1000).
    pub dram_bytes_per_us: f64,
    /// 2026-10-23: The device's datasheet DRAM bandwidth, bytes per microsecond: no kernel and
    /// no set of kernels reads faster.
    pub dram_peak_bytes_per_us: f64,
}

/// 2026-10-23: CTAs of a launch one SM holds at once (at least 1).
pub fn per_sm(threads: u32, regs: u32, smem: u32, l: &SmLimits) -> u32 {
    let warps = threads.div_ceil(32).max(1);
    let regs_cta = warps * (regs * 32).div_ceil(256) * 256;
    let by = |cap: u32, need: u32| if need == 0 { u32::MAX } else { cap / need };
    by(l.regs, regs_cta)
        .min(by(l.threads, threads))
        .min(by(l.smem, smem + if smem > 0 { 1024 } else { 0 }))
        .min(l.ctas)
        .max(1)
}

/// 2026-10-23: The SM capacity node `i` wants while resident, in SM units (0: not a kernel).
pub fn cap_of(g: &Graph, i: usize, r: &Resources) -> f64 {
    let n = &g.nodes[i];
    if n.kind != NodeKind::Kernel || n.ctas == 0 {
        return 0.0;
    }
    let p = f64::from(per_sm(n.threads, n.regs, n.smem, &r.sm));
    (f64::from(n.ctas)).min(r.sms * p) / p
}

#[derive(Clone, Copy, PartialEq)]
struct At(f64, u32);
impl Eq for At {}
impl PartialOrd for At {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for At {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

/// 2026-10-23: A resident kernel.
#[derive(Clone, Copy)]
struct Res {
    node: usize,
    since: f64,
    running: bool,
    left: f64,
    rate: f64,
}

/// 2026-10-23: Rates of `res` (ordered oldest first): granted capacity share, then the DRAM
/// scale over the progressing kernels' demand.
fn rate_law(res: &mut [Res], cap: &[f64], bw: &[f64], r: &Resources) {
    let mut free = r.sms;
    let mut demand = 0.0;
    for x in res.iter_mut() {
        let want = cap[x.node];
        let got = want.min(free.max(0.0));
        free -= got;
        x.rate = if !x.running {
            0.0
        } else if want > 0.0 {
            got / want
        } else {
            1.0
        };
        demand += x.rate * bw[x.node];
    }
    if demand > r.dram_bytes_per_us {
        let s = r.dram_bytes_per_us / demand;
        for x in res.iter_mut() {
            if bw[x.node] > 0.0 {
                x.rate *= s;
            }
        }
    }
}

/// 2026-10-23: Per node, DRAM bytes per microsecond of solo work.
fn bandwidth(bytes: &[f64], solo: &[f64]) -> Vec<f64> {
    bytes
        .iter()
        .zip(solo)
        .map(|(&b, &w)| if b > 0.0 && w > 1e-6 { b / w } else { 0.0 })
        .collect()
}

/// 2026-10-23: Solo work of every kernel: what the rate law integrates to over its measured
/// running interval (three fixed-point rounds, since a kernel's DRAM demand depends on it).
/// Non-kernel nodes keep `work`.
pub fn solo_work(g: &Graph, work: &[f64], bytes: &[f64], r: &Resources) -> Vec<f64> {
    let cap: Vec<f64> = (0..g.nodes.len()).map(|i| cap_of(g, i, r)).collect();
    let mut solo = work.to_vec();
    let kernels: Vec<usize> = (0..g.nodes.len())
        .filter(|&i| g.nodes[i].kind == NodeKind::Kernel)
        .collect();
    let mut marks: Vec<(f64, u8, usize)> = Vec::with_capacity(3 * kernels.len());
    for &i in &kernels {
        let (s, f) = (g.t[2 * i], g.t[2 * i + 1]);
        marks.push((s, 1, i));
        marks.push((f - work[i], 2, i));
        marks.push((f, 0, i));
    }
    marks.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for _ in 0..3 {
        let bw = bandwidth(bytes, &solo);
        let mut acc = vec![0.0; g.nodes.len()];
        let mut res: Vec<Res> = Vec::new();
        let mut now = f64::NEG_INFINITY;
        for &(t, kind, i) in &marks {
            if !res.is_empty() && t > now {
                rate_law(&mut res, &cap, &bw, r);
                for x in &res {
                    acc[x.node] += x.rate * (t - now);
                }
            }
            now = t;
            match kind {
                1 => {
                    let k = res.partition_point(|x| (x.since, x.node) < (t, i));
                    res.insert(
                        k,
                        Res {
                            node: i,
                            since: t,
                            running: work[i] <= 0.0,
                            left: 0.0,
                            rate: 0.0,
                        },
                    );
                }
                2 => {
                    if let Some(x) = res.iter_mut().find(|x| x.node == i) {
                        x.running = true;
                    }
                }
                _ => res.retain(|x| x.node != i),
            }
        }
        for &i in &kernels {
            solo[i] = if work[i] > 0.0 { acc[i].max(1e-3) } else { 0.0 };
        }
    }
    solo
}

/// 2026-10-23: Simulate: the time of every event. `w` are the max-plus weights (kernels'
/// `Duration`/`PdlWait` weights are replaced by progress), `solo` each kernel's solo work,
/// `bytes` each kernel's DRAM bytes, `cap` (optional) overrides of the capacity a node wants.
pub fn simulate(
    g: &Graph,
    w: &[f64],
    solo: &[f64],
    bytes: &[f64],
    cap_over: &[(usize, f64)],
    r: &Resources,
) -> Vec<f64> {
    let nn = g.nodes.len();
    let mut cap: Vec<f64> = (0..nn).map(|i| cap_of(g, i, r)).collect();
    for &(i, c) in cap_over {
        cap[i] = c;
    }
    let bw = bandwidth(bytes, solo);
    let is_kernel = |v: u32| g.nodes[node_of(v)].kind == NodeKind::Kernel;
    let fixed = |v: u32| g.t[(v & !1) as usize] < 0.0;
    let nv = g.t.len();
    let mut pend: Vec<u32> = (0..nv).map(|v| g.in_off[v + 1] - g.in_off[v]).collect();
    let mut lb = vec![f64::NEG_INFINITY; nv];
    let mut at = vec![f64::NAN; nv];
    let mut heap: BinaryHeap<Reverse<At>> = BinaryHeap::new();
    for v in 0..nv {
        if pend[v] == 0 {
            heap.push(Reverse(At(g.t[v], ix(v))));
        }
    }
    let mut res: Vec<Res> = Vec::new();
    let mut now = f64::NEG_INFINITY;
    loop {
        let next_fin = res
            .iter()
            .enumerate()
            .filter(|(_, x)| x.running && x.rate > 0.0)
            .map(|(k, x)| (now + x.left / x.rate, k))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let next_ev = heap.peek().map(|Reverse(a)| a.0);
        let (t, fin) = match (next_ev, next_fin) {
            (None, None) => break,
            (Some(e), Some((f, k))) if f < e => (f, Some(k)),
            (None, Some((f, k))) => (f, Some(k)),
            (Some(e), _) => (e, None),
        };
        let dt = (t - now).max(0.0);
        if dt > 0.0 && now.is_finite() {
            for x in res.iter_mut().filter(|x| x.running) {
                x.left = (x.left - x.rate * dt).max(0.0);
            }
        }
        now = now.max(t);
        let v = match fin {
            Some(k) => {
                let x = res.remove(k);
                f_of(x.node)
            }
            None => {
                let Some(Reverse(At(_, v))) = heap.pop() else {
                    break;
                };
                v
            }
        };
        at[v as usize] = if fixed(v) { g.t[v as usize] } else { now };
        let node = node_of(v);
        if v & 1 == 0 && is_kernel(v) && !fixed(v) {
            let k = res.partition_point(|x| (x.since, x.node) < (now, node));
            res.insert(
                k,
                Res {
                    node,
                    since: now,
                    running: false,
                    left: solo[node],
                    rate: 0.0,
                },
            );
        }
        if v & 1 == 1 && fixed(v) {
            res.retain(|x| x.node != node);
        }
        for &k in g.outgoing(v) {
            let e = g.edges[k as usize];
            let d = e.dst;
            let progress = d & 1 == 1 && is_kernel(d) && !fixed(d);
            if !progress || !matches!(e.kind, EdgeKind::Duration | EdgeKind::PdlWait) {
                lb[d as usize] = lb[d as usize].max(at[v as usize] + w[k as usize]);
            }
            pend[d as usize] -= 1;
            if pend[d as usize] > 0 {
                continue;
            }
            if progress {
                let n = node_of(d);
                match res.iter_mut().find(|x| x.node == n) {
                    Some(x) if x.left > 0.0 => x.running = true,
                    _ => {
                        res.retain(|x| x.node != n);
                        heap.push(Reverse(At(now.max(lb[d as usize]), d)));
                    }
                }
            } else {
                let when = if fixed(d) {
                    g.t[d as usize]
                } else {
                    lb[d as usize]
                };
                heap.push(Reverse(At(when, d)));
            }
        }
        rate_law(&mut res, &cap, &bw, r);
    }
    at
}

/// 2026-10-23: The shared-DRAM constant and what it was calibrated on.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DramCalibration {
    /// 2026-10-23: Shared bandwidth, bytes per microsecond.
    pub bytes_per_us: f64,
    /// 2026-10-23: Intervals with two or more byte-reading kernels running, and their total
    /// length, microseconds.
    pub overlapped_intervals: usize,
    /// 2026-10-23: Total overlapped time, microseconds.
    pub overlapped_us: f64,
    /// 2026-10-23: The highest in-context bandwidth of any single kernel (sanity: at or below
    /// the shared constant), bytes per microsecond.
    pub solo_max_bytes_per_us: f64,
    /// 2026-10-23: Kernels whose plan bytes over their in-context work exceeded the datasheet
    /// peak (floor bytes the kernel cannot have read from DRAM in that time: L2 hits or a floor
    /// that overstates them), clipped to the peak.
    pub clipped: usize,
    /// 2026-10-23: Kernels whose bytes were reduced so the measured schedule fits the shared
    /// constant (they ran beside readers whose floors together exceed it).
    pub scaled: usize,
    /// 2026-10-23: How.
    pub basis: String,
}

/// 2026-10-23: The quantile `q` of `(value, weight)` pairs, by weight.
fn weighted_quantile(mut v: Vec<(f64, f64)>, q: f64) -> f64 {
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = v.iter().map(|x| x.1).sum();
    let mut acc = 0.0;
    for (x, wt) in &v {
        acc += wt;
        if acc >= q * total {
            return *x;
        }
    }
    v.last().map_or(0.0, |x| x.0)
}

/// 2026-10-23: The measured schedule cut into maximal intervals during which the same
/// byte-reading kernels do their in-context work: `(length us, kernels)`.
fn intervals(g: &Graph, work: &[f64], bytes: &[f64]) -> Vec<(f64, Vec<usize>)> {
    let mut marks: Vec<(f64, bool, usize)> = Vec::new();
    for i in 0..g.nodes.len() {
        if g.nodes[i].kind != NodeKind::Kernel
            || bytes[i] <= 0.0
            || work[i] <= 1e-6
            || g.t[2 * i] < 0.0
        {
            continue;
        }
        let f = g.t[2 * i + 1];
        marks.push((f - work[i], true, i));
        marks.push((f, false, i));
    }
    marks.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut active: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    let mut now = f64::NEG_INFINITY;
    for (t, start, i) in marks {
        if !active.is_empty() && t > now {
            out.push((t - now, active.clone()));
        }
        now = t;
        if start {
            active.push(i);
        } else {
            active.retain(|&k| k != i);
        }
    }
    out
}

/// 2026-10-23: Calibrate the shared DRAM bandwidth on the capture's overlapped intervals: over
/// every interval in which two or more byte-reading kernels run (measured), their summed
/// in-context bandwidth (`bytes / work` each); the constant is the time-weighted 95th
/// percentile, the aggregate the GPU sustained when kernels shared DRAM. Then every kernel's
/// bytes are reduced to what the measured schedule could have moved at that constant: by the
/// smallest `constant / summed bandwidth` over its intervals (where the floors of the kernels
/// running together exceed the bus, the floor overstates their DRAM traffic, L2 served the
/// rest), so the measured schedule is feasible and the shared bus binds only where a rewrite
/// overlaps more DRAM traffic than the capture did. Returns the calibration and the bytes.
pub fn calibrate_dram(
    g: &Graph,
    work: &[f64],
    bytes: &[f64],
    peak: f64,
    clipped: usize,
) -> (DramCalibration, Vec<f64>) {
    let iv = intervals(g, work, bytes);
    let bw = |k: usize| bytes[k] / work[k];
    let solo_max = iv
        .iter()
        .flat_map(|(_, a)| a.iter().map(|&k| bw(k)))
        .fold(0.0, f64::max);
    let samples: Vec<(f64, f64)> = iv
        .iter()
        .filter(|(_, a)| a.len() >= 2)
        .map(|(len, a)| (a.iter().map(|&k| bw(k)).sum(), *len))
        .collect();
    let n = samples.len();
    let total: f64 = samples.iter().map(|x| x.1).sum();
    let b = if n == 0 {
        solo_max
    } else {
        weighted_quantile(samples, 0.95).max(solo_max)
    }
    .min(peak);
    let mut scale = vec![1.0_f64; bytes.len()];
    for (_, a) in &iv {
        let agg: f64 = a.iter().map(|&k| bw(k)).sum();
        if agg > b {
            for &k in a {
                scale[k] = scale[k].min(b / agg);
            }
        }
    }
    let scaled = scale.iter().filter(|&&x| x < 1.0).count();
    let out: Vec<f64> = bytes.iter().zip(&scale).map(|(&x, &s)| x * s).collect();
    let d = DramCalibration {
        bytes_per_us: b,
        overlapped_intervals: n,
        overlapped_us: total,
        solo_max_bytes_per_us: solo_max,
        clipped,
        scaled,
        basis: "time-weighted p95 of the summed in-context bandwidth (plan floor bytes / work) \
                over the capture's intervals with >= 2 byte-reading kernels running, raised to \
                the largest single kernel's, at most the datasheet peak"
            .into(),
    };
    (d, out)
}
