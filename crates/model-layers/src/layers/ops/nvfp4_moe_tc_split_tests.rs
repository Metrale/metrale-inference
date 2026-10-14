// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The split NVFP4 tensor-core MoE decode pair writes the unsplit pair's bytes:
//! `moe_expert_gate_up_act_nvfp4_grouped_tc_split` (a column tile's gate and up rows on two
//! warps) against `moe_expert_gate_up_act_nvfp4_grouped_tc` (act and shared act), and
//! `moe_expert_down_act_nvfp4_grouped_tc_mt1` (one m-tile per warp) against
//! `moe_expert_down_act_nvfp4_grouped_tc` on the same act (output and shared output). Shapes are
//! the Flash-Next experts (512 experts, top-10, hidden 2560, intermediate 640) at
//! {1, 2, 4, 8, 16, 32, 64, 128} tokens; three hot experts take a third of the tokens each, so
//! their rows run many 8-row passes (the split's double-buffered exchange), and one expert has a
//! null gate pointer and another a null down pointer (the zero-fill branches). The test also
//! prints both pairs' time per launch at each width.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release nvfp4_moe_tc_split_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{
    Fp8GroupedGeometry, NVFP4_GROUPED_DOWN_TC, NVFP4_GROUPED_DOWN_TC_MT1, NVFP4_GROUPED_GATE_UP_TC,
    NVFP4_GROUPED_GATE_UP_TC_SPLIT, Nvfp4ExpertTables, moe_expert_down_act_nvfp4_grouped,
    moe_expert_gate_up_act_nvfp4_grouped,
};
use crate::weight_map::QuantizedWeight;

const SENTINEL: u8 = 0xAB;
const EXPERTS: u32 = 512;
const TOP_K: u32 = 10;
const HIDDEN: u32 = 2560;
const INTER: u32 = 640;
/// 2026-10-14: Distinct weight buffers; expert e points at buffer e % SETS.
const SETS: u32 = 16;
/// 2026-10-14: The expert whose gate pointer and the expert whose down pointer are null.
const NULL_GATE: u32 = 5;
const NULL_DOWN: u32 = 9;
const WIDTHS: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const TIMED: u32 = 200;
const MODULE: &str = "moe_nvfp4_grouped_tc";

/// 2026-10-14: Deterministic bytes; scale bytes stay in E4M3's finite range (0.25 .. 0.94).
fn bytes(n: usize, seed: u32, scale: bool) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if scale {
                0x28 + (x % 24) as u8
            } else {
                x as u8
            }
        })
        .collect()
}

/// 2026-10-14: BF16 activations in [-2, 2].
fn bf16s(n: usize, seed: u32) -> Vec<u8> {
    bytes(n, seed, false)
        .into_iter()
        .flat_map(|b| {
            let v = b as f32 / 63.75 - 2.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

fn upload(g: &dyn GpuBackend, data: &[u8]) -> DevicePtr {
    let p = g.alloc(data.len()).unwrap();
    g.copy_h2d(data, p).unwrap();
    p
}

fn sentinel(g: &dyn GpuBackend, n: usize) -> DevicePtr {
    let p = g.alloc(n).unwrap();
    g.memset(p, SENTINEL, n).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

/// 2026-10-14: One projection's `[n, k]` weights: the routed tables (expert `null` points at
/// NULL) and the shared expert.
fn projection(
    g: &dyn GpuBackend,
    n: u32,
    k: u32,
    seed: u32,
    s2: f32,
    null: Option<u32>,
) -> (Nvfp4ExpertTables, QuantizedWeight) {
    let (n, k) = (n as usize, k as usize);
    let sets: Vec<(DevicePtr, DevicePtr)> = (0..=SETS)
        .map(|s| {
            (
                upload(g, &bytes(n * k / 2, seed + s * 7919, false)),
                upload(g, &bytes(n * k / 16, seed + s * 104_729, true)),
            )
        })
        .collect();
    let ptr = |e: u32, f: fn(&(DevicePtr, DevicePtr)) -> DevicePtr| {
        if Some(e) == null {
            0
        } else {
            f(&sets[(e % SETS) as usize]).0
        }
    };
    let packed: Vec<u64> = (0..EXPERTS).map(|e| ptr(e, |s| s.0)).collect();
    let scales: Vec<u64> = (0..EXPERTS).map(|e| ptr(e, |s| s.1)).collect();
    let s2s: Vec<f32> = (0..EXPERTS)
        .map(|e| s2 * (1.0 + (e % 7) as f32 / 8.0))
        .collect();
    let shared = sets[SETS as usize];
    let mut sh = QuantizedWeight::null();
    sh.weight = shared.0;
    sh.weight_scale = shared.1;
    sh.weight_scale_2 = s2 * 1.5;
    (
        Nvfp4ExpertTables {
            packed_ptrs: upload(g, &le(&packed, u64::to_le_bytes)),
            scale_ptrs: upload(g, &le(&scales, u64::to_le_bytes)),
            scale2_vals: upload(g, &le(&s2s, f32::to_le_bytes)),
        },
        sh,
    )
}

/// 2026-10-14: Token t's experts: hot expert t % 3, NULL_GATE / NULL_DOWN on every 7th / 11th
/// token, then distinct pseudo-random experts up to TOP_K.
fn picks(t: u32) -> Vec<u32> {
    let mut p = vec![t % 3];
    for (every, e) in [(7, NULL_GATE), (11, NULL_DOWN)] {
        if t.is_multiple_of(every) {
            p.push(e);
        }
    }
    let mut x = (t + 1).wrapping_mul(2_654_435_761) | 1;
    while p.len() < TOP_K as usize {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let e = x % EXPERTS;
        if !p.contains(&e) {
            p.push(e);
        }
    }
    p
}

/// 2026-10-14: The routing tables of `m` tokens on the device, and (positions, active experts).
struct Routing {
    offsets: DevicePtr,
    sorted: DevicePtr,
    active: DevicePtr,
    count: DevicePtr,
    positions: u32,
    cap: u32,
}

fn routing(g: &dyn GpuBackend, m: u32) -> Routing {
    let mut rows: Vec<Vec<i32>> = vec![Vec::new(); EXPERTS as usize];
    for t in 0..m {
        for e in picks(t) {
            rows[e as usize].push(t as i32);
        }
    }
    let mut offsets = vec![0i32];
    for r in &rows {
        offsets.push(offsets.last().unwrap() + r.len() as i32);
    }
    let sorted: Vec<i32> = rows.concat();
    let active: Vec<i32> = (0..EXPERTS as i32)
        .filter(|&e| !rows[e as usize].is_empty())
        .collect();
    Routing {
        offsets: upload(g, &le(&offsets, i32::to_le_bytes)),
        sorted: upload(g, &le(&sorted, i32::to_le_bytes)),
        active: upload(g, &le(&active, i32::to_le_bytes)),
        count: upload(g, &(active.len() as i32).to_le_bytes()),
        positions: sorted.len() as u32,
        cap: active.len() as u32,
    }
}

struct Weights {
    gate: (Nvfp4ExpertTables, QuantizedWeight),
    up: (Nvfp4ExpertTables, QuantizedWeight),
    down: (Nvfp4ExpertTables, QuantizedWeight),
}

/// 2026-10-14: One gate+up launch into sentinel-filled (act, shared act); their bytes.
fn gate_up(
    g: &dyn GpuBackend,
    (kernel, geometry): (KernelHandle, Fp8GroupedGeometry),
    w: &Weights,
    a: DevicePtr,
    r: &Routing,
    m: u32,
) -> (Vec<u8>, Vec<u8>) {
    let (na, ns) = ((r.positions * INTER * 4) as usize, (m * INTER * 4) as usize);
    let (act, sh_act) = (sentinel(g, na), sentinel(g, ns));
    moe_expert_gate_up_act_nvfp4_grouped(
        g, kernel, geometry, a, w.gate.0, w.up.0, act, r.offsets, r.sorted, r.active, r.count,
        &w.gate.1, &w.up.1, sh_act, INTER, HIDDEN, r.cap, m, 0,
    )
    .unwrap();
    g.synchronize(0).unwrap();
    let out = (download(g, act, na), download(g, sh_act, ns));
    g.free(act).unwrap();
    g.free(sh_act).unwrap();
    out
}

/// 2026-10-14: One down launch over (act, shared act) into sentinel-filled (C, shared out).
fn down(
    g: &dyn GpuBackend,
    (kernel, geometry): (KernelHandle, Fp8GroupedGeometry),
    w: &Weights,
    (act, sh_act): (DevicePtr, DevicePtr),
    r: &Routing,
    m: u32,
) -> (Vec<u8>, Vec<u8>) {
    let (nc, ns) = (
        (r.positions * HIDDEN * 2) as usize,
        (m * HIDDEN * 2) as usize,
    );
    let (c, sh_out) = (sentinel(g, nc), sentinel(g, ns));
    moe_expert_down_act_nvfp4_grouped(
        g, kernel, geometry, act, w.down.0, c, r.offsets, r.active, r.count, sh_act, &w.down.1,
        sh_out, HIDDEN, INTER, r.cap, m, 0,
    )
    .unwrap();
    g.synchronize(0).unwrap();
    let out = (download(g, c, nc), download(g, sh_out, ns));
    g.free(c).unwrap();
    g.free(sh_out).unwrap();
    out
}

/// 2026-10-14: Every `row`-byte row of `out` was written (not all sentinel).
fn all_rows_written(out: &[u8], row: usize, label: &str) {
    for (i, r) in out.chunks_exact(row).enumerate() {
        assert!(
            r.iter().any(|&b| b != SENTINEL),
            "{label}: row {i} unwritten"
        );
    }
}

/// 2026-10-14: Microseconds per launch of `f` over TIMED launches.
fn time_us(g: &dyn GpuBackend, mut f: impl FnMut()) -> f64 {
    f();
    g.synchronize(0).unwrap();
    let t = std::time::Instant::now();
    for _ in 0..TIMED {
        f();
    }
    g.synchronize(0).unwrap();
    t.elapsed().as_secs_f64() * 1e6 / TIMED as f64
}

#[test]
#[ignore]
fn nvfp4_tc_split_pair_writes_the_unsplit_pairs_bytes() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let k = |name: &str| g.kernel(MODULE, name).unwrap();
    let gu_base = (
        k("moe_expert_gate_up_act_nvfp4_grouped_tc"),
        NVFP4_GROUPED_GATE_UP_TC,
    );
    let gu_split = (
        k("moe_expert_gate_up_act_nvfp4_grouped_tc_split"),
        NVFP4_GROUPED_GATE_UP_TC_SPLIT,
    );
    let dn_base = (
        k("moe_expert_down_act_nvfp4_grouped_tc"),
        NVFP4_GROUPED_DOWN_TC,
    );
    let dn_mt1 = (
        k("moe_expert_down_act_nvfp4_grouped_tc_mt1"),
        NVFP4_GROUPED_DOWN_TC_MT1,
    );
    assert_ne!(gu_base.0.0, gu_split.0.0);
    assert_ne!(dn_base.0.0, dn_mt1.0.0);
    let w = Weights {
        gate: projection(g, INTER, HIDDEN, 11, 0.02, Some(NULL_GATE)),
        up: projection(g, INTER, HIDDEN, 13, 0.02, None),
        down: projection(g, HIDDEN, INTER, 17, 0.01, Some(NULL_DOWN)),
    };
    for m in WIDTHS {
        let r = routing(g, m);
        let a = upload(g, &bf16s((m * HIDDEN) as usize, 23 + m));
        let want = gate_up(g, gu_base, &w, a, &r, m);
        let got = gate_up(g, gu_split, &w, a, &r, m);
        let label = format!("m={m} positions={} active={}", r.positions, r.cap);
        all_rows_written(&want.0, (INTER * 4) as usize, &format!("{label} act"));
        all_rows_written(
            &want.1,
            (INTER * 4) as usize,
            &format!("{label} shared act"),
        );
        assert_eq!(got.0, want.0, "{label}: act");
        assert_eq!(got.1, want.1, "{label}: shared act");
        let act = (upload(g, &want.0), upload(g, &want.1));
        let want = down(g, dn_base, &w, act, &r, m);
        let got = down(g, dn_mt1, &w, act, &r, m);
        all_rows_written(&want.0, (HIDDEN * 2) as usize, &format!("{label} down"));
        all_rows_written(
            &want.1,
            (HIDDEN * 2) as usize,
            &format!("{label} shared down"),
        );
        assert_eq!(got.0, want.0, "{label}: down");
        assert_eq!(got.1, want.1, "{label}: shared down");
        let (na, ns) = ((r.positions * INTER * 4) as usize, (m * INTER * 4) as usize);
        let scratch = (g.alloc(na).unwrap(), g.alloc(ns).unwrap());
        let (nc, nd) = (
            (r.positions * HIDDEN * 2) as usize,
            (m * HIDDEN * 2) as usize,
        );
        let out = (g.alloc(nc).unwrap(), g.alloc(nd).unwrap());
        let gu_us = |(kernel, geometry): (KernelHandle, Fp8GroupedGeometry)| {
            time_us(g, || {
                moe_expert_gate_up_act_nvfp4_grouped(
                    g, kernel, geometry, a, w.gate.0, w.up.0, scratch.0, r.offsets, r.sorted,
                    r.active, r.count, &w.gate.1, &w.up.1, scratch.1, INTER, HIDDEN, r.cap, m, 0,
                )
                .unwrap()
            })
        };
        let dn_us = |(kernel, geometry): (KernelHandle, Fp8GroupedGeometry)| {
            time_us(g, || {
                moe_expert_down_act_nvfp4_grouped(
                    g, kernel, geometry, act.0, w.down.0, out.0, r.offsets, r.active, r.count,
                    act.1, &w.down.1, out.1, HIDDEN, INTER, r.cap, m, 0,
                )
                .unwrap()
            })
        };
        println!(
            "SPLIT_US m={m} gate_up base={:.2} split={:.2} down base={:.2} mt1={:.2}",
            gu_us(gu_base),
            gu_us(gu_split),
            dn_us(dn_base),
            dn_us(dn_mt1)
        );
        for p in [
            a, act.0, act.1, scratch.0, scratch.1, out.0, out.1, r.offsets, r.sorted, r.active,
            r.count,
        ] {
            g.free(p).unwrap();
        }
    }
}
