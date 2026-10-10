// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Device setup and launch helpers of the `glm5next_moe_wide_bench` example: the
//! per-rank GLM-5.3 routed-expert shapes, seeded weights and routings, the host port of the
//! serial union builder, graph-replay timing and one launcher per kernel under test.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_mlp::weights::Nvfp4Proj;

pub const HIDDEN: usize = 4096;
pub const MI: usize = 2048;
pub const EXPERTS: usize = 288;
pub const LOCAL: usize = 96;
pub const TOP_K: usize = 8;
pub const MAX_ROWS: usize = 16;
/// 2026-10-09: Replays per candidate and launches per replay; `GLM_BENCH_REPS` and
/// `GLM_BENCH_ITERS` override them (many one-launch replays let the low percentiles show the
/// uncontended time while another process time-slices the GPU).
pub const REPS: usize = 9;
pub const ITERS: usize = 20;

pub fn env_or(name: &str, v: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(v)
}
pub const GS_GATE_UP: f32 = 3.0 / 2688.0;
pub const GS_DOWN: f32 = 100.0 / 2688.0;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

pub fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

pub fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    up(gpu, &b)
}

pub fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub fn read_i32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    Ok(read(gpu, p, n * 4)?
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

pub fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// 2026-10-09: Median, 10th percentile and max microseconds per launch of each of `fs`: each closure
/// (`ITERS` launches) captured once into a graph, one warm replay each, then `REPS` rounds that
/// replay every graph once in turn, so that a load another process puts on the GPU falls on
/// every candidate alike.
pub fn time_set(
    gpu: &dyn GpuBackend,
    stream: u64,
    fs: &[&dyn Fn() -> Result<()>],
) -> Result<Vec<[f64; 3]>> {
    let (reps, iters) = (
        env_or("GLM_BENCH_REPS", REPS),
        env_or("GLM_BENCH_ITERS", ITERS),
    );
    let mut graphs = Vec::with_capacity(fs.len());
    for f in fs {
        gpu.begin_capture(stream)?;
        for _ in 0..iters {
            if let Err(e) = f() {
                gpu.abort_capture_if_active(stream);
                return Err(e);
            }
        }
        let graph = gpu.end_capture(stream)?;
        ensure!(graph.0 != 0, "graph capture returned no graph");
        gpu.launch_graph(graph, stream)?;
        graphs.push(graph);
    }
    gpu.synchronize(stream)?;
    let mut v = vec![Vec::with_capacity(reps); fs.len()];
    for _ in 0..reps {
        for (i, &graph) in graphs.iter().enumerate() {
            let t = std::time::Instant::now();
            gpu.launch_graph(graph, stream)?;
            gpu.synchronize(stream)?;
            v[i].push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
    }
    for graph in graphs {
        gpu.destroy_graph(graph)?;
    }
    Ok(v.into_iter()
        .map(|mut x| {
            x.sort_by(f64::total_cmp);
            [x[reps / 2], x[reps / 10], x[reps - 1]]
        })
        .collect())
}

pub fn time(gpu: &dyn GpuBackend, stream: u64, f: &dyn Fn() -> Result<()>) -> Result<[f64; 3]> {
    Ok(time_set(gpu, stream, &[f])?[0])
}

pub fn fmt(t: [f64; 3]) -> String {
    format!("{:8.1} us [p10 {:.1}, max {:.1}]", t[0], t[1], t[2])
}

/// 2026-10-09: One projection of every local expert, and its global-id pointer table.
pub struct Table {
    pub packed: DevicePtr,
    pub scale: DevicePtr,
    pub scale2: DevicePtr,
    pub projs: Vec<Nvfp4Proj>,
    pub bytes_per_expert: usize,
}

pub fn table(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize, gs: f32) -> Result<Table> {
    let mut projs = Vec::with_capacity(LOCAL);
    for _ in 0..LOCAL {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (rng.next() % 9) as u8)
            .collect();
        projs.push(Nvfp4Proj {
            packed: up(gpu, &packed)?,
            scale: up(gpu, &scales)?,
            scale_2: 0.01,
            input_scale: Some(gs),
        });
    }
    let ptrs = |f: &dyn Fn(&Nvfp4Proj) -> u64| -> Vec<u8> {
        (0..EXPERTS)
            .flat_map(|e| projs.get(e).map_or(0, f).to_le_bytes())
            .collect()
    };
    let packed = up(gpu, &ptrs(&|p| p.packed.0))?;
    let scale = up(gpu, &ptrs(&|p| p.scale.0))?;
    let s2: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| projs.get(e).map_or(0.0f32, |p| p.scale_2).to_le_bytes())
        .collect();
    Ok(Table {
        packed,
        scale,
        scale2: up(gpu, &s2)?,
        projs,
        bytes_per_expert: n * k / 2 + n * k / 16,
    })
}

/// 2026-10-09: `rows` rows of `TOP_K` distinct experts each, uniform over all `EXPERTS`.
pub fn routing(rng: &mut Rng, rows: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let e = (rng.next() % EXPERTS as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

/// 2026-10-09: Host port of the serial union builder (the kernel before 2026-10-09):
/// entries in order of first appearance, slot tables written by each id's first thread.
pub fn host_union(ids: &[i32], rows: usize) -> (Vec<i32>, Vec<i32>) {
    let t_n = ids.len();
    let (mut eid, mut slot) = (vec![-1; t_n], vec![-1; t_n * rows]);
    let mut n = 0usize;
    for t in 0..t_n {
        let e = ids[t];
        if e < 0 || ids[..t].contains(&e) {
            continue;
        }
        eid[n] = e;
        for (tp, &x) in ids.iter().enumerate().skip(t) {
            if x == e {
                slot[n * rows + tp / TOP_K] = (tp % TOP_K) as i32;
            }
        }
        n += 1;
    }
    (eid, slot)
}

pub struct Kern {
    pub quant: KernelHandle,
    pub row_union: KernelHandle,
    pub slots: KernelHandle,
    pub union: [KernelHandle; 2],
    pub sweep: [KernelHandle; 2],
    pub ctas: u32,
}

/// 2026-10-09: The quantized activations of one GEMV input: `rows` rows of width `k`.
pub struct Act {
    pub aq: DevicePtr,
    pub as_: DevicePtr,
    pub ag: DevicePtr,
}

pub fn quantize(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    x: DevicePtr,
    rows: usize,
    k: usize,
    gs: f32,
) -> Result<Act> {
    let a = Act {
        aq: gpu.alloc(rows * k / 2)?,
        as_: gpu.alloc(rows * k / 16)?,
        ag: gpu.alloc(rows * 4)?,
    };
    KernelLaunch::new(gpu, kn.quant)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_u32(k as u32)
        .arg_f32(gs)
        .launch(0)?;
    Ok(a)
}

/// 2026-10-09: Device buffers of one routing: ids and the union tables.
pub struct Route {
    pub rows: usize,
    pub ids: DevicePtr,
    pub u_eid: DevicePtr,
    pub u_slot: DevicePtr,
    pub host: Vec<i32>,
}

pub fn row_union(gpu: &dyn GpuBackend, kn: &Kern, r: &Route, s: u64) -> Result<()> {
    KernelLaunch::new(gpu, kn.row_union)
        .grid([1, 1, 1])
        .block([(r.rows * TOP_K) as u32, 1, 1])
        .arg_ptr(r.ids)
        .arg_ptr(r.u_eid)
        .arg_ptr(r.u_slot)
        .arg_u32(r.rows as u32)
        .arg_u32(TOP_K as u32)
        .launch(s)
}

#[allow(clippy::too_many_arguments)]
pub fn union_gemv(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    a: &Act,
    r: &Route,
    t: &Table,
    out: DevicePtr,
    (n, k, act_div): (usize, usize, usize),
    s: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kern)
        .grid([div_ceil(n as u32, 16), (r.rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_ptr(r.u_eid)
        .arg_ptr(r.u_slot)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(r.rows as u32)
        .arg_u32(TOP_K as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(s)
}

pub fn slots_gemv(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    a: &Act,
    r: &Route,
    t: &Table,
    out: DevicePtr,
    (n, k, act_div): (usize, usize, usize),
) -> Result<()> {
    KernelLaunch::new(gpu, kn.slots)
        .grid([div_ceil(n as u32, 16), (r.rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_ptr(r.ids)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(0)
}

/// 2026-10-09: The persistent sweep over one or two projections (`ts`, outputs `outs`).
#[allow(clippy::too_many_arguments)]
pub fn sweep_gemv(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    a: &Act,
    r: &Route,
    ts: &[&Table],
    outs: &[DevicePtr],
    (n, k, act_div): (usize, usize, usize),
    s: u64,
) -> Result<()> {
    let kern = if r.rows <= 8 {
        kn.sweep[0]
    } else {
        kn.sweep[1]
    };
    let mut l = KernelLaunch::new(gpu, kern)
        .grid([kn.ctas, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_ptr(r.u_eid)
        .arg_ptr(r.u_slot);
    for i in 0..2 {
        let (t, o) = (ts.get(i).unwrap_or(&ts[0]), outs.get(i).unwrap_or(&outs[0]));
        l = l
            .arg_ptr(t.packed)
            .arg_ptr(t.scale)
            .arg_ptr(t.scale2)
            .arg_ptr(*o);
    }
    l.arg_u32(ts.len() as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(r.rows as u32)
        .arg_u32(TOP_K as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(s)
}

pub fn union_kernel(kn: &Kern, rows: usize) -> KernelHandle {
    if rows <= 8 { kn.union[0] } else { kn.union[1] }
}

pub fn route(gpu: &dyn GpuBackend, rng: &mut Rng, rows: usize) -> Result<Route> {
    let host = routing(rng, rows);
    let b: Vec<u8> = host.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(Route {
        rows,
        ids: up(gpu, &b)?,
        u_eid: gpu.alloc(rows * TOP_K * 4)?,
        u_slot: gpu.alloc(rows * TOP_K * rows * 4)?,
        host,
    })
}

/// 2026-10-09: Local experts in the union of `r`'s routing.
pub fn local_union(r: &Route) -> usize {
    let mut v: Vec<i32> = r
        .host
        .iter()
        .copied()
        .filter(|&e| (e as usize) < LOCAL)
        .collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}
