// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `GLM_BENCH_PWD=1`: the go/no-go of the decode-only wide union sweep. The routed
//! W4A4 experts of one tp rank (704 wide, all 288 experts, top 8) on the serve's kernels, at
//! 16 rows (`mx16`), 24 (`mx24`), 32 (`mx32`) and 32 as two 16-row groups (the serial path),
//! all on the same routing per draw. The router ids are injected (`pwd_route.rs`): `uniform`,
//! and a `zipf` and a `paired` model each fitted to the serve's measured U(16) (`GLM_PWD_U16`,
//! default `pwd_verdict::U16_SERVE`), or real ids from `GLM_PWD_IDS`.
//!
//! Per draw it builds every case once (quantize, union, gate+up sweep, SwiGLU, quantize, down
//! sweep) and checks bytes: the 32-row union tables against the host builder, and every
//! (row, slot) of the 24- and 32-row gate, up and down outputs against the 16-row groups'.
//! Then it times (CUDA-graph replay, all candidates interleaved per round) the gate+up sweep,
//! the down sweep and the whole pipeline of each case, and prints one `PWD` line. Per routing
//! model a `SUMMARY` and a `VERDICT` line follow, and one `DECISION` line ends the run.
//!
//! The go rule, per layer, `t` the median gate+up plus down sweep time:
//! - plan: U(32) <= `pwd_verdict::U32_PLAN_MAX` and t(32) < t(2x16).
//! - break-even: the wide verify's MoE fits the step budget at which it only ties the serial
//!   rate (`pwd_verdict::RUNGS`); in the ratio form `t(rows) <= t(16) * budget / T16_SERVE_US`, which
//!   carries the bench-to-serve difference of t(16) over to the wide cases.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch, an id outside the contract or
//! an absent wide entry.

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::pwd_route::{self, Router};
use super::pwd_verdict::{Draw, U16_SERVE, row, verdict};
use super::*;

const WIDTH: usize = 704;
/// 2026-10-15: The `_k64` down quantizer's stored row width for a 704-wide input.
const K_STORE: usize = 768;
const GROUP: usize = 16;
const WIDE: usize = 32;
const SWIGLU_LIMIT: f32 = 10.0;
const BLOCK: u32 = 256;

/// 2026-10-15: One routed launch: its routing, input rows, quantized activations and outputs.
struct Case {
    route: Route,
    x: DevicePtr,
    a: Act,
    gate: DevicePtr,
    up: DevicePtr,
    act: DevicePtr,
    d: Act,
    out: DevicePtr,
}

struct Tables {
    gate: Table,
    up: Table,
    down: Table,
}

impl Case {
    fn new(g: &dyn GpuBackend, rows: usize, x: DevicePtr) -> Result<Self> {
        let slots = rows * TOP_K;
        let act = |r: usize, k: usize| -> Result<Act> {
            Ok(Act {
                aq: g.alloc(r * k / 2)?,
                as_: g.alloc(r * k / 16)?,
                ag: g.alloc(r * 4)?,
            })
        };
        Ok(Self {
            route: Route {
                rows,
                ids: g.alloc(slots * 4)?,
                u_eid: g.alloc(slots * 4)?,
                u_slot: g.alloc(slots * rows * 4)?,
                host: Vec::new(),
            },
            x,
            a: act(rows, HIDDEN)?,
            gate: g.alloc(slots * WIDTH * 2)?,
            up: g.alloc(slots * WIDTH * 2)?,
            act: g.alloc(slots * WIDTH * 2)?,
            d: act(slots, K_STORE)?,
            out: g.alloc(slots * HIDDEN * 2)?,
        })
    }

    fn set_ids(&mut self, g: &dyn GpuBackend, ids: &[i32]) -> Result<()> {
        ensure!(
            ids.len() == self.route.rows * TOP_K,
            "ids for another row count"
        );
        let b: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        g.copy_h2d(&b, self.route.ids)?;
        self.route.host = ids.to_vec();
        Ok(())
    }

    fn gate_up(
        &self,
        g: &dyn GpuBackend,
        k: &Glm5NextMlpKernels,
        t: &Tables,
        s: u64,
    ) -> Result<()> {
        let projs = [(&t.gate, self.gate), (&t.up, self.up)];
        sweep(
            g,
            k,
            k.w4a4_moe_sweep,
            &self.a,
            &self.route,
            &projs,
            (WIDTH, HIDDEN, TOP_K),
            s,
        )
    }

    fn down(&self, g: &dyn GpuBackend, k: &Glm5NextMlpKernels, t: &Tables, s: u64) -> Result<()> {
        let projs = [(&t.down, self.out)];
        let sw = k.w4a4_moe_k64.sweep;
        sweep(
            g,
            k,
            sw,
            &self.d,
            &self.route,
            &projs,
            (HIDDEN, WIDTH, 1),
            s,
        )
    }

    /// 2026-10-15: The routed experts as `w4a4_experts` runs them on the sweep path.
    fn full(&self, g: &dyn GpuBackend, k: &Glm5NextMlpKernels, t: &Tables, s: u64) -> Result<()> {
        let (rows, slots) = (self.route.rows, self.route.rows * TOP_K);
        quant(
            g,
            k.w4a4_quant_static,
            self.x,
            &self.a,
            rows,
            HIDDEN,
            GS_GATE_UP,
            s,
        )?;
        KernelLaunch::new(g, k.moe_row_union)
            .grid([1, 1, 1])
            .block([slots as u32, 1, 1])
            .arg_ptr(self.route.ids)
            .arg_ptr(self.route.u_eid)
            .arg_ptr(self.route.u_slot)
            .arg_u32(rows as u32)
            .arg_u32(TOP_K as u32)
            .launch(s)?;
        self.gate_up(g, k, t, s)?;
        let n = slots * WIDTH;
        KernelLaunch::new(g, k.swiglu)
            .grid([(n as u32).div_ceil(BLOCK), 1, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(self.gate)
            .arg_ptr(self.up)
            .arg_ptr(self.act)
            .arg_u32(n as u32)
            .arg_f32(SWIGLU_LIMIT)
            .launch(s)?;
        quant(
            g,
            k.w4a4_moe_k64.quant,
            self.act,
            &self.d,
            slots,
            WIDTH,
            GS_DOWN,
            s,
        )?;
        self.down(g, k, t, s)
    }
}

#[allow(clippy::too_many_arguments)]
fn quant(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    x: DevicePtr,
    a: &Act,
    rows: usize,
    k: usize,
    gs: f32,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([rows as u32, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(x)
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_u32(k as u32)
        .arg_f32(gs)
        .launch(s)
}

/// 2026-10-15: The union sweep entry for `rows` rows, as `w4a4::sweep_kernel` picks it with
/// the wide lever on.
fn entry(sw: [KernelHandle; 4], rows: usize) -> Result<KernelHandle> {
    let h = match rows {
        2..=8 => sw[0],
        9..=16 => sw[1],
        17..=24 => sw[2],
        25..=32 => sw[3],
        _ => anyhow::bail!("no union sweep entry for {rows} rows"),
    };
    ensure!(
        h.0 != 0,
        "the {rows}-row union sweep is absent from the PTX"
    );
    Ok(h)
}

/// 2026-10-15: The persistent union sweep over one or two projections (the kernel reads the
/// second table only when `nproj` is 2), the argument order of `w4a4::moe_sweep`.
#[allow(clippy::too_many_arguments)]
fn sweep(
    g: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    sw: [KernelHandle; 4],
    a: &Act,
    r: &Route,
    projs: &[(&Table, DevicePtr)],
    (n, kk, act_div): (usize, usize, usize),
    s: u64,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, entry(sw, r.rows)?)
        .grid([k.w4a4_sweep_ctas, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_ptr(r.u_eid)
        .arg_ptr(r.u_slot);
    for (t, o) in [projs[0], projs[projs.len() - 1]] {
        l = l
            .arg_ptr(t.packed)
            .arg_ptr(t.scale)
            .arg_ptr(t.scale2)
            .arg_ptr(o);
    }
    l.arg_u32(projs.len() as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(r.rows as u32)
        .arg_u32(TOP_K as u32)
        .arg_u32(act_div as u32)
        .arg_u32(EXPERTS as u32)
        .launch(s)
}

pub fn run(g: &dyn GpuBackend, k: &Glm5NextMlpKernels, stream: u64) -> Result<()> {
    ensure!(
        k.moe_row_union.0 != 0 && k.w4a4_sweep_ctas != 0 && k.w4a4_moe_k64.quant.0 != 0,
        "GLM_BENCH_PWD=1 needs the union builder, the sweep grid and the _k64 quantizer"
    );
    for rows in [GROUP, 24, WIDE] {
        entry(k.w4a4_moe_sweep, rows)?;
        entry(k.w4a4_moe_k64.sweep, rows)?;
    }
    let target = std::env::var("GLM_PWD_U16")
        .ok()
        .map(|v| v.parse::<f64>())
        .transpose()?
        .unwrap_or(U16_SERVE);
    let mut rng = Rng(0xbb67_ae85_84ca_a73b);
    let t = Tables {
        gate: table_n(g, &mut rng, WIDTH, HIDDEN, GS_GATE_UP, EXPERTS)?,
        up: table_n(g, &mut rng, WIDTH, HIDDEN, GS_GATE_UP, EXPERTS)?,
        down: table_n(g, &mut rng, HIDDEN, WIDTH, GS_DOWN, EXPERTS)?,
    };
    let expert_bytes = t.gate.bytes_per_expert + t.up.bytes_per_expert + t.down.bytes_per_expert;
    let x = up_bf16(
        g,
        &(0..WIDE * HIDDEN)
            .map(|_| rng.unit() * 3.0)
            .collect::<Vec<_>>(),
    )?;
    // 2026-10-15: a = rows 0..16 (the 16-row case and the serial path's first group), b = rows
    // 16..32, c24 = rows 0..24, c32 = rows 0..32.
    let mut cs = [
        Case::new(g, GROUP, x)?,
        Case::new(g, GROUP, x.offset(GROUP * HIDDEN * 2))?,
        Case::new(g, 24, x)?,
        Case::new(g, WIDE, x)?,
    ];
    let mut routers = vec![Router::Uniform];
    match std::env::var("GLM_PWD_IDS") {
        Ok(path) => routers.push(pwd_route::from_file(&path)?),
        Err(_) => {
            routers.push(pwd_route::fit_zipf(target, 0x3c6e_f372_fe94_f82b)?);
            routers.push(pwd_route::fit_paired(target)?);
        }
    }
    println!(
        "pwd bench: tp {WIDTH}, {EXPERTS} experts, top {TOP_K}, {:.3} MB per expert per rank, \
         U(16) target {target}, ctas {}",
        expert_bytes as f64 / 1e6,
        k.w4a4_sweep_ctas
    );
    let mut decisive = Vec::new();
    for mut router in routers {
        let name = router.name();
        let mut draws = Vec::new();
        for d in 0..env_or("GLM_BENCH_DRAWS", 6) {
            let ids = router.draw(&mut rng, WIDE)?;
            let split = GROUP * TOP_K;
            cs[0].set_ids(g, &ids[..split])?;
            cs[1].set_ids(g, &ids[split..])?;
            cs[2].set_ids(g, &ids[..24 * TOP_K])?;
            cs[3].set_ids(g, &ids)?;
            let draw = one_draw(g, k, &t, &cs, stream)?;
            println!(
                "PWD route={name} draw={d} U16={} U24={} U32={} U2x16={}+{} | gu_us {} | \
                 down_us {} | full_us {} | sweep_GBps32 {:.0}",
                draw.u[0],
                draw.u[2],
                draw.u[3],
                draw.u[0],
                draw.u[1],
                row(&draw.gu),
                row(&draw.dn),
                row(&draw.full),
                draw.u[3] as f64 * expert_bytes as f64 / ((draw.gu[2] + draw.dn[2]) * 1e3)
            );
            draws.push(draw);
        }
        let go = verdict(&name, &draws, expert_bytes);
        if !matches!(router, Router::Uniform) {
            decisive.push(go);
        }
    }
    let all = |i: usize| decisive.iter().all(|g| g[i]);
    let word = |b: bool| if b { "GO" } else { "NO-GO" };
    println!(
        "DECISION plan={} break_even_c12={} break_even_c16={} (over the fitted or file routings)",
        word(all(0)),
        word(all(1)),
        word(all(2))
    );
    Ok(())
}

fn one_draw(
    g: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    t: &Tables,
    cs: &[Case; 4],
    s: u64,
) -> Result<Draw> {
    for c in cs {
        c.full(g, k, t, s)?;
    }
    g.synchronize(s)?;
    check_bytes(g, cs)?;
    let [a, b, c24, c32] = cs;
    type F<'a> = Box<dyn Fn() -> Result<()> + 'a>;
    let mut fs: Vec<F<'_>> = Vec::new();
    type Stage = fn(&Case, &dyn GpuBackend, &Glm5NextMlpKernels, &Tables, u64) -> Result<()>;
    let stages: [Stage; 3] = [Case::gate_up, Case::down, Case::full];
    for f in stages {
        for c in [a, c24, c32] {
            fs.push(Box::new(move || f(c, g, k, t, s)));
        }
        fs.push(Box::new(move || {
            f(a, g, k, t, s)?;
            f(b, g, k, t, s)
        }));
    }
    let refs: Vec<&dyn Fn() -> Result<()>> = fs.iter().map(|f| f.as_ref()).collect();
    let m = time_set(g, s, &refs)?;
    let pick = |i: usize| [m[i][0], m[i + 1][0], m[i + 2][0], m[i + 3][0]];
    let u = |c: &Case| pwd_route::union_size(&c.route.host);
    Ok(Draw {
        u: [u(a), u(b), u(c24), u(c32)],
        gu: pick(0),
        dn: pick(4),
        full: pick(8),
    })
}

/// 2026-10-15: The 32-row union tables against the host builder, and each (row, slot) of the
/// wide cases' outputs against the 16-row groups' bytes: the sweep computes a row in the same
/// order whatever the row count.
fn check_bytes(g: &dyn GpuBackend, cs: &[Case; 4]) -> Result<()> {
    let [a, b, c24, c32] = cs;
    let r = &c32.route;
    let (eid, slot) = host_union(&r.host, WIDE);
    ensure!(
        read_i32(g, r.u_eid, WIDE * TOP_K)? == eid
            && read_i32(g, r.u_slot, WIDE * TOP_K * WIDE)? == slot,
        "the 32-row union tables differ from the host builder"
    );
    let half = GROUP * TOP_K;
    for (what, width, f) in [
        ("gate", WIDTH, (|c: &Case| c.gate) as fn(&Case) -> DevicePtr),
        ("up", WIDTH, |c: &Case| c.up),
        ("down", HIDDEN, |c: &Case| c.out),
    ] {
        let bytes = |c: &Case, slots: usize| read(g, f(c), slots * width * 2);
        let (ga, gb) = (bytes(a, half)?, bytes(b, half)?);
        let w32 = bytes(c32, 2 * half)?;
        let w24 = bytes(c24, 24 * TOP_K)?;
        ensure!(
            w32[..ga.len()] == ga[..] && w32[ga.len()..] == gb[..],
            "{what}: the 32-row sweep differs from the two 16-row groups"
        );
        ensure!(
            w24[..ga.len()] == ga[..] && w24[ga.len()..] == gb[..w24.len() - ga.len()],
            "{what}: the 24-row sweep differs from the 16-row groups"
        );
    }
    Ok(())
}
