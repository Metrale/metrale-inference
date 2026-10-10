// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Part 2 of `glm5next_moe_narrow_bench`: the routed experts' GEMVs (gate, up and
//! down of one decode step's layer) at one row and at 2..=8 rows of one sequence. Each graph
//! replays `ITERS` calls, call i on routing i of a set (so consecutive calls read different
//! experts and the weights come from DRAM, as layer after layer does), every candidate on the
//! same set, interleaved. GB/s is over the local experts' gate, up and down bytes.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use std::cell::Cell;

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::device::*;
use crate::{DN, Fixture, GU, Setup, own_sweep, route_of, row_with_local, slots_on};

/// 2026-10-09: Distinct local experts over all rows of `r`.
fn local_union_of(host: &[i32]) -> usize {
    let mut v: Vec<i32> = host
        .iter()
        .copied()
        .filter(|&e| e >= 0 && (e as usize) < LOCAL)
        .collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// 2026-10-09: Times each candidate over the routing set and prints medians with GB/s.
fn time_routes(gpu: &dyn GpuBackend, s: &Setup, label: &str, rs: &[Route], st: u64) -> Result<()> {
    let rows = rs[0].rows;
    let mean_u = rs.iter().map(|r| local_union_of(&r.host)).sum::<usize>() as f64 / rs.len() as f64;
    let bytes = mean_u * (2 * s.gate.bytes_per_expert + s.down.bytes_per_expert) as f64;
    let (o0, o1) = (s.out[0], s.out[1]);
    let (xa, da) = (&s.x_act, &s.d_act);
    let i = Cell::new(0usize);
    let next = || {
        let r = &rs[i.get() % rs.len()];
        i.set(i.get() + 1);
        r
    };
    let slots = || -> Result<()> {
        let r = next();
        slots_on(gpu, s.kn.slots, xa, r, &s.gate, o0, GU, st)?;
        slots_on(gpu, s.kn.slots, xa, r, &s.upt, o1, GU, st)?;
        slots_on(gpu, s.kn.slots, da, r, &s.down, o0, DN, st)
    };
    let union = || -> Result<()> {
        let r = next();
        row_union(gpu, &s.kn, r, st)?;
        sweep_gemv(gpu, &s.kn, xa, r, &[&s.gate, &s.upt], &[o0, o1], GU, st)?;
        sweep_gemv(gpu, &s.kn, da, r, &[&s.down], &[o0], DN, st)
    };
    let build = || -> Result<()> { row_union(gpu, &s.kn, next(), st) };
    let union_gu = || -> Result<()> {
        let r = next();
        sweep_gemv(gpu, &s.kn, xa, r, &[&s.gate, &s.upt], &[o0, o1], GU, st)
    };
    let union_dn = || -> Result<()> {
        let r = next();
        sweep_gemv(gpu, &s.kn, da, r, &[&s.down], &[o0], DN, st)
    };
    let own = || -> Result<()> {
        let r = next();
        own_sweep(
            gpu,
            s.own,
            s.kn.ctas,
            xa,
            r,
            &[&s.gate, &s.upt],
            &[o0, o1],
            GU,
            st,
        )?;
        own_sweep(gpu, s.own, s.kn.ctas, da, r, &[&s.down], &[o0], DN, st)
    };
    // 2026-10-09: The union tables the union-only candidates read are built once per routing.
    for r in rs {
        row_union(gpu, &s.kn, r, st)?;
    }
    type F<'a> = &'a dyn Fn() -> Result<()>;
    let mut fs: Vec<(&str, F)> = vec![
        ("union: build + sweep gu + sweep d", &union as F),
        ("  union sweep gate+up only", &union_gu as F),
        ("  union sweep down only", &union_dn as F),
        ("  union build only", &build as F),
    ];
    if rows == 1 {
        fs.insert(0, ("slot GEMV gate, up, down", &slots as F));
        fs.push(("own-slots sweep gu + d", &own as F));
    }
    let closures: Vec<F> = fs.iter().map(|(_, f)| *f).collect();
    let t = time_set(gpu, st, &closures)?;
    println!(
        "{label}: {rows} row(s), mean local union {mean_u:.2} ({:.1} MB)",
        bytes / 1e6
    );
    for ((name, _), t) in fs.iter().zip(&t) {
        let gbs = if name.starts_with("  ") {
            String::new()
        } else {
            format!("  {:6.1} GB/s", bytes / t[0] / 1e3)
        };
        println!("  {name:34} {}{gbs}", fmt(*t));
    }
    Ok(())
}

/// 2026-10-09: `n` routings of `rows` rows of one sequence whose local union is `u` experts:
/// each local expert goes to at least one row, each row takes 1..=3 of them, the other slots
/// remote.
fn seq_routes(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    rows: usize,
    u: usize,
    n: usize,
) -> Result<Vec<Route>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut pool: Vec<i32> = Vec::with_capacity(u);
        while pool.len() < u {
            let e = (rng.next() % LOCAL as u64) as i32;
            if !pool.contains(&e) {
                pool.push(e);
            }
        }
        let mut host = Vec::with_capacity(rows * TOP_K);
        for r in 0..rows {
            let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
            // 2026-10-09: Expert j of the pool goes to row j % rows first.
            for (j, &e) in pool.iter().enumerate() {
                if j % rows == r && row.len() < TOP_K {
                    row.push(e);
                }
            }
            let want = 1 + (rng.next() % 3) as usize;
            while row.len() < want.min(u) {
                let e = pool[(rng.next() % u as u64) as usize];
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            while row.len() < TOP_K {
                let e = (LOCAL as u64 + rng.next() % (EXPERTS - LOCAL) as u64) as i32;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            host.extend(row);
        }
        out.push(route_of(gpu, host)?);
    }
    Ok(out)
}

pub fn run(
    gpu: &dyn GpuBackend,
    s: &Setup,
    rng: &mut Rng,
    fx: Option<&Fixture>,
    st: u64,
) -> Result<()> {
    let big = 256usize << 20;
    let (src, dst) = (gpu.alloc(big)?, gpu.alloc(big)?);
    // 2026-10-09: The DRAM reference, first and last: a lower last copy flags a run that
    // another process shared.
    let copy = |when: &str| -> Result<()> {
        let cp = time(gpu, st, &|| gpu.copy_d2d_async(src, dst, big, st))?;
        println!(
            "device copy 256 MiB ({when}): {}  ({:.0} GB/s read + write)",
            fmt(cp),
            2.0 * big as f64 / cp[0] / 1e3
        );
        Ok(())
    };
    copy("first")?;
    let n = env_or("GLM_BENCH_ITERS", ITERS);
    for local in [1usize, 2, 3, 4, 6, 8] {
        let rs: Vec<Route> = (0..n)
            .map(|_| route_of(gpu, row_with_local(rng, local)))
            .collect::<Result<_>>()?;
        time_routes(gpu, s, &format!("1 row, {local} local"), &rs, st)?;
    }
    if let Some(f) = fx {
        // 2026-10-09: Fixture rows spread over its layers and tokens, one per call.
        let step = (f.layers * f.tokens / n).max(1);
        let rs: Vec<Route> = (0..n)
            .map(|i| {
                let t = (i * step) % (f.layers * f.tokens);
                route_of(gpu, f.ids[t * TOP_K..(t + 1) * TOP_K].to_vec())
            })
            .collect::<Result<_>>()?;
        time_routes(gpu, s, "1 row, fixture", &rs, st)?;
    }
    for rows in [2usize, 4, 8] {
        for u in [3usize, 5, 8] {
            let rs = seq_routes(gpu, rng, rows, u, n)?;
            time_routes(gpu, s, &format!("one sequence, local union {u}"), &rs, st)?;
        }
        if let Some(f) = fx {
            // 2026-10-09: Consecutive-token windows of the fixture, inside one layer.
            let per = f.tokens / rows;
            let rs: Vec<Route> = (0..n)
                .map(|i| {
                    let (l, w) = (i % f.layers, (i / f.layers) % per);
                    let t0 = l * f.tokens + w * rows;
                    route_of(gpu, f.ids[t0 * TOP_K..(t0 + rows) * TOP_K].to_vec())
                })
                .collect::<Result<_>>()?;
            time_routes(gpu, s, "one sequence, fixture windows", &rs, st)?;
        }
    }
    copy("last")?;
    gpu.free(src)?;
    gpu.free(dst)
}
