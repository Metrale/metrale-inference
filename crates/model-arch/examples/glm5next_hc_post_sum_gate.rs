// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Bit-identity gate and microbench: `glm5next_hc_post_rank_sum` (one launch) against
//! `bf16_add_rank_sum` (the one-shot all-reduce's sum, launched as `NcclBackend` launches it)
//! followed by `glm_hc_post`, the pair the GLM layer runs without `METRALE_GLM_AR_FUSED_POST`.
//! No communicator: the peers' partials are uploaded into a slot buffer laid out as the one-shot
//! receive buffer.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails unless the reduced row (`block_out`) and the post-mixed highway are
//!   byte-identical between the two arms for every case (in place, `out` aliasing `residual`, as
//!   the layer launches it).
//! - Two negative controls must DIFFER from the oracle: the fused arm told the wrong rank (the
//!   slot mapping, so a partial is counted twice), and `hc_post` on the unsummed partial.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_hc_post_sum_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (H 4096, hc 4, world 3): each arm captured as a CUDA graph over `REPS` sites, replayed
//! `TRIALS` times; median, min and max per-site microseconds. `--no-timing` skips it.

use anyhow::{Result, bail};
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, glm_hc_post};
use metrale_model_arch::glm5next_mhc_fused::glm_hc_post_rank_sum;

/// 2026-10-10: `(hidden, hc_mult, tokens, world, rank)`. Rows 1, 4 and 16 at the GLM-5.3 shape on
/// every rank of 3; then other widths, `hc_mult` and worlds.
const CASES: [(usize, usize, usize, usize, usize); 10] = [
    (4096, 4, 1, 3, 0),
    (4096, 4, 1, 3, 1),
    (4096, 4, 1, 3, 2),
    (4096, 4, 4, 3, 1),
    (4096, 4, 16, 3, 2),
    (5120, 4, 3, 3, 0),
    (4096, 3, 2, 4, 3),
    (1024, 2, 5, 5, 2),
    (256, 4, 1, 3, 1),
    (512, 1, 2, 3, 0),
];
/// 2026-10-10: The serve's one-shot bound is 1024 KB: 512 Ki BF16 elements per peer slot.
const SLOT_ELEMS: usize = 512 * 1024;
const TIMED_TOKENS: [usize; 3] = [1, 4, 16];
const REPS: usize = 90;
const TRIALS: usize = 9;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn f32_bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_bytes(d: &[f32]) -> Vec<u8> {
    d.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect()
}

/// 2026-10-10: One site's host inputs; `upload` gives each arm its own device copy, since both
/// arms write their inputs in place.
struct Host {
    own: Vec<u8>,
    peers: Vec<u8>,
    streams: Vec<u8>,
    post: Vec<u8>,
    comb: Vec<u8>,
}

struct Dev {
    own: DevicePtr,
    peers: DevicePtr,
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
}

fn host(hid: usize, hc: usize, t: usize, world: usize, salt: u64) -> Host {
    let mut seed = 0x7375_6d00_u64 ^ (hid as u64) << 20 ^ (hc as u64) << 8 ^ t as u64 ^ salt << 40;
    let n = t * hid;
    let own: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 2.0).collect();
    let mut peers = vec![0f32; (world - 1) * SLOT_ELEMS];
    for s in 0..world - 1 {
        for v in &mut peers[s * SLOT_ELEMS..s * SLOT_ELEMS + n] {
            *v = lcg(&mut seed) * 2.0;
        }
    }
    let streams: Vec<f32> = (0..t * hc * hid).map(|_| lcg(&mut seed) * 3.0).collect();
    let post: Vec<f32> = (0..t * hc).map(|_| 1.0 + lcg(&mut seed)).collect();
    let comb: Vec<f32> = (0..t * hc * hc)
        .map(|_| 0.5 + 0.5 * lcg(&mut seed))
        .collect();
    Host {
        own: bf16_bytes(&own),
        peers: bf16_bytes(&peers),
        streams: f32_bytes(&streams),
        post: f32_bytes(&post),
        comb: f32_bytes(&comb),
    }
}

fn upload(g: &dyn GpuBackend, h: &Host) -> Result<Dev> {
    Ok(Dev {
        own: up(g, &h.own)?,
        peers: up(g, &h.peers)?,
        streams: up(g, &h.streams)?,
        post: up(g, &h.post)?,
        comb: up(g, &h.comb)?,
    })
}

/// 2026-10-10: `bf16_add_rank_sum` with `NcclBackend::all_reduce_oneshot`'s launch shape.
fn rank_sum(
    g: &dyn GpuBackend,
    k: KernelHandle,
    d: &Dev,
    p: &PeerPartials,
    n: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(n as u32).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d.own)
        .arg_ptr(d.peers)
        .arg_u64(p.slot_elems as u64)
        .arg_i32(p.world_size as i32)
        .arg_i32(p.rank as i32)
        .arg_i32(n as i32)
        .launch(stream)
}

fn post(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    d: &Dev,
    (hid, hc, t): (usize, usize, usize),
    stream: u64,
) -> Result<()> {
    glm_hc_post(
        g, k.hc_post, d.own, d.streams, d.post, d.comb, d.streams, t as u32, hid as u32, hc as u32,
        stream,
    )
}

fn fused(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    d: &Dev,
    p: &PeerPartials,
    (hid, hc, t): (usize, usize, usize),
    stream: u64,
) -> Result<()> {
    glm_hc_post_rank_sum(
        g,
        k.hc_post_rank_sum,
        d.own,
        p,
        d.streams,
        d.post,
        d.comb,
        d.streams,
        t as u32,
        hid as u32,
        hc as u32,
        stream,
    )
}

fn outs(g: &dyn GpuBackend, d: &Dev, hid: usize, hc: usize, t: usize) -> Result<[Vec<u8>; 2]> {
    Ok([
        dn(g, d.own, t * hid * 2)?,
        dn(g, d.streams, t * hc * hid * 4)?,
    ])
}

fn partials(d: &Dev, world: usize, rank: usize) -> PeerPartials {
    PeerPartials {
        recv: d.peers.0,
        slot_elems: SLOT_ELEMS,
        world_size: world,
        rank,
    }
}

/// 2026-10-10: Per-site microseconds of `f(stream, i)` for i in 0..REPS captured into one graph,
/// replayed `TRIALS` times: (median, min, max).
fn graph_us(
    g: &dyn GpuBackend,
    stream: u64,
    mut f: impl FnMut(u64, usize) -> Result<()>,
) -> Result<(f64, f64, f64)> {
    g.begin_capture(stream)?;
    for i in 0..REPS {
        if let Err(e) = f(stream, i) {
            g.abort_capture_if_active(stream);
            return Err(e);
        }
    }
    let graph = g.end_capture(stream)?;
    g.launch_graph(graph, stream)?;
    g.synchronize(stream)?;
    let mut v = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let t0 = std::time::Instant::now();
        g.launch_graph(graph, stream)?;
        g.synchronize(stream)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    g.destroy_graph(graph)?;
    v.sort_by(|a, b| a.total_cmp(b));
    Ok((v[TRIALS / 2], v[0], v[TRIALS - 1]))
}

fn main() -> Result<()> {
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Glm5NextMhcKernels::resolve(g)?;
    if k.hc_post_rank_sum.0 == 0 {
        bail!("glm5next_hc_post_rank_sum missing from this build");
    }
    let sum_k = g.kernel("bf16_add", "bf16_add_rank_sum")?;
    println!(
        "glm5next hc_post rank-sum gate: bf16_add_rank_sum + glm5next_hc_post is the ORACLE\n"
    );

    let mut failed = false;
    for (hid, hc, t, world, rank) in CASES {
        let h = host(hid, hc, t, world, 0);
        let shape = (hid, hc, t);
        let a = upload(g, &h)?;
        rank_sum(g, sum_k, &a, &partials(&a, world, rank), t * hid, 0)?;
        post(g, &k, &a, shape, 0)?;
        let b = upload(g, &h)?;
        fused(g, &k, &b, &partials(&b, world, rank), shape, 0)?;
        // 2026-10-10: Control 1: the fused kernel told the wrong rank.
        let c1 = upload(g, &h)?;
        let wrong = (rank + 1) % world;
        fused(g, &k, &c1, &partials(&c1, world, wrong), shape, 0)?;
        // 2026-10-10: Control 2: hc_post on the partial, no sum.
        let c2 = upload(g, &h)?;
        post(g, &k, &c2, shape, 0)?;
        g.synchronize(0)?;
        let (oa, ob) = (outs(g, &a, hid, hc, t)?, outs(g, &b, hid, hc, t)?);
        let (o1, o2) = (outs(g, &c1, hid, hc, t)?, outs(g, &c2, hid, hc, t)?);
        let same = oa == ob;
        let c1_differs = o1[1] != oa[1];
        let c2_differs = o2[1] != oa[1];
        println!(
            "  H={hid:5} hc={hc} T={t:3} world={world} rank={rank}: reduced row {} highway {} | \
             control wrong-rank {} | control no-sum {}",
            if oa[0] == ob[0] { "SAME" } else { "DIFF" },
            if oa[1] == ob[1] { "SAME" } else { "DIFF" },
            if c1_differs {
                "differs (ok)"
            } else {
                "SAME (BAD)"
            },
            if c2_differs {
                "differs (ok)"
            } else {
                "SAME (BAD)"
            },
        );
        failed |= !same || !c1_differs || !c2_differs;
        for d in [a, b, c1, c2] {
            for p in [d.own, d.peers, d.streams, d.post, d.comb] {
                g.free(p)?;
            }
        }
    }

    if timing {
        let stream = g.create_stream()?;
        let (hid, hc, world, rank) = (4096usize, 4usize, 3usize, 1usize);
        println!(
            "\n  TIMING H={hid} hc={hc} world={world}: graph of {REPS} sites x {TRIALS} replays, us/site median [min, max]"
        );
        for t in TIMED_TOKENS {
            let h = host(hid, hc, t, world, 1);
            let d = upload(g, &h)?;
            let p = partials(&d, world, rank);
            let shape = (hid, hc, t);
            let old = graph_us(g, stream, |st, _| {
                rank_sum(g, sum_k, &d, &p, t * hid, st)?;
                post(g, &k, &d, shape, st)
            })?;
            let new = graph_us(g, stream, |st, _| fused(g, &k, &d, &p, shape, st))?;
            println!(
                "    T={t:3}: rank_sum+hc_post {:6.2} [{:6.2}, {:6.2}]  fused {:6.2} [{:6.2}, {:6.2}]  saved {:5.2}",
                old.0,
                old.1,
                old.2,
                new.0,
                new.1,
                new.2,
                old.0 - new.0
            );
            for q in [d.own, d.peers, d.streams, d.post, d.comb] {
                g.free(q)?;
            }
        }
    }

    if failed {
        bail!("glm5next_hc_post_rank_sum gate FAILED");
    }
    println!("\nPASS: fused output byte-identical on every case; both controls differ");
    Ok(())
}
