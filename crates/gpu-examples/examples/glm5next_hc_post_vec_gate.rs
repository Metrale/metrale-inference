// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Bit-identity gate and microbench: `glm5next_hc_post_rank_sum_vec8`
//! (`METRALE_GLM_MHC_POST_VEC`, `glm5next_hc_post_vec`) against `glm5next_hc_post_rank_sum`, the
//! kernel the SP-mHC prefill post-mix runs without the lever. No communicator: the peers'
//! partials are uploaded into a slot buffer laid out as the one-shot receive buffer.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails unless, for every case and both `write_x` values, the post-mixed streams are
//!   byte-identical to the scalar kernel's (in place, `out` aliasing `residual`, as the layer
//!   launches it); with `write_x = 1` the summed row is the scalar kernel's bytes, and with
//!   `write_x = 0` the row still holds this rank's partial.
//! - The cases cover the prefill pieces' row counts (1, 86, 341, 684, 2736, 8208) on every rank of
//!   3 at the GLM-5.3 shape, plus other widths (one not a multiple of a block), `hc_mult` and
//!   worlds.
//! - Negative control, which must DIFFER: one bit of a peer's partial flipped in the last column
//!   of the last row, so a kernel that skipped the tail of a row or the last row would pass the
//!   gate and fail the control.
//!
//!   cargo run -p metrale-gpu-examples --release --example glm5next_hc_post_vec_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (H 4096, hc 4, world 3, rank 1): each arm captured as a CUDA graph over `REPS` sites,
//! replayed `TRIALS` times; median, min and max per-site microseconds. `--no-timing` skips it.

use anyhow::{Result, bail, ensure};
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_hc_post_vec as vec8;
use metrale_model_arch::glm5next_mhc::Glm5NextMhcKernels;
use metrale_model_arch::glm5next_mhc_fused::glm_hc_post_rank_sum;

/// 2026-10-15: The widest case: an 8208-row piece at H 4096, hc 4, world 5.
const MAX_ROWS: usize = 8208;
const MAX_HID: usize = 4096;
const MAX_HC: usize = 4;
const MAX_WORLD: usize = 5;
/// 2026-10-15: A slot stride past the widest piece, so a kernel that strides peers by the piece
/// size instead of the slot reads the wrong rows.
const SLOT_ELEMS: usize = MAX_ROWS * MAX_HID + 8 * 1024;
const GLM_ROWS: [usize; 6] = [1, 86, 341, 684, 2736, 8208];
/// 2026-10-15: `(hidden, hc_mult, tokens, world, rank)` beyond the GLM-5.3 shape.
const EXTRA: [(usize, usize, usize, usize, usize); 5] = [
    (5120, 4, 3, 3, 0),
    (4096, 3, 17, 4, 3),
    (1024, 2, 5, 5, 2),
    (1032, 4, 7, 3, 1),
    (512, 1, 2, 3, 0),
];
const TIMED_ROWS: [usize; 5] = [86, 341, 684, 2736, 8208];
const REPS: usize = 20;
const TRIALS: usize = 7;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn f32_bytes(n: usize, scale: f32, bias: f32, seed: &mut u64) -> Vec<u8> {
    (0..n)
        .flat_map(|_| (bias + scale * lcg(seed)).to_le_bytes())
        .collect()
}

fn bf16_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
    (0..n)
        .flat_map(|_| half::bf16::from_f32(2.0 * lcg(seed)).to_le_bytes())
        .collect()
}

/// 2026-10-15: Host inputs at the widest shape; a case uploads the prefix it needs. Both arms
/// write `own` (the summed row) and `streams` in place, so each run re-uploads them.
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

fn host() -> Host {
    let mut seed = 0x7665_6338_u64;
    Host {
        own: bf16_bytes(MAX_ROWS * MAX_HID, &mut seed),
        peers: bf16_bytes((MAX_WORLD - 1) * SLOT_ELEMS, &mut seed),
        streams: f32_bytes(MAX_ROWS * MAX_HC * MAX_HID, 3.0, 0.0, &mut seed),
        post: f32_bytes(MAX_ROWS * MAX_HC, 1.0, 1.0, &mut seed),
        comb: f32_bytes(MAX_ROWS * MAX_HC * MAX_HC, 0.5, 0.5, &mut seed),
    }
}

fn alloc(g: &dyn GpuBackend, h: &Host) -> Result<Dev> {
    let d = Dev {
        own: g.alloc(h.own.len())?,
        peers: g.alloc(h.peers.len())?,
        streams: g.alloc(h.streams.len())?,
        post: g.alloc(h.post.len())?,
        comb: g.alloc(h.comb.len())?,
    };
    g.copy_h2d(&h.peers, d.peers)?;
    g.copy_h2d(&h.post, d.post)?;
    g.copy_h2d(&h.comb, d.comb)?;
    Ok(d)
}

/// 2026-10-15: One case's shape: `(hidden, hc_mult, tokens, world, rank)`.
type Case = (usize, usize, usize, usize, usize);

fn partials(d: &Dev, (_, _, _, world, rank): Case) -> PeerPartials {
    PeerPartials {
        recv: d.peers.0,
        slot_elems: SLOT_ELEMS,
        world_size: world,
        rank,
    }
}

/// 2026-10-15: Re-upload the in-place inputs, run `arm` (None = the scalar kernel, Some(write_x)
/// = the vectorized one) and read back `(summed row, streams)`.
fn run(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    h: &Host,
    d: &Dev,
    c: Case,
    arm: Option<bool>,
) -> Result<[Vec<u8>; 2]> {
    let (hid, hc, t, _, _) = c;
    g.copy_h2d(&h.own[..t * hid * 2], d.own)?;
    g.copy_h2d(&h.streams[..t * hc * hid * 4], d.streams)?;
    launch(g, k, d, c, arm, 0)?;
    g.synchronize(0)?;
    Ok([
        dn(g, d.own, t * hid * 2)?,
        dn(g, d.streams, t * hc * hid * 4)?,
    ])
}

fn launch(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    d: &Dev,
    c: Case,
    arm: Option<bool>,
    stream: u64,
) -> Result<()> {
    let (hid, hc, t, _, _) = c;
    let p = partials(d, c);
    let dims = [t as u32, hid as u32, hc as u32];
    match arm {
        None => glm_hc_post_rank_sum(
            g,
            k.hc_post_rank_sum,
            d.own,
            &p,
            d.streams,
            d.post,
            d.comb,
            d.streams,
            dims[0],
            dims[1],
            dims[2],
            stream,
        ),
        Some(write_x) => vec8::launch(
            g,
            k.hc_post_rank_sum_vec8,
            d.own,
            &p,
            [d.streams, d.post, d.comb, d.streams],
            dims,
            write_x,
            stream,
        ),
    }
}

/// 2026-10-15: The gate of one case; true when it passes and its control fires.
fn gate(g: &dyn GpuBackend, k: &Glm5NextMhcKernels, h: &Host, d: &Dev, c: Case) -> Result<bool> {
    let (hid, hc, t, world, rank) = c;
    let oracle = run(g, k, h, d, c, None)?;
    let [x0, s0] = run(g, k, h, d, c, Some(false))?;
    let [x1, s1] = run(g, k, h, d, c, Some(true))?;
    let partial = &h.own[..t * hid * 2];
    let ok = s0 == oracle[1] && s1 == oracle[1] && x1 == oracle[0] && x0 == partial;
    // 2026-10-15: Control: flip an exponent bit of the first peer slot's last element of the
    // piece, then restore it.
    let at = ((t - 1) * hid + hid - 1) * 2 + 1;
    let byte = DevicePtr(d.peers.0 + at as u64);
    g.copy_h2d(&[h.peers[at] ^ 0x40], byte)?;
    let [_, sc] = run(g, k, h, d, c, Some(true))?;
    g.copy_h2d(&h.peers[at..=at], byte)?;
    let fires = sc != oracle[1];
    println!(
        "  H={hid:5} hc={hc} T={t:5} world={world} rank={rank}: streams wx0 {} wx1 {} | row wx1 {} \
         wx0 {} | control {}",
        if s0 == oracle[1] { "SAME" } else { "DIFF" },
        if s1 == oracle[1] { "SAME" } else { "DIFF" },
        if x1 == oracle[0] { "SAME" } else { "DIFF" },
        if x0 == partial {
            "untouched"
        } else {
            "WRITTEN (BAD)"
        },
        if fires { "differs (ok)" } else { "SAME (BAD)" },
    );
    Ok(ok && fires)
}

/// 2026-10-15: Per-site microseconds of `REPS` launches captured into one graph, replayed
/// `TRIALS` times: (median, min, max).
fn graph_us(
    g: &dyn GpuBackend,
    stream: u64,
    f: impl Fn(u64) -> Result<()>,
) -> Result<(f64, f64, f64)> {
    g.begin_capture(stream)?;
    for _ in 0..REPS {
        if let Err(e) = f(stream) {
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
    ensure!(
        k.hc_post_rank_sum.0 != 0 && k.hc_post_rank_sum_vec8.0 != 0,
        "glm5next_hc_post_rank_sum or its vec8 twin is missing from this build"
    );
    println!("glm5next hc_post rank-sum vec8 gate: glm5next_hc_post_rank_sum is the ORACLE\n");
    let h = host();
    let d = alloc(g, &h)?;

    let mut failed = false;
    let glm = GLM_ROWS
        .iter()
        .flat_map(|&t| (0..3).map(move |r| (MAX_HID, MAX_HC, t, 3, r)));
    for c in glm.chain(EXTRA) {
        failed |= !gate(g, &k, &h, &d, c)?;
    }

    if timing {
        let stream = g.create_stream()?;
        println!(
            "\n  TIMING H={MAX_HID} hc={MAX_HC} world=3 rank=1: graph of {REPS} sites x {TRIALS} \
             replays, us/site median [min, max]"
        );
        for t in TIMED_ROWS {
            let c = (MAX_HID, MAX_HC, t, 3, 1);
            g.copy_h2d(&h.own[..t * MAX_HID * 2], d.own)?;
            g.copy_h2d(&h.streams[..t * MAX_HC * MAX_HID * 4], d.streams)?;
            let arm = |a| graph_us(g, stream, |st| launch(g, &k, &d, c, a, st));
            let old = arm(None)?;
            let wx0 = arm(Some(false))?;
            let wx1 = arm(Some(true))?;
            let again = arm(None)?;
            println!(
                "    T={t:5}: scalar {:8.2} [{:8.2}, {:8.2}]  vec8 wx0 {:8.2} [{:8.2}, {:8.2}]  \
                 vec8 wx1 {:8.2} [{:8.2}, {:8.2}]  scalar again {:8.2}  saved wx0 {:7.2}",
                old.0,
                old.1,
                old.2,
                wx0.0,
                wx0.1,
                wx0.2,
                wx1.0,
                wx1.1,
                wx1.2,
                again.0,
                old.0.min(again.0) - wx0.0,
            );
        }
    }
    for p in [d.own, d.peers, d.streams, d.post, d.comb] {
        g.free(p)?;
    }

    if failed {
        bail!("glm5next_hc_post_rank_sum_vec8 gate FAILED");
    }
    println!(
        "\nPASS: vec8 streams byte-identical on every case at both write_x; the summed row matches \
         at write_x 1 and is untouched at 0; every control differs"
    );
    Ok(())
}
