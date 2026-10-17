// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Bit-identity gate and microbench: `glm5next_hc_post_pre_norm` (one launch) against
//! the decode's unfused sequence, `bf16_add_rank_sum` (when the partials are separate),
//! `glm5next_hc_post` (in place) and `glm5next_hc_pre_norm`.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Fails unless the post-mixed streams, `y`, `post`, `comb` and the normed row are
//!   byte-identical between the arms for every case, on each of three back-to-back folded launches
//!   (the arrival counter must come back to zero), and fails if the reference left an output
//!   unwritten.
//! - A negative control must DIFFER: the folded arm with one `comb` entry of the previous site
//!   changed (the streams and every downstream output).
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_hc_post_fold_gate \
//!       --features cuda,gpu-examples
//!
//! Timing (H 4096, hc 4, one token): each arm captured as a CUDA graph over `REPS` distinct sites
//! and replayed; median, min and max per site.

use anyhow::{Result, bail};
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_post, mhc_mix_max_tokens, mix_hc,
};
use metrale_model_arch::glm5next_mhc_fused::glm_hc_pre_norm;
use metrale_model_arch::glm5next_mhc_post_fold::{PostFoldIo, glm_hc_post_pre_norm};

/// 2026-10-10: `(hidden, hc_mult, partials: Some((world, rank)), or None)`.
/// 2026-10-16: Two-rank cases (the one-shot is offered from world 2): both ranks at the GLM-5.3
/// shape and one narrower width.
const CASES: [(usize, usize, Option<(usize, usize)>); 9] = [
    (4096, 4, Some((3, 0))),
    (4096, 4, Some((3, 1))),
    (4096, 4, Some((3, 2))),
    (4096, 4, None),
    (1024, 2, Some((3, 1))),
    (256, 3, None),
    (4096, 4, Some((2, 0))),
    (4096, 4, Some((2, 1))),
    (1024, 2, Some((2, 1))),
];
const ITERS: u32 = 20;
const EPS: [f32; 3] = [1e-5, 1e-6, 1e-5];
const REPS: usize = 90;
const TRIALS: usize = 9;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * s).collect()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn f32b(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bf16b(d: &[f32]) -> Vec<u8> {
    d.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect()
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-10: One case's host data: the previous site's post/comb, the residual streams, this
/// rank's partial and the peers' (two slots; a two-rank case reads the first), and the next
/// site's weights.
struct Case {
    hid: usize,
    hc: usize,
    post: Vec<f32>,
    comb: Vec<f32>,
    res: Vec<f32>,
    own: Vec<f32>,
    peers: Vec<f32>,
}

/// 2026-10-10: One arm's device buffers: own, residual, streams out, post, comb, y, normed.
struct Arm {
    own: DevicePtr,
    res: DevicePtr,
    out: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    y: DevicePtr,
    normed: DevicePtr,
}

impl Arm {
    fn new(g: &dyn GpuBackend, c: &Case) -> Result<Self> {
        let s = c.hc * c.hid * 4;
        Ok(Self {
            own: up(g, &bf16b(&c.own))?,
            res: up(g, &f32b(&c.res))?,
            out: up(g, &vec![0xABu8; s])?,
            post: up(g, &f32b(&c.post))?,
            comb: up(g, &f32b(&c.comb))?,
            y: up(g, &vec![0xABu8; c.hid * 2])?,
            normed: up(g, &vec![0xABu8; c.hid * 2])?,
        })
    }
    /// 2026-10-10: The five compared outputs; `streams` is where the post-mix landed.
    fn outs(&self, g: &dyn GpuBackend, c: &Case, streams: DevicePtr) -> Result<Vec<Vec<u8>>> {
        Ok(vec![
            dn(g, streams, c.hc * c.hid * 4)?,
            dn(g, self.y, c.hid * 2)?,
            dn(g, self.post, c.hc * 4)?,
            dn(g, self.comb, c.hc * c.hc * 4)?,
            dn(g, self.normed, c.hid * 2)?,
        ])
    }
}

const NAMES: [&str; 5] = ["streams", "y", "post", "comb", "normed"];

fn site(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    hid: usize,
    hc: usize,
) -> Result<Glm5NextMhcSiteWeights> {
    let m = mix_hc(hc);
    Ok(Glm5NextMhcSiteWeights {
        hc_fn: up(g, &bf16b(&rng.vec(m * hc * hid, 0.02)))?,
        hc_fn_bf16: true,
        hc_scale: up(g, &f32b(&[0.7, 1.3, 0.9]))?,
        hc_base: up(g, &f32b(&rng.vec(m, 1.0)))?,
        mix: g.alloc(mhc_mix_max_tokens() * m * 4)?,
        done: up(g, &vec![0u8; mhc_mix_max_tokens() * 4])?,
    })
}

#[allow(clippy::too_many_arguments)]
fn reference(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    sum: KernelHandle,
    w: &Glm5NextMhcSiteWeights,
    norm_w: DevicePtr,
    a: &Arm,
    peers: Option<(DevicePtr, usize, usize)>,
    hid: usize,
    hc: usize,
    s: u64,
) -> Result<()> {
    if let Some((p, world, rank)) = peers {
        KernelLaunch::new(g, sum)
            .grid([hid.div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.own)
            .arg_ptr(p)
            .arg_u64(hid as u64)
            .arg_i32(world as i32)
            .arg_i32(rank as i32)
            .arg_i32(hid as i32)
            .launch(s)?;
    }
    let (h, m) = (hid as u32, hc as u32);
    glm_hc_post(
        g, k.hc_post, a.own, a.res, a.post, a.comb, a.res, 1, h, m, s,
    )?;
    glm_hc_pre_norm(
        g, k, a.res, w, norm_w, a.y, a.normed, a.post, a.comb, 1, h, m, ITERS, EPS[0], EPS[1],
        EPS[2], s,
    )
}

#[allow(clippy::too_many_arguments)]
fn folded(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    norm_w: DevicePtr,
    a: &Arm,
    peers: Option<(DevicePtr, usize, usize)>,
    hid: usize,
    hc: usize,
    s: u64,
) -> Result<()> {
    let partials = peers.map(|(p, world, rank)| PeerPartials {
        recv: p.0,
        slot_elems: hid,
        world_size: world,
        rank,
    });
    glm_hc_post_pre_norm(
        g,
        k,
        w,
        PostFoldIo {
            own: a.own,
            residual: a.res,
            streams_out: a.out,
            post: a.post,
            comb: a.comb,
            norm_w,
            y_out: a.y,
            normed_out: a.normed,
        },
        partials.as_ref(),
        [hid as u32, hc as u32, ITERS],
        EPS,
        s,
    )
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .ok_or_else(|| anyhow::anyhow!("glm-5.3-flash kernel target not built"))?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let k = Glm5NextMhcKernels::resolve(g)?;
    let sum = g.kernel("bf16_add", "bf16_add_rank_sum")?;
    let mut rng = Lcg(0x6c6d_f01d_0001);
    println!("glm5next hc post fold gate: rank sum + hc_post + hc_pre_norm is the ORACLE");
    for (hid, hc, ranks) in CASES {
        let c = Case {
            hid,
            hc,
            post: rng.vec(hc, 1.0).iter().map(|x| 1.0 + 0.5 * x).collect(),
            comb: rng.vec(hc * hc, 0.4).iter().map(|x| 0.3 + x).collect(),
            res: rng.vec(hc * hid, 3.0),
            own: rng.vec(hid, 2.0),
            peers: rng.vec(2 * hid, 2.0),
        };
        let w = site(g, &mut rng, hid, hc)?;
        let norm_w = up(
            g,
            &bf16b(
                &rng.vec(hid, 0.5)
                    .iter()
                    .map(|x| 1.0 + x)
                    .collect::<Vec<_>>(),
            ),
        )?;
        let peers = match ranks {
            Some((world, r)) => Some((up(g, &bf16b(&c.peers))?, world, r)),
            None => None,
        };
        let r = Arm::new(g, &c)?;
        reference(g, &k, sum, &w, norm_w, &r, peers, hid, hc, 0)?;
        let want = r.outs(g, &c, r.res)?;
        for (o, n) in want.iter().zip(NAMES) {
            if o.chunks(4).any(|q| q == [0xAB; 4]) && n != "post" && n != "comb" {
                bail!("H={hid} hc={hc}: the reference left {n} unwritten");
            }
        }
        for rep in 0..3 {
            let f = Arm::new(g, &c)?;
            folded(g, &k, &w, norm_w, &f, peers, hid, hc, 0)?;
            for ((a, b), n) in want.iter().zip(f.outs(g, &c, f.out)?).zip(NAMES) {
                if *a != b {
                    bail!("H={hid} hc={hc} ranks={ranks:?} rep {rep}: {n} differs");
                }
            }
        }
        // 2026-10-10: Negative control: one comb entry of the previous site changed.
        let mut bad = Case { ..c };
        bad.comb[0] += 0.25;
        let f = Arm::new(g, &bad)?;
        folded(g, &k, &w, norm_w, &f, peers, hid, hc, 0)?;
        let got = f.outs(g, &bad, f.out)?;
        if got[0] == want[0] || got[4] == want[4] {
            bail!("H={hid} hc={hc}: the comb control did not change the outputs");
        }
        println!(
            "  H={hid:>5} hc={hc} partials={ranks:?}  streams/y/post/comb/normed BYTE-IDENTICAL x3; control differs"
        );
    }
    timing(g, &k, sum, &mut rng)?;
    println!("post fold gate PASS");
    Ok(())
}

/// 2026-10-10: Per-site microseconds of both arms at H 4096, hc 4, three ranks, one token.
fn timing(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    sum: KernelHandle,
    rng: &mut Lcg,
) -> Result<()> {
    let (hid, hc) = (4096usize, 4usize);
    let c = Case {
        hid,
        hc,
        post: vec![1.0; hc],
        comb: vec![0.25; hc * hc],
        res: rng.vec(hc * hid, 3.0),
        own: rng.vec(hid, 2.0),
        peers: rng.vec(2 * hid, 2.0),
    };
    let sites = (0..REPS)
        .map(|_| site(g, rng, hid, hc))
        .collect::<Result<Vec<_>>>()?;
    let norm_w = up(g, &bf16b(&vec![1.0; hid]))?;
    let peers = Some((up(g, &bf16b(&c.peers))?, 3usize, 1usize));
    let (ra, fa) = (Arm::new(g, &c)?, Arm::new(g, &c)?);
    let s = g.create_stream()?;
    let time = |f: &dyn Fn(usize, u64) -> Result<()>| -> Result<(f64, f64, f64)> {
        g.begin_capture(s)?;
        for i in 0..REPS {
            f(i, s)?;
        }
        let graph = g.end_capture(s)?;
        for _ in 0..3 {
            g.launch_graph(graph, s)?;
        }
        g.synchronize(s)?;
        let mut t = Vec::with_capacity(TRIALS);
        for _ in 0..TRIALS {
            let t0 = std::time::Instant::now();
            g.launch_graph(graph, s)?;
            g.synchronize(s)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
        }
        g.destroy_graph(graph)?;
        t.sort_by(f64::total_cmp);
        Ok((t[TRIALS / 2], t[0], t[TRIALS - 1]))
    };
    let a = time(&|i, s| reference(g, k, sum, &sites[i], norm_w, &ra, peers, hid, hc, s))?;
    // 2026-10-10: The folded arm writes `out`, never its residual, so replays stay valid.
    let b = time(&|i, s| folded(g, k, &sites[i], norm_w, &fa, peers, hid, hc, s))?;
    println!(
        "  TIMING H=4096 hc=4, {REPS} sites, us/site median [min, max]: rank sum + post + \
         pre_norm {:.2} [{:.2}, {:.2}]  folded {:.2} [{:.2}, {:.2}]  saves {:.3} ms/step at {REPS} \
         sites",
        a.0,
        a.1,
        a.2,
        b.0,
        b.1,
        b.2,
        (a.0 - b.0) * REPS as f64 / 1000.0
    );
    Ok(())
}
