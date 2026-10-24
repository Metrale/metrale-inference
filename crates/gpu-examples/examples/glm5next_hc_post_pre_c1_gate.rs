// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Byte gate and microbench of `--glm-mhc-post-pre` (R3): the row forward's
//! attention-site reduce + post-mix (`glm5next_hc_post_rank_sum`) followed by the FFN site's c1
//! pre-norm (`glm5next_hc_pre_norm_c1`) against `glm5next_hc_post_pre_norm_c1` (one launch, through
//! the production `glm5next_mhc_post_pre_c1::launch`), at GLM-5.3's shape (hidden 4096, hc 4, 20
//! Sinkhorn iterations), in place as the layer launches it.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Exits with an error unless every output (the summed row `block_out`, the streams, `y`,
//!   `normed`, `post`, `comb` and the `mix` scratch) of the fused launch equals the pair's byte for
//!   byte at 1 and 2 rows, for every rank of worlds 1, 2, 3, 4 and 8. The inputs carry spikes
//!   (x64 stream values, x32 partials), signed zeros and subnormal partials.
//! - Exits with an error unless the negative control (one peer partial of the last row changed)
//!   changes the fused bytes, at every world above 1.
//! - `sabotage` as the first argument passes the fused launch the next rank's index: the gate must
//!   then fail (proof that it can).
//! - Times both arms at TP=3 as captured CUDA graphs of 45 copies (one per layer and step),
//!   median [min..max] (`common/graph_timing.rs`).
//!
//! Run (GB10):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_hc_post_pre_c1_gate \
//!       --features cuda,gpu-examples [-- sabotage]

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, bail, ensure};
use fixture::{Lcg, read, up, up_bf16, up_f32};
use half::bf16;
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
use metrale_model_arch::glm5next_mhc_fused::{
    PreNormChoice, glm_hc_post_rank_sum, glm_hc_pre_norm_with,
};
use metrale_model_arch::glm5next_mhc_post_pre_c1::{kernel_for, launch};
use metrale_model_arch::glm5next_mhc_post_pre_tok::PostPreTok;

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;
const ITERS: u32 = 20;
/// 2026-10-24: `[norm_eps, hc_eps, rms_eps]` as the layer passes them (rms, hc, rms).
const EPS: [f32; 3] = [1e-5, 1e-6, 1e-5];
const WORLDS: [usize; 5] = [1, 2, 3, 4, 8];
const TMAX: usize = 2;
const SLOT: usize = TMAX * H;
const OUT_NAMES: [&str; 7] = ["block_out", "streams", "y", "normed", "post", "comb", "mix"];

/// 2026-10-24: Host inputs at `TMAX` rows.
struct Host {
    own: Vec<u8>,
    streams: Vec<u8>,
    post: Vec<u8>,
    comb: Vec<u8>,
}

/// 2026-10-24: One arm's device buffers, its own `mix`/`done` scratch included.
struct Arm {
    own: DevicePtr,
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    y: DevicePtr,
    normed: DevicePtr,
    site: Glm5NextMhcSiteWeights,
}

impl Arm {
    fn alloc(g: &dyn GpuBackend, shared: &Glm5NextMhcSiteWeights) -> Result<Self> {
        Ok(Self {
            own: g.alloc(TMAX * H * 2)?,
            streams: g.alloc(TMAX * HC * H * 4)?,
            post: g.alloc(TMAX * HC * 4)?,
            comb: g.alloc(TMAX * HC * HC * 4)?,
            y: g.alloc(TMAX * H * 2)?,
            normed: g.alloc(TMAX * H * 2)?,
            site: Glm5NextMhcSiteWeights {
                mix: g.alloc(TMAX * MIX * 4)?,
                done: up(g, &[0u8; TMAX * 4])?,
                mix_tokens: TMAX,
                ..*shared
            },
        })
    }

    fn sizes(t: usize) -> [usize; 7] {
        [
            t * H * 2,
            t * HC * H * 4,
            t * H * 2,
            t * H * 2,
            t * HC * 4,
            t * HC * HC * 4,
            t * MIX * 4,
        ]
    }

    fn ptrs(&self) -> [DevicePtr; 7] {
        [
            self.own,
            self.streams,
            self.y,
            self.normed,
            self.post,
            self.comb,
            self.site.mix,
        ]
    }

    /// 2026-10-24: The inputs of `t` rows; the other outputs poisoned with 0xAB.
    fn load(&self, g: &dyn GpuBackend, h: &Host, t: usize) -> Result<()> {
        let [own, streams, _, _, post, comb, _] = Self::sizes(t);
        g.copy_h2d(&h.own[..own], self.own)?;
        g.copy_h2d(&h.streams[..streams], self.streams)?;
        g.copy_h2d(&h.post[..post], self.post)?;
        g.copy_h2d(&h.comb[..comb], self.comb)?;
        for (p, n) in [
            (self.y, t * H * 2),
            (self.normed, t * H * 2),
            (self.site.mix, t * MIX * 4),
        ] {
            g.copy_h2d(&vec![0xABu8; n], p)?;
        }
        g.synchronize(0)
    }

    fn outs(&self, g: &dyn GpuBackend, t: usize) -> Result<Vec<Vec<u8>>> {
        self.ptrs()
            .iter()
            .zip(Self::sizes(t))
            .map(|(p, n)| read(g, *p, n))
            .collect()
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect()
}

fn host(rng: &mut Lcg) -> Host {
    let mut own = rng.vec(TMAX * H, 2.0);
    for (i, v) in own.iter_mut().enumerate() {
        match i % 1031 {
            0 => *v *= 32.0,
            7 => *v = -0.0,
            _ => {}
        }
    }
    let mut streams = rng.vec(TMAX * HC * H, 3.0);
    for v in streams.iter_mut().step_by(4099) {
        *v *= 64.0;
    }
    let post: Vec<f32> = rng.vec(TMAX * HC, 1.0).iter().map(|v| 1.0 + v).collect();
    let comb: Vec<f32> = rng
        .vec(TMAX * HC * HC, 0.5)
        .iter()
        .map(|v| 0.5 + v)
        .collect();
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    Host {
        own: bf16_bytes(&own),
        streams: f32b(&streams),
        post: f32b(&post),
        comb: f32b(&comb),
    }
}

struct Bench<'a> {
    g: &'a dyn GpuBackend,
    k: Glm5NextMhcKernels,
    norm_w: DevicePtr,
}

impl Bench<'_> {
    /// 2026-10-24: The pair the row forward runs: the post-sum, then the c1 pre-norm.
    fn pair(&self, a: &Arm, p: &PeerPartials, t: usize, s: u64) -> Result<()> {
        let (tt, ht, hct) = (t as u32, H as u32, HC as u32);
        glm_hc_post_rank_sum(
            self.g, &self.k, a.own, p, a.streams, a.post, a.comb, a.streams, tt, ht, hct, s,
        )?;
        let [norm_eps, hc_eps, rms_eps] = EPS;
        let c1 = PreNormChoice {
            prefetch: false,
            splitk: false,
            tok: false,
            c1: true,
        };
        glm_hc_pre_norm_with(
            self.g,
            &self.k,
            a.streams,
            &a.site,
            self.norm_w,
            a.y,
            a.normed,
            a.post,
            a.comb,
            tt,
            ht,
            hct,
            ITERS,
            norm_eps,
            hc_eps,
            rms_eps,
            c1,
            s,
        )
    }

    fn fused(
        &self,
        kernel: KernelHandle,
        a: &Arm,
        p: &PeerPartials,
        t: usize,
        s: u64,
    ) -> Result<()> {
        let args = PostPreTok {
            block_out: a.own,
            partials: p,
            residual: a.streams,
            post: a.post,
            comb: a.comb,
            streams_out: a.streams,
            site: &a.site,
            norm_w: self.norm_w,
            y: a.y,
            post_out: a.post,
            comb_out: a.comb,
            normed: a.normed,
            num_tokens: t as u32,
            dims: [H as u32, HC as u32, ITERS],
            eps: EPS,
        };
        launch(self.g, kernel, &args, s)
    }
}

/// 2026-10-24: The names of the outputs that differ, each with its first differing byte.
fn diff(want: &[Vec<u8>], got: &[Vec<u8>]) -> Vec<String> {
    OUT_NAMES
        .iter()
        .zip(want.iter().zip(got))
        .filter(|(_, (w, g))| w != g)
        .map(|(n, (w, g))| {
            format!(
                "{n}@{}",
                w.iter()
                    .zip(g.iter())
                    .position(|(x, y)| x != y)
                    .unwrap_or(0)
            )
        })
        .collect()
}

fn main() -> Result<()> {
    let sabotage = std::env::args().nth(1).as_deref() == Some("sabotage");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Glm5NextMhcKernels::resolve(g, metrale_gpu_examples::glm_levers()?.mhc)?;
    ensure!(
        k.hc_post_rank_sum.0 != 0,
        "glm5next_hc_post_rank_sum is not in this build"
    );
    let kernel = kernel_for(&k, [true; 3], [2, H as u32, HC as u32])?
        .expect("the declared levers take c1 at two rows");
    let mut rng = Lcg(0x6d68_6331_7070_6331);
    let hc_fn = rng.vec(MIX * HC * H, 0.05);
    let base = rng.vec(MIX, 0.5);
    let norm = rng.vec(H, 0.2).iter().map(|v| 1.0 + v).collect::<Vec<_>>();
    let mut peers_h = rng.vec((WORLDS[4] - 1) * SLOT, 2.0);
    for (i, v) in peers_h.iter_mut().enumerate() {
        match i % 2053 {
            0 => *v *= 32.0,
            3 => *v = -0.0,
            5 => *v = 1.0e-39,
            _ => {}
        }
    }
    let shared = Glm5NextMhcSiteWeights {
        hc_fn: up_bf16(g, &hc_fn)?,
        hc_fn_bf16: true,
        hc_scale: up_f32(g, &[0.7, 1.3, 0.9])?,
        hc_base: up_f32(g, &base)?,
        mix: DevicePtr(0),
        done: DevicePtr(0),
        mix_tokens: 0,
    };
    let peers = up_bf16(g, &peers_h)?;
    let b = Bench {
        g,
        norm_w: up_bf16(g, &norm)?,
        k,
    };
    let h = host(&mut rng);
    let (a1, a2) = (Arm::alloc(g, &shared)?, Arm::alloc(g, &shared)?);
    let at = |world: usize, rank: usize| PeerPartials {
        recv: peers.0,
        slot_elems: SLOT,
        world_size: world,
        rank,
    };

    println!(
        "glm5next hc post-sum + c1 pre-norm fused gate: H={H} hc={HC}{}",
        if sabotage { " (SABOTAGE)" } else { "" }
    );
    let mut failed = false;
    for world in WORLDS {
        for rank in 0..world {
            for t in 1..=TMAX {
                let p = at(world, rank);
                let pf = if sabotage {
                    at(world, (rank + 1) % world)
                } else {
                    at(world, rank)
                };
                a1.load(g, &h, t)?;
                b.pair(&a1, &p, t, 0)?;
                let pair = a1.outs(g, t)?;
                a2.load(g, &h, t)?;
                b.fused(kernel, &a2, &pf, t, 0)?;
                let d = diff(&pair, &a2.outs(g, t)?);
                // 2026-10-24: Control: one partial of the last row's column 5 in peer slot 0.
                let control = if world > 1 {
                    let e = (t - 1) * H + 5;
                    let old = bf16::from_f32(peers_h[e]).to_le_bytes();
                    g.copy_h2d(
                        &bf16::from_f32(peers_h[e] + 8.0).to_le_bytes(),
                        peers.offset(e * 2),
                    )?;
                    a2.load(g, &h, t)?;
                    b.fused(kernel, &a2, &pf, t, 0)?;
                    let c = a2.outs(g, t)?;
                    g.copy_h2d(&old, peers.offset(e * 2))?;
                    Some(!diff(&pair, &c).is_empty())
                } else {
                    None
                };
                let bad = !d.is_empty() || control == Some(false);
                if bad || (rank == 0 && t == TMAX) {
                    println!(
                        "  world={world} rank={rank} T={t}: {} | control {}",
                        if d.is_empty() {
                            "SAME".into()
                        } else {
                            format!("DIFF {d:?}")
                        },
                        match control {
                            Some(true) => "differs (ok)",
                            Some(false) => "SAME (BAD)",
                            None => "n/a (one rank)",
                        }
                    );
                }
                failed |= bad;
            }
        }
    }
    if failed {
        bail!("glm5next_hc_post_pre_norm_c1 gate FAILED");
    }

    let s = g.create_stream()?;
    let p = at(3, 1);
    println!("\n  TIMING at TP=3 rank 1 (graph of 45 copies, median [min..max] us/launch)");
    for t in 1..=TMAX {
        a1.load(g, &h, t)?;
        a2.load(g, &h, t)?;
        let pair = timing::time_graph(g, s, 45, &mut |st| b.pair(&a1, &p, t, st))?;
        let fused = timing::time_graph(g, s, 45, &mut |st| b.fused(kernel, &a2, &p, t, st))?;
        timing::report("post-sum + c1 pre-norm", t, pair, fused);
    }
    println!(
        "\nPASS: fused bytes equal the pair's at every world, rank and width; the control differs"
    );
    Ok(())
}
