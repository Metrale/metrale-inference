// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of `METRALE_GLM_MHC_POST_PRE_FUSED`: the SP-mHC layer's
//! attention-site reduce + post-mix (`glm5next_hc_post_rank_sum`) followed by the FFN site's
//! pre-norm (`glm_hc_pre_norm_with`: the 8-row grid, or `glm5next_hc_pre_norm_tok4`) against
//! `glm5next_hc_post_pre_norm_tok4` (one launch, through the production
//! `glm5next_mhc_post_pre_tok::launch`), at GLM-5.3's shape (hidden 4096, hc 4, 20 Sinkhorn
//! iterations), three ranks (this one rank 1), in place as the layer launches it: the new
//! streams over the old, the FFN site's post/comb over the attention site's.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Exits with an error unless every output (the reduced row `block_out`, the streams, `y`,
//!   `normed`, `post`, `comb` and the `mix` scratch) of the fused launch equals both pairs' byte
//!   for byte, at 64, 67, 2736, 2737 and 8208 rows (full and partial last blocks of four). The
//!   inputs carry spikes (x64 stream values, x32 partials) beside the random ones.
//! - Exits with an error unless the negative control (one peer partial of the last row changed)
//!   changes the fused bytes.
//! - Times each arm as a captured CUDA graph, median [min..max] (`common/graph_timing.rs`), and
//!   prints GB/s over the fused kernel's compulsory bytes (the same numerator for every arm).
//!
//! Run (GB10, about 1.7 GB of device memory; the `mix` scratch bound needs
//! `METRALE_GLM_PREFILL_ROWS` of at least 8208):
//!   METRALE_GLM_PREFILL_ROWS=8208 METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash \
//!   METRALE_TARGET_QUANT=nvfp4 cargo run -p metrale-gpu-examples --release \
//!       --example glm5next_hc_post_pre_tok_bench --features cuda,gpu-examples

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
use metrale_model_arch::glm5next_mhc_fused::{glm_hc_post_rank_sum, glm_hc_pre_norm_with};
use metrale_model_arch::glm5next_mhc_post_pre_tok::{PostPreTok, kernel_for, launch};

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;
const ITERS: u32 = 20;
/// 2026-10-14: `[norm_eps, hc_eps, rms_eps]` as the layer passes them (rms, hc, rms).
const EPS: [f32; 3] = [1e-5, 1e-6, 1e-5];
const WORLD: usize = 3;
const RANK: usize = 1;
/// 2026-10-14: The narrowest fused piece, partial last blocks, an SP piece of a 8208-row
/// sub-chunk (8208 / 3) and the whole sub-chunk.
const GATE_ROWS: [usize; 5] = [64, 67, 2736, 2737, 8208];
const TIMED_ROWS: [usize; 3] = [64, 2736, 8208];
const TMAX: usize = 8208;
/// 2026-10-14: Peer slot = the widest piece's partial, as the one-shot receive buffer lays it out.
const SLOT: usize = TMAX * H;

/// 2026-10-14: Host inputs at `TMAX` rows; a run of `t` rows uses the leading rows.
struct Host {
    own: Vec<u8>,
    streams: Vec<u8>,
    post: Vec<u8>,
    comb: Vec<u8>,
}

/// 2026-10-14: One arm's device buffers. Each arm owns its `mix`/`done` scratch (`site`), so the
/// scratch is compared too.
struct Arm {
    own: DevicePtr,
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    y: DevicePtr,
    normed: DevicePtr,
    site: Glm5NextMhcSiteWeights,
}

const OUT_NAMES: [&str; 7] = ["block_out", "streams", "y", "normed", "post", "comb", "mix"];

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
                done: up(g, &vec![0u8; TMAX * 4])?,
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

    /// 2026-10-14: The inputs of `t` rows; every output buffer poisoned with 0xAB.
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

fn host(rng: &mut Lcg) -> Host {
    let mut own = rng.vec(TMAX * H, 2.0);
    for v in own.iter_mut().step_by(1031) {
        *v *= 32.0;
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
        own: own
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        streams: f32b(&streams),
        post: f32b(&post),
        comb: f32b(&comb),
    }
}

struct Bench<'a> {
    g: &'a dyn GpuBackend,
    k: Glm5NextMhcKernels,
    peers: PeerPartials,
    norm_w: DevicePtr,
}

impl Bench<'_> {
    /// 2026-10-14: The unfused pair the SP-mHC body runs; `tok` picks the pre-norm grid.
    fn pair(&self, a: &Arm, t: usize, tok: bool, s: u64) -> Result<()> {
        let (tt, ht, hct) = (t as u32, H as u32, HC as u32);
        glm_hc_post_rank_sum(
            self.g,
            self.k.hc_post_rank_sum,
            a.own,
            &self.peers,
            a.streams,
            a.post,
            a.comb,
            a.streams,
            tt,
            ht,
            hct,
            s,
        )?;
        let [norm_eps, hc_eps, rms_eps] = EPS;
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
            false,
            false,
            tok,
            s,
        )
    }

    fn fused(&self, kernel: KernelHandle, a: &Arm, t: usize, s: u64) -> Result<()> {
        let args = PostPreTok {
            block_out: a.own,
            partials: &self.peers,
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

/// 2026-10-14: The names of the outputs that differ (empty when every byte matches).
fn diff(want: &[Vec<u8>], got: &[Vec<u8>]) -> Vec<String> {
    OUT_NAMES
        .iter()
        .zip(want.iter().zip(got))
        .filter(|(_, (w, g))| w != g)
        .map(|(n, (w, g))| {
            let at = w
                .iter()
                .zip(g.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            format!("{n}@{at}")
        })
        .collect()
}

/// 2026-10-14: Bytes the fused launch must move at `t` rows: every partial, the old streams,
/// post and comb in; the reduced row, the new streams, y, normed, post, comb and mix out.
fn compulsory_bytes(t: usize) -> f64 {
    let rows_in = WORLD * t * H * 2 + t * HC * H * 4 + t * (HC + HC * HC) * 4;
    let rows_out = t * H * 2 + t * HC * H * 4 + 2 * t * H * 2 + t * (HC + HC * HC + MIX) * 4;
    (rows_in + rows_out) as f64
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Glm5NextMhcKernels::resolve(g)?;
    ensure!(
        k.hc_post_rank_sum.0 != 0,
        "glm5next_hc_post_rank_sum is not in this build"
    );
    ensure!(
        k.hc_pre_norm_tok.0 != 0,
        "glm5next_hc_pre_norm_tok4 is not in this build"
    );
    let mut rng = Lcg(0x6d68_6366_7573_6564);
    let hc_fn = rng.vec(MIX * HC * H, 0.05);
    let base = rng.vec(MIX, 0.5);
    let norm = rng.vec(H, 0.2).iter().map(|v| 1.0 + v).collect::<Vec<_>>();
    let mut peers_h = rng.vec((WORLD - 1) * SLOT, 2.0);
    for v in peers_h.iter_mut().step_by(2053) {
        *v *= 32.0;
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
        peers: PeerPartials {
            recv: peers.0,
            slot_elems: SLOT,
            world_size: WORLD,
            rank: RANK,
        },
        norm_w: up_bf16(g, &norm)?,
        k,
    };
    let h = host(&mut rng);
    let (a1, a2) = (Arm::alloc(g, &shared)?, Arm::alloc(g, &shared)?);

    println!("glm5next hc post+pre fused gate: H={H} hc={HC} world={WORLD} rank={RANK}");
    let mut failed = false;
    for t in GATE_ROWS {
        let kernel = kernel_for(&b.k, true, t as u32)?.expect("every gate width is fused");
        a1.load(g, &h, t)?;
        b.pair(&a1, t, false, 0)?;
        let rows8 = a1.outs(g, t)?;
        a1.load(g, &h, t)?;
        b.pair(&a1, t, true, 0)?;
        let tok4 = a1.outs(g, t)?;
        a2.load(g, &h, t)?;
        b.fused(kernel, &a2, t, 0)?;
        let fused = a2.outs(g, t)?;
        // 2026-10-14: Control: one partial of the last row's column 5 in peer slot 0 changed.
        let e = (t - 1) * H + 5;
        let old = bf16::from_f32(peers_h[e]).to_le_bytes();
        let new = bf16::from_f32(peers_h[e] + 8.0).to_le_bytes();
        g.copy_h2d(&new, peers.offset(e * 2))?;
        a2.load(g, &h, t)?;
        b.fused(kernel, &a2, t, 0)?;
        let control = a2.outs(g, t)?;
        g.copy_h2d(&old, peers.offset(e * 2))?;
        let (vs8, vs4) = (diff(&rows8, &fused), diff(&tok4, &fused));
        let control_moves = !diff(&rows8, &control).is_empty();
        println!(
            "  T={t:5}: fused vs pair(8-row grid) {} | vs pair(tok4) {} | control {}",
            if vs8.is_empty() {
                "SAME".into()
            } else {
                format!("DIFF {vs8:?}")
            },
            if vs4.is_empty() {
                "SAME".into()
            } else {
                format!("DIFF {vs4:?}")
            },
            if control_moves {
                "differs (ok)"
            } else {
                "SAME (BAD)"
            },
        );
        failed |= !vs8.is_empty() || !vs4.is_empty() || !control_moves;
    }
    if failed {
        bail!("glm5next_hc_post_pre_norm_tok4 gate FAILED");
    }

    let s = g.create_stream()?;
    println!(
        "\n  TIMING (graph of copies, median [min..max] ms/launch; GB/s over compulsory bytes)"
    );
    for t in TIMED_ROWS {
        let kernel = kernel_for(&b.k, true, t as u32)?.expect("every timed width is fused");
        let copies = if t >= 2048 { 4 } else { 32 };
        a1.load(g, &h, t)?;
        a2.load(g, &h, t)?;
        let mut arms: Vec<(&str, (f64, f64, f64))> = Vec::new();
        for (name, tok) in [("pair(8-row)", false), ("pair(tok4)", true)] {
            let us = timing::time_graph(g, s, copies, &mut |st| b.pair(&a1, t, tok, st))?;
            arms.push((name, us));
        }
        let us = timing::time_graph(g, s, copies, &mut |st| b.fused(kernel, &a2, t, st))?;
        arms.push(("fused", us));
        let bytes = compulsory_bytes(t);
        for (name, (med, lo, hi)) in &arms {
            println!(
                "    T={t:5} {name:<12} {:8.3} ms [{:.3}..{:.3}]  {:7.1} GB/s",
                med / 1e3,
                lo / 1e3,
                hi / 1e3,
                bytes / (med * 1e3)
            );
        }
        let fused_ms = arms[2].1.0;
        println!(
            "    T={t:5} saved vs pair(8-row) {:.3} ms, vs pair(tok4) {:.3} ms",
            (arms[0].1.0 - fused_ms) / 1e3,
            (arms[1].1.0 - fused_ms) / 1e3
        );
    }
    println!("\nPASS: fused bytes equal both pairs' at every width; the control differs");
    Ok(())
}
