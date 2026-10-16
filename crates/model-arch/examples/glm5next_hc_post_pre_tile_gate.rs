// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Byte gate and microbench of `METRALE_GLM_MHC_POST_PRE_TILE`
//! (every `glm5next_mhc_post_tile::ENTRIES` kernel: `glm5next_hc_post_pre_tile1`, `..._tile2`
//! and the persistent rings `..._ring0`, `..._ring1`, `..._ring1e`; the SP-mHC post-mix and the
//! FFN pre-norm in one launch) against the pair it replaces, `glm5next_hc_post_rank_sum_vec8` then
//! `glm5next_hc_pre_norm_tile4`, at GLM-5.3's shape (hidden 4096, hc 4, 20 Sinkhorn iterations).
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Returns an error unless, for every entry point, every byte of `block_out`, the streams,
//!   `post`, `comb`, `y`, `normed` and the `mix` scratch over rows `0..r0 + T + 8` (the rows
//!   before the piece and the poisoned rows after it included) equals the pair's, for:
//!   T in {1, 2, 3, 4, 5, 17, 64, 96, 100, 347, 684, 2736} on every site (two synthetic, both
//!   sites of every packet layer) and two stream distributions, `write_x` on, aliased as the
//!   layer calls it (streams, post and comb updated in place); and on the first site, without
//!   aliasing and with `write_x` off, from row offset 3, at 1 to 4 ranks, split into pieces
//!   (the layer's four-way split and an odd split), and as a continuation chunk after a
//!   prefix-cache hit: the chunk starts at row 61 or 64 and its residual rows are restored from
//!   a cache buffer by an async copy on the launch stream right before each piece, so the
//!   pre-norm reads streams the post-mix formed from restored columns.
//! - Returns an error unless the fused kernel split at row 100 equals the pair over the unsplit
//!   chunk (`mix` aside: it is piece-local).
//! - Returns an error unless each negative control changes the fused kernel's bytes against the
//!   pair's unperturbed run (one peer partial, one `hc_fn` bit, one residual value + 1.0) and
//!   the unperturbed control does not.
//! - The kernels are launched directly, so the lever and the 64-row dispatch floor do not apply.
//!
//! Run (GB10, about 1 GB of device memory; packets from extract_mhc_packets.py; the mix scratch
//! must take 2736 rows):
//!   MHC_PACKET_DIR=/path/to/packets METRALE_GLM_PREFILL_ROWS=4096 cargo run --release \
//!     -p metrale-model-arch --features cuda,gpu-examples --example glm5next_hc_post_pre_tile_gate

#[path = "common/glm5next_hc_post_pre_bufs.rs"]
mod bufs;
#[path = "common/glm5next_mhc_sites.rs"]
mod sites;

use anyhow::{Context, Result, bail, ensure};
use bufs::{Arm, DIMS, EPS, Inputs, PAD, R0MAX, ROW, SLOT, free, inputs, locate};
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_hc_post_vec;
use metrale_model_arch::glm5next_mhc::Glm5NextMhcKernels;
use metrale_model_arch::glm5next_mhc_post_pre_tok::PostPreTok;
use metrale_model_arch::glm5next_mhc_post_tile::{self as post_tile, ENTRIES, Entry};
use metrale_model_arch::glm5next_mhc_tile::launch_tile;
use sites::{H, HC, Rng, Site, all_sites, bf16b, read, up};

/// 2026-10-15: One gate case: rows `r0..r0 + t`, run as `pieces` (offset, rows) in order.
#[derive(Clone, Debug)]
struct Case {
    r0: usize,
    t: usize,
    world: usize,
    rank: usize,
    write_x: bool,
    aliased: bool,
    pieces: Vec<(usize, usize)>,
    restore: bool,
}

impl Case {
    /// 2026-10-15: The layer's call: three ranks, `write_x` on, aliased, one piece.
    fn new(t: usize) -> Self {
        Self {
            r0: 0,
            t,
            world: 3,
            rank: 1,
            write_x: true,
            aliased: true,
            pieces: vec![(0, t)],
            restore: false,
        }
    }
}

/// 2026-10-15: A fused entry point and its kernel.
type Fused = (KernelHandle, Entry);

struct Gate<'a> {
    gpu: &'a dyn GpuBackend,
    k: &'a Glm5NextMhcKernels,
    inp: Inputs,
    want: Arm,
    got: Arm,
    fused: [Fused; ENTRIES.len()],
}

impl Gate<'_> {
    /// 2026-10-15: `c` on `arm`: the pair when `fused` is `None`.
    fn run(&self, arm: &Arm, site: &Site, c: &Case, fused: Option<Fused>) -> Result<()> {
        let w = site.weights(arm.0[6]);
        for &(off, n) in &c.pieces {
            let r = c.r0 + off;
            let at = |p: DevicePtr, i: usize| p.offset(r * ROW[i]);
            if c.restore {
                let (src, dst) = (at(self.inp.cache, 1), at(arm.0[1], 1));
                self.gpu.copy_d2d_async(src, dst, n * ROW[1], 0)?;
            }
            let src = |i: usize| at(if c.aliased { arm.0[i] } else { self.inp.src[i] }, i);
            let partials = PeerPartials {
                recv: at(self.inp.peers, 0).0,
                slot_elems: SLOT,
                world_size: c.world,
                rank: c.rank,
            };
            let (block_out, streams) = (at(arm.0[0], 0), at(arm.0[1], 1));
            let outs = [
                at(arm.0[4], 4),
                at(arm.0[2], 2),
                at(arm.0[3], 3),
                at(arm.0[5], 5),
            ];
            let rows = n as u32;
            match fused {
                None => {
                    glm5next_hc_post_vec::launch(
                        self.gpu,
                        self.k.hc_post_rank_sum_vec8,
                        block_out,
                        &partials,
                        [src(1), src(2), src(3), streams],
                        [rows, H as u32, HC as u32],
                        c.write_x,
                        0,
                    )?;
                    let norm_w = self.inp.norm_w;
                    let k = self.k.hc_pre_norm_tile;
                    launch_tile(self.gpu, k, streams, &w, norm_w, outs, rows, DIMS, EPS, 0)?;
                }
                Some((kernel, entry)) => {
                    let a = PostPreTok {
                        block_out,
                        partials: &partials,
                        residual: src(1),
                        post: src(2),
                        comb: src(3),
                        streams_out: streams,
                        site: &w,
                        norm_w: self.inp.norm_w,
                        y: outs[0],
                        post_out: outs[1],
                        comb_out: outs[2],
                        normed: outs[3],
                        num_tokens: rows,
                        dims: DIMS,
                        eps: EPS,
                    };
                    post_tile::launch(self.gpu, kernel, entry, &a, c.write_x, 0)?;
                }
            }
        }
        Ok(())
    }

    fn pair_bytes(&self, site: &Site, c: &Case, with_mix: bool) -> Result<Vec<u8>> {
        self.want.reset(self.gpu, &self.inp, c.aliased)?;
        self.run(&self.want, site, c, None)?;
        self.want.bytes(self.gpu, c.r0 + c.t + PAD, with_mix)
    }

    fn fused_bytes(&self, site: &Site, c: &Case, f: Fused, with_mix: bool) -> Result<Vec<u8>> {
        self.got.reset(self.gpu, &self.inp, c.aliased)?;
        self.run(&self.got, site, c, Some(f))?;
        self.got.bytes(self.gpu, c.r0 + c.t + PAD, with_mix)
    }

    /// 2026-10-15: Every entry point against the pair on `c`.
    fn check(&self, site: &Site, label: &str, c: &Case) -> Result<()> {
        let want = self.pair_bytes(site, c, true)?;
        for f in self.fused {
            let got = self.fused_bytes(site, c, f, true)?;
            if got != want {
                let i = want.iter().zip(&got).position(|(a, b)| a != b).unwrap_or(0);
                bail!(
                    "{} / {} / {label} {c:?}: bytes differ from the pair, first at {}",
                    f.1.name(),
                    site.name,
                    locate(i, c.r0 + c.t + PAD)
                );
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let dir = std::env::var("MHC_PACKET_DIR")
        .context("MHC_PACKET_DIR (the checkpoint mHC packets) is required")?;
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Glm5NextMhcKernels::resolve(&gpu)?;
    let fused = ENTRIES.map(|e| {
        let h = metrale_model_layers::layers::try_kernel(&gpu, e.module(), e.name());
        (h, e)
    });
    for (h, e) in fused {
        ensure!(h.0 != 0, "{} is not in this build", e.name());
    }
    ensure!(
        k.hc_post_rank_sum_vec8.0 != 0 && k.hc_pre_norm_tile.0 != 0,
        "the pair (vec8 + tile4) is not in this build"
    );
    let mut rng = Rng(0x5be0_cd19_137e_2179);
    let norm: Vec<f32> = (0..H).map(|_| 1.0 + 0.2 * rng.unit()).collect();
    let norm_w = up(&gpu, &bf16b(&norm))?;
    let sites = all_sites(&gpu, &mut rng, &dir)?;
    let mut g = Gate {
        gpu: &gpu,
        k: &k,
        inp: inputs(&gpu, &mut rng, 2.0, norm_w)?,
        want: Arm::alloc(&gpu)?,
        got: Arm::alloc(&gpu)?,
        fused,
    };
    let dists: [(&str, f32); 2] = [("streams 2", 2.0), ("streams 50 + outliers", 50.0)];
    for (i, (dname, mag)) in dists.into_iter().enumerate() {
        if i > 0 {
            free(&gpu, &g.inp)?;
            g.inp = inputs(&gpu, &mut rng, mag, norm_w)?;
        }
        for site in &sites {
            for t in [1, 2, 3, 4, 5, 17, 64, 96, 100, 347, 684, 2736] {
                g.check(site, dname, &Case::new(t))?;
            }
            println!("{} / {dname}: identical at every T", site.name);
        }
        variants(&g, &sites[0], dname)?;
    }
    split_vs_unsplit(&g, &sites[0])?;
    negative_controls(&g, &sites[2])?;
    println!("BYTE GATE: PASS");
    for t in [64usize, 347, 684, 2736] {
        let c = Case {
            write_x: false,
            aliased: false,
            ..Case::new(t)
        };
        let mut line = format!("T={t:5}:");
        let mut pair = 0.0;
        let arms = std::iter::once(("pair", None));
        for (name, f) in arms.chain(fused.map(|f| (f.1.name(), Some(f)))) {
            let m = median_us(&g, &sites[2], &c, f)?;
            if f.is_none() {
                pair = m[0];
            }
            line += &format!(" {name} {:7.1} us [{:.1}, {:.1}]", m[0], m[1], m[2]);
            if f.is_some() {
                line += &format!(" ({:.2}x)", pair / m[0]);
            }
        }
        println!("{line} (median [min, max], n=9 x 5 launches, clean)");
    }
    free(&gpu, &g.inp)?;
    Ok(())
}

/// 2026-10-15: The first site's other call shapes: aliasing, `write_x`, a row offset, rank
/// counts, piece splits and the prefix-hit continuation.
fn variants(g: &Gate, site: &Site, dname: &str) -> Result<()> {
    for t in [1, 5, 17, 684, 2736] {
        for (write_x, aliased) in [(false, true), (true, false), (false, false)] {
            let c = Case {
                write_x,
                aliased,
                ..Case::new(t)
            };
            g.check(site, dname, &c)?;
        }
    }
    for t in [17, 684] {
        g.check(
            site,
            dname,
            &Case {
                r0: 3,
                ..Case::new(t)
            },
        )?;
    }
    for (world, rank) in [(1, 0), (2, 0), (2, 1), (3, 0), (3, 2), (4, 0), (4, 3)] {
        for t in [5, 684] {
            let c = Case {
                world,
                rank,
                ..Case::new(t)
            };
            g.check(site, dname, &c)?;
        }
    }
    let quarters = (0..4).map(|q| (q * 171, 171)).collect();
    let splits = [
        Case {
            write_x: false,
            pieces: quarters,
            ..Case::new(684)
        },
        Case {
            aliased: false,
            pieces: vec![(0, 341), (341, 343)],
            ..Case::new(684)
        },
        Case {
            r0: 61,
            write_x: false,
            pieces: vec![(0, 341), (341, 343)],
            restore: true,
            ..Case::new(684)
        },
        Case {
            r0: R0MAX,
            restore: true,
            ..Case::new(347)
        },
    ];
    for c in &splits {
        g.check(site, dname, c)?;
    }
    println!("{} / {dname}: identical in every call shape", site.name);
    Ok(())
}

/// 2026-10-15: The fused kernel over two pieces equals the pair over the whole chunk.
fn split_vs_unsplit(g: &Gate, site: &Site) -> Result<()> {
    let whole = Case::new(684);
    let want = g.pair_bytes(site, &whole, false)?;
    let split = Case {
        pieces: vec![(0, 100), (100, 584)],
        ..whole
    };
    for f in g.fused {
        let got = g.fused_bytes(site, &split, f, false)?;
        ensure!(
            got == want,
            "{}: split at row 100 differs from the pair over the whole chunk",
            f.1.name()
        );
    }
    println!("split vs unsplit: identical");
    Ok(())
}

/// 2026-10-15: Each perturbation must change the fused bytes against the pair's unperturbed
/// run; the unperturbed fused run must not.
fn negative_controls(g: &Gate, site: &Site) -> Result<()> {
    let c = Case::new(684);
    let want = g.pair_bytes(site, &c, true)?;
    let last = c.t - 1;
    // 2026-10-15: Rank 1 of 3 reads peer slots 0 and 1; slot 1, last row, column 77.
    let peer = g.inp.peers.offset((SLOT + last * H + 77) * 2);
    let resid = g.inp.src[1].offset((last * HC * H + 3 * H + 9) * 4);
    for f in g.fused {
        let name = f.1.name();
        ensure!(
            g.fused_bytes(site, &c, f, true)? == want,
            "{name}: control differs"
        );
        let orig = read(g.gpu, peer, 2)?;
        g.gpu.copy_h2d(&[orig[0], orig[1] ^ 0x01], peer)?;
        let fired = g.fused_bytes(site, &c, f, true)? != want;
        g.gpu.copy_h2d(&orig, peer)?;
        ensure!(
            fired,
            "{name}: negative control (peer partial bit) did not fire"
        );
        let mut flipped = site.hc_fn.clone();
        flipped[2 * (23 * HC * H + 5) + 1] ^= 0x01;
        g.gpu.copy_h2d(&flipped, site.hc_fn_d)?;
        let fired = g.fused_bytes(site, &c, f, true)? != want;
        g.gpu.copy_h2d(&site.hc_fn, site.hc_fn_d)?;
        ensure!(
            fired,
            "{name}: negative control (hc_fn bit flip) did not fire"
        );
        let orig = read(g.gpu, resid, 4)?;
        let x = f32::from_le_bytes(orig[..4].try_into()?) + 1.0;
        g.gpu.copy_h2d(&x.to_le_bytes(), resid)?;
        let fired = g.fused_bytes(site, &c, f, true)? != want;
        g.gpu.copy_h2d(&orig, resid)?;
        ensure!(
            fired,
            "{name}: negative control (residual + 1.0) did not fire"
        );
        println!("{name}: negative controls (peer bit, hc_fn bit, residual + 1.0) FIRED");
    }
    Ok(())
}

/// 2026-10-15: Median, min and max microseconds per launch of `c` over 9 x 5 launches.
fn median_us(g: &Gate, site: &Site, c: &Case, f: Option<Fused>) -> Result<[f64; 3]> {
    let arm = if f.is_some() { &g.got } else { &g.want };
    arm.reset(g.gpu, &g.inp, c.aliased)?;
    g.run(arm, site, c, f)?;
    let mut v = Vec::new();
    for _ in 0..9 {
        g.gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            g.run(arm, site, c, f)?;
        }
        g.gpu.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() * 2e5);
    }
    v.sort_by(f64::total_cmp);
    Ok([v[4], v[0], v[8]])
}
