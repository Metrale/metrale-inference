// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Byte gate and microbench of `METRALE_GLM_MHC_PRE_TILE`
//! (`glm5next_hc_pre_norm_tile4`, the cp.async-ring pre-norm) against
//! `glm5next_hc_pre_norm_tok4`, whose bytes `glm5next_mhc_slice_bench` ties to the 8-row grid, at
//! GLM-5.3's shape (hidden 4096, hc 4, 20 Sinkhorn iterations).
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Returns an error unless every byte of y, normed, post, comb and the `mix` scratch, and the
//!   poisoned rows past the last token, equals tok4's at T in {1, 2, 3, 4, 5, 17, 684, 2736}
//!   (17 and 684 also from a stream row offset of 3), for two synthetic weight and stream
//!   distributions and for the checkpoint's `hc_{attn,ffn}_{fn,base,scale}` of every packet.
//! - Returns an error unless both negative controls change the tile kernel's bytes: the last
//!   token's stream value + 1.0, and one flipped bit of `hc_fn`.
//! - The kernels are launched directly (`launch_tok`, `launch_tile`), so the lever and the
//!   64-token dispatch floor do not apply here.
//!
//! Run (GB10, < 1 GB of device memory; packets from extract_mhc_packets.py):
//!   MHC_PACKET_DIR=/path/to/packets cargo run --release -p metrale-gpu-examples \
//!     --features cuda,gpu-examples --example glm5next_mhc_tile_gate

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
use metrale_model_arch::glm5next_mhc_tile::{HC_PRE_TILE_KERNEL, launch_tile};
use metrale_model_arch::glm5next_mhc_tok::launch_tok;

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;
const TMAX: usize = 2736 + 3;
const PAD: usize = 8;
const LAYERS: [usize; 4] = [0, 3, 22, 44];

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-15: One arm's outputs (y, normed, post, comb) and its `mix` scratch.
struct Out([DevicePtr; 5]);

fn sizes(t: usize) -> [usize; 5] {
    [
        t * H * 2,
        t * H * 2,
        t * HC * 4,
        t * HC * HC * 4,
        t * MIX * 4,
    ]
}

impl Out {
    fn alloc(gpu: &dyn GpuBackend) -> Result<Self> {
        let s = sizes(TMAX + PAD);
        Ok(Self([
            gpu.alloc(s[0])?,
            gpu.alloc(s[1])?,
            gpu.alloc(s[2])?,
            gpu.alloc(s[3])?,
            gpu.alloc(s[4])?,
        ]))
    }
    fn poison(&self, gpu: &dyn GpuBackend, t: usize) -> Result<()> {
        for (p, n) in self.0.iter().zip(sizes(t + PAD)) {
            gpu.copy_h2d(&vec![0xABu8; n], *p)?;
        }
        Ok(())
    }
    fn bytes(&self, gpu: &dyn GpuBackend, t: usize) -> Result<Vec<u8>> {
        let mut v = Vec::new();
        for (p, n) in self.0.iter().zip(sizes(t + PAD)) {
            v.extend(read(gpu, *p, n)?);
        }
        Ok(v)
    }
}

/// 2026-10-15: One site's weights on the device: `hc_fn` (BF16 bytes), `hc_base`, `hc_scale`.
struct Site {
    name: String,
    hc_fn: Vec<u8>,
    hc_fn_d: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
}

impl Site {
    fn new(
        gpu: &dyn GpuBackend,
        name: String,
        hc_fn: Vec<u8>,
        base: &[f32],
        scale: &[f32],
    ) -> Result<Self> {
        ensure!(hc_fn.len() == MIX * HC * H * 2 && base.len() == MIX && scale.len() == 3);
        Ok(Self {
            name,
            hc_fn_d: up(gpu, &hc_fn)?,
            hc_fn,
            hc_scale: up(gpu, &f32b(scale))?,
            hc_base: up(gpu, &f32b(base))?,
        })
    }
    fn weights(&self, o: &Out) -> Glm5NextMhcSiteWeights {
        Glm5NextMhcSiteWeights {
            hc_fn: self.hc_fn_d,
            hc_fn_bf16: true,
            hc_scale: self.hc_scale,
            hc_base: self.hc_base,
            mix: o.0[4],
            done: DevicePtr(0),
            mix_tokens: TMAX + PAD,
        }
    }
}

/// 2026-10-15: The checkpoint's two sites of `mhc_layer{L}.safetensors` (BF16 `fn`, F32 `base`
/// and `scale`); a wrong dtype or shape is an error.
fn packet_sites(gpu: &dyn GpuBackend, dir: &str, layer: usize) -> Result<Vec<Site>> {
    let path = format!("{dir}/mhc_layer{layer}.safetensors");
    let raw = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
    let n = u64::from_le_bytes(raw[..8].try_into()?) as usize;
    let head: serde_json::Value = serde_json::from_slice(&raw[8..8 + n])?;
    let tensor = |name: &str, dtype: &str, shape: &[u64]| -> Result<&[u8]> {
        let t = &head[name];
        ensure!(t["dtype"] == dtype, "{path} {name}: dtype {}", t["dtype"]);
        ensure!(
            t["shape"] == serde_json::json!(shape),
            "{path} {name}: shape {}",
            t["shape"]
        );
        let (a, b) = (
            t["data_offsets"][0].as_u64().context("offset")? as usize,
            t["data_offsets"][1].as_u64().context("offset")? as usize,
        );
        Ok(&raw[8 + n + a..8 + n + b])
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let mut out = Vec::new();
    for s in ["attn", "ffn"] {
        let hc_fn = tensor(
            &format!("hc_{s}_fn"),
            "BF16",
            &[MIX as u64, (HC * H) as u64],
        )?;
        let base = f32s(tensor(&format!("hc_{s}_base"), "F32", &[MIX as u64])?);
        let scale = f32s(tensor(&format!("hc_{s}_scale"), "F32", &[3])?);
        out.push(Site::new(
            gpu,
            format!("layer {layer} {s}"),
            hc_fn.to_vec(),
            &base,
            &scale,
        )?);
    }
    Ok(out)
}

fn synthetic(gpu: &dyn GpuBackend, rng: &mut Rng, name: &str, w: f32) -> Result<Site> {
    let hc_fn: Vec<u8> = (0..MIX * HC * H)
        .flat_map(|_| bf16::from_f32(rng.unit() * w).to_le_bytes())
        .collect();
    let base: Vec<f32> = (0..MIX).map(|_| rng.unit() * 0.5).collect();
    Site::new(gpu, name.to_string(), hc_fn, &base, &[0.7, 1.3, 0.9])
}

struct Bench<'a> {
    gpu: &'a dyn GpuBackend,
    k: &'a Glm5NextMhcKernels,
    norm_w: DevicePtr,
    streams: DevicePtr,
    tok: Out,
    tile: Out,
}

impl Bench<'_> {
    /// 2026-10-15: Tokens `r0..r0 + t` of the streams into rows `0..t` of `o`.
    fn run(&self, kernel: KernelHandle, site: &Site, o: &Out, r0: usize, t: usize) -> Result<()> {
        let w = site.weights(o);
        let s = self.streams.offset(r0 * HC * H * 4);
        let outs = [o.0[0], o.0[2], o.0[3], o.0[1]];
        let (dims, eps) = ([H as u32, HC as u32, 20], [1e-5, 1e-6, 1e-5]);
        if kernel.0 == self.k.hc_pre_norm_tok.0 {
            launch_tok(
                self.gpu,
                kernel,
                s,
                &w,
                self.norm_w,
                outs,
                t as u32,
                dims,
                eps,
                0,
            )
        } else {
            launch_tile(
                self.gpu,
                kernel,
                s,
                &w,
                self.norm_w,
                outs,
                t as u32,
                dims,
                eps,
                0,
            )
        }
    }

    fn pair(&self, site: &Site, r0: usize, t: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        self.tok.poison(self.gpu, t)?;
        self.tile.poison(self.gpu, t)?;
        self.run(self.k.hc_pre_norm_tok, site, &self.tok, r0, t)?;
        self.run(self.k.hc_pre_norm_tile, site, &self.tile, r0, t)?;
        Ok((self.tok.bytes(self.gpu, t)?, self.tile.bytes(self.gpu, t)?))
    }

    fn median_us(&self, kernel: KernelHandle, site: &Site, o: &Out, t: usize) -> Result<[f64; 3]> {
        let mut v = Vec::new();
        for _ in 0..9 {
            self.gpu.synchronize(0)?;
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                self.run(kernel, site, o, 0, t)?;
            }
            self.gpu.synchronize(0)?;
            v.push(t0.elapsed().as_secs_f64() * 2e5);
        }
        v.sort_by(f64::total_cmp);
        Ok([v[4], v[0], v[8]])
    }
}

fn main() -> Result<()> {
    let dir = std::env::var("MHC_PACKET_DIR")
        .context("MHC_PACKET_DIR (the checkpoint mHC packets) is required")?;
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Glm5NextMhcKernels::resolve(&gpu)?;
    ensure!(
        k.hc_pre_norm_tok.0 != 0,
        "glm5next_hc_pre_norm_tok4 is not in this build"
    );
    ensure!(
        k.hc_pre_norm_tile.0 != 0,
        "{HC_PRE_TILE_KERNEL} is not in this build"
    );
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    let norm: Vec<f32> = (0..H).map(|_| 1.0 + 0.2 * rng.unit()).collect();
    let norm_w: Vec<u8> = norm
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    let mut sites = vec![
        synthetic(&gpu, &mut rng, "synthetic w 0.05", 0.05)?,
        synthetic(&gpu, &mut rng, "synthetic w 0.5", 0.5)?,
    ];
    for l in LAYERS {
        sites.extend(packet_sites(&gpu, &dir, l)?);
    }
    let mut b = Bench {
        gpu: &gpu,
        k: &k,
        norm_w: up(&gpu, &norm_w)?,
        streams: DevicePtr(0),
        tok: Out::alloc(&gpu)?,
        tile: Out::alloc(&gpu)?,
    };
    // 2026-10-15: Streams of magnitude 2 and of magnitude 50 with every 97th value x 40.
    let dists: [(&str, f32); 2] = [("streams 2", 2.0), ("streams 50 + outliers", 50.0)];
    for (dname, mag) in dists {
        let mut h: Vec<f32> = (0..TMAX * HC * H).map(|_| rng.unit() * mag).collect();
        if mag > 2.0 {
            h.iter_mut().step_by(97).for_each(|x| *x *= 40.0);
        }
        b.streams = up(&gpu, &f32b(&h))?;
        for site in &sites {
            for (t, r0) in [
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (17, 0),
                (17, 3),
                (684, 0),
                (684, 3),
                (2736, 0),
            ] {
                let (want, got) = b.pair(site, r0, t)?;
                if got != want {
                    let i = want.iter().zip(&got).position(|(a, b)| a != b).unwrap_or(0);
                    bail!(
                        "{} / {dname} T={t} r0={r0}: bytes differ from tok4 (first at {i})",
                        site.name
                    );
                }
            }
            println!("{} / {dname}: identical at every T", site.name);
        }
        gpu.free(b.streams)?;
    }
    negative_controls(&mut b, &sites[2], &mut rng)?;
    println!("BYTE GATE: PASS");
    // 2026-10-15: Fresh streams for the timing; the gate's were freed above.
    let h: Vec<f32> = (0..TMAX * HC * H).map(|_| rng.unit() * 2.0).collect();
    b.streams = up(&gpu, &f32b(&h))?;
    for t in [684usize, 2736] {
        let site = &sites[2];
        let a = b.median_us(k.hc_pre_norm_tok, site, &b.tok, t)?;
        let c = b.median_us(k.hc_pre_norm_tile, site, &b.tile, t)?;
        println!(
            "T={t:5}: tok4 {:8.1} us [{:.1}, {:.1}], tile {:8.1} us [{:.1}, {:.1}] (median [min, max], \
             n=9 x 5 launches) -> {:.2}x",
            a[0],
            a[1],
            a[2],
            c[0],
            c[1],
            c[2],
            a[0] / c[0]
        );
    }
    gpu.free(b.streams)?;
    Ok(())
}

/// 2026-10-15: Each control must change the tile kernel's bytes against tok4's unperturbed run.
fn negative_controls(b: &mut Bench, site: &Site, rng: &mut Rng) -> Result<()> {
    let t = 684;
    let h: Vec<f32> = (0..t * HC * H).map(|_| rng.unit() * 2.0).collect();
    b.streams = up(b.gpu, &f32b(&h))?;
    let (want, got) = b.pair(site, 0, t)?;
    ensure!(got == want, "negative-control baseline differs");
    let row = (t - 1) * HC * H;
    b.gpu
        .copy_h2d(&f32b(&[h[row + 7] + 1.0]), b.streams.offset((row + 7) * 4))?;
    b.tile.poison(b.gpu, t)?;
    b.run(b.k.hc_pre_norm_tile, site, &b.tile, 0, t)?;
    ensure!(
        b.tile.bytes(b.gpu, t)? != want,
        "negative control (stream + 1.0) did not fire"
    );
    println!("negative control stream + 1.0: FIRED");
    b.gpu
        .copy_h2d(&f32b(&h[row + 7..row + 8]), b.streams.offset((row + 7) * 4))?;
    let mut flipped = site.hc_fn.clone();
    flipped[2 * (5 * HC * H + 1234) + 1] ^= 0x01;
    b.gpu.copy_h2d(&flipped, site.hc_fn_d)?;
    b.tile.poison(b.gpu, t)?;
    b.run(b.k.hc_pre_norm_tile, site, &b.tile, 0, t)?;
    let fired = b.tile.bytes(b.gpu, t)? != want;
    b.gpu.copy_h2d(&site.hc_fn, site.hc_fn_d)?;
    ensure!(fired, "negative control (hc_fn bit flip) did not fire");
    println!("negative control hc_fn bit flip: FIRED");
    b.gpu.free(b.streams)?;
    Ok(())
}
