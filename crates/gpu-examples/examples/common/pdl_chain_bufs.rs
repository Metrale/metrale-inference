// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-21: The buffers of `pdl_chain.rs`: one rank's (TP=3, rank 0) weights of `layers`
//! KDA + MoE layer pairs at GLM-5.3's shapes, the per-layer KDA state, and the scratch the chain
//! passes between its kernels at M = 2 rows (a speculative verify of one sequence). Every device
//! pointer is plain memory of the shared CUDA context, so the same `Bufs` serves a PDL-on and a
//! PDL-off backend.
//!
//! Owner: gpu-examples.
//! Invariants:
//! - Weights are drawn once from a seed and never written. FP8 and E4M3 scale bytes avoid the NaN
//!   codes, so a stale read shows as a byte difference rather than NaN over NaN.
//! - The routed experts' NVFP4 slices are views into one pool far larger than L2, each expert at its
//!   own offset (wrapping), so the sweeps stream from DRAM as the serve's do.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::QuantizedWeight;

pub const T: u32 = 2;
pub const H: u32 = 4096;
pub const HC: u32 = 4;
pub const MIX: u32 = 24;
pub const MIX_TOKENS: usize = 64;
pub const HEADS: u32 = 22;
pub const D: u32 = 128;
pub const QKV: u32 = HEADS * D;
pub const CD: u32 = 3 * QKV;
pub const QK_CH: u32 = 2 * QKV;
pub const LOW: u32 = 128;
pub const EXPERTS: u32 = 288;
pub const TOP_K: u32 = 8;
pub const SLOTS: u32 = T * TOP_K;
pub const MI: u32 = 704;
pub const MI_PAD: u32 = 768;
pub const SHARED: u32 = 768;
pub const SIX: [u32; 6] = [QKV, QKV, QKV, LOW, HEADS, LOW];
const EXPERT_POOL: usize = 768 << 20;
const DENSE_POOL: usize = 256 << 20;

/// 2026-10-21: xorshift64*.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// 2026-10-21: Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n + 8);
        while v.len() < n {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(n);
        v
    }
}

pub fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
pub fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(16))?;
    g.memset(p, 0, n.max(16))?;
    Ok(p)
}
pub fn bf16s(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}
pub fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn rnd_bf16(g: &dyn GpuBackend, r: &mut Rng, n: usize, scale: f32) -> Result<DevicePtr> {
    up(
        g,
        &bf16s(&(0..n).map(|_| r.unit() * scale).collect::<Vec<_>>()),
    )
}
fn rnd_f32(g: &dyn GpuBackend, r: &mut Rng, n: usize, scale: f32, bias: f32) -> Result<DevicePtr> {
    up(
        g,
        &f32s(&(0..n).map(|_| r.unit() * scale + bias).collect::<Vec<_>>()),
    )
}
/// 2026-10-21: E4M3 bytes of magnitude 2^-3 .. 2^2, both signs (never NaN).
fn e4m3(r: &mut Rng, n: usize) -> Vec<u8> {
    r.bytes(n)
        .iter()
        .map(|b| (b & 0x80) | (0x30 + (b & 0x7f) % 0x20))
        .collect()
}

/// 2026-10-21: NVFP4 weights as views into a random pool: codes, then E4M3 scales at 1/8 the
/// offset; consecutive requests take consecutive 256-byte-aligned slices, wrapping.
pub struct Nvfp4Pool {
    packed: DevicePtr,
    scale: DevicePtr,
    bytes: usize,
    next: usize,
}
impl Nvfp4Pool {
    fn new(g: &dyn GpuBackend, r: &mut Rng, bytes: usize) -> Result<Self> {
        Ok(Self {
            packed: up(g, &r.bytes(bytes))?,
            scale: up(g, &e4m3(r, bytes / 8))?,
            bytes,
            next: 0,
        })
    }
    pub fn take(&mut self, n: u32, k: u32) -> QuantizedWeight {
        let pb = (n as usize * k as usize / 2).div_ceil(256) * 256;
        if self.next + pb > self.bytes {
            self.next = 0;
        }
        let off = self.next;
        self.next += pb;
        QuantizedWeight {
            weight: self.packed.offset(off),
            weight_scale: self.scale.offset(off / 8),
            weight_scale_2: 0.0123,
            ..QuantizedWeight::null()
        }
    }
}

/// 2026-10-21: One mHC site: hc_fn, scales, biases, the norm weight and its mix scratch.
pub struct Site {
    pub hc_fn: DevicePtr,
    pub scale: DevicePtr,
    pub base: DevicePtr,
    pub norm_w: DevicePtr,
    pub mix: DevicePtr,
    pub done: DevicePtr,
    pub post: DevicePtr,
    pub comb: DevicePtr,
}
impl Site {
    fn new(g: &dyn GpuBackend, r: &mut Rng) -> Result<Self> {
        Ok(Self {
            hc_fn: rnd_bf16(g, r, (MIX * HC * H) as usize, 0.02)?,
            scale: up(g, &f32s(&[0.5, 0.5, 0.5]))?,
            base: rnd_f32(g, r, MIX as usize, 0.3, 0.0)?,
            norm_w: rnd_bf16(g, r, H as usize, 0.5)?,
            mix: zeros(g, MIX_TOKENS * MIX as usize * 4)?,
            done: zeros(g, MIX_TOKENS * 4)?,
            post: zeros(g, (T * HC) as usize * 4)?,
            comb: zeros(g, (T * HC * HC) as usize * 4)?,
        })
    }
}

/// 2026-10-21: The routed experts' pointer tables (u64 packed, u64 scale, f32 scale_2 per id).
pub struct Table {
    pub packed: DevicePtr,
    pub scale: DevicePtr,
    pub scale2: DevicePtr,
}
fn table(g: &dyn GpuBackend, pool: &mut Nvfp4Pool, n: u32, k: u32) -> Result<Table> {
    let ws: Vec<QuantizedWeight> = (0..EXPERTS).map(|_| pool.take(n, k)).collect();
    let p: Vec<u8> = ws.iter().flat_map(|w| w.weight.0.to_le_bytes()).collect();
    let s: Vec<u8> = ws
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    Ok(Table {
        packed: up(g, &p)?,
        scale: up(g, &s)?,
        scale2: up(g, &f32s(&vec![0.01; EXPERTS as usize]))?,
    })
}

/// 2026-10-21: One KDA + MoE layer pair's weights and its KDA state.
pub struct Layer {
    pub attn: Site,
    pub ffn: Site,
    pub six: Vec<QuantizedWeight>,
    pub o: QuantizedWeight,
    pub fb: DevicePtr,
    pub fb_s: DevicePtr,
    pub gb: DevicePtr,
    pub gb_s: DevicePtr,
    pub dt_bias: DevicePtr,
    pub a_log: DevicePtr,
    pub conv_w: DevicePtr,
    pub o_norm_w: DevicePtr,
    pub conv: DevicePtr,
    pub rec: DevicePtr,
    pub conv_int: DevicePtr,
    pub h_int: DevicePtr,
    pub router: DevicePtr,
    pub router_bias: DevicePtr,
    pub sh_gu: [QuantizedWeight; 2],
    pub sh_down: QuantizedWeight,
}

pub const CONV_BYTES: usize = (CD * 4) as usize * 4;
pub const REC_BYTES: usize = (HEADS * D * D) as usize * 4;

/// 2026-10-21: Scratch passed between the chain's kernels (one set for every layer).
pub struct Scratch {
    pub streams: DevicePtr,
    pub y: DevicePtr,
    pub normed: DevicePtr,
    pub qkv_parts: DevicePtr,
    pub lowrank: DevicePtr,
    pub beta_bf16: DevicePtr,
    pub lowrank_g: DevicePtr,
    pub qkv_proj: DevicePtr,
    pub g_raw: DevicePtr,
    pub out_gate: DevicePtr,
    pub gate: DevicePtr,
    pub beta: DevicePtr,
    pub conv_out: DevicePtr,
    pub core: DevicePtr,
    pub o_norm_out: DevicePtr,
    pub final_out: DevicePtr,
    pub peers: DevicePtr,
    pub ar_mark: DevicePtr,
    pub logits: DevicePtr,
    pub ids: DevicePtr,
    pub wts: DevicePtr,
    pub aq: DevicePtr,
    pub as_: DevicePtr,
    pub ag: DevicePtr,
    pub u_eid: DevicePtr,
    pub u_slot: DevicePtr,
    pub a_gate: DevicePtr,
    pub a_up: DevicePtr,
    pub a_act: DevicePtr,
    pub aq2: DevicePtr,
    pub as2: DevicePtr,
    pub ag2: DevicePtr,
    pub expert_out: DevicePtr,
    pub sg: DevicePtr,
    pub su: DevicePtr,
    pub sact: DevicePtr,
    pub shared_out: DevicePtr,
    pub moe_out: DevicePtr,
}

pub const STREAM_BYTES: usize = (T * HC * H) as usize * 4;

impl Scratch {
    fn new(g: &dyn GpuBackend, r: &mut Rng) -> Result<Self> {
        let z = |n: u32| zeros(g, n as usize);
        Ok(Self {
            streams: z(STREAM_BYTES as u32)?,
            y: z(T * H * 2)?,
            normed: z(T * H * 2)?,
            qkv_parts: z(3 * T * QKV * 2)?,
            lowrank: z(T * LOW * 2)?,
            beta_bf16: z(T * HEADS * 2)?,
            lowrank_g: z(T * LOW * 2)?,
            qkv_proj: z(T * CD * 2)?,
            g_raw: z(T * QKV * 2)?,
            out_gate: z(T * QKV * 2)?,
            gate: z(T * QKV * 4)?,
            beta: z(T * HEADS * 4)?,
            conv_out: z(T * CD * 2)?,
            core: z(T * QKV * 4)?,
            o_norm_out: z(T * QKV * 2)?,
            final_out: z(T * H * 2)?,
            peers: rnd_bf16(g, r, (2 * T * H) as usize, 0.5)?,
            ar_mark: z(16)?,
            logits: z(T * EXPERTS * 4)?,
            ids: z(SLOTS * 4)?,
            wts: z(SLOTS * 4)?,
            aq: z(T * H / 2)?,
            as_: z(T * H / 16)?,
            ag: z(T * 4)?,
            u_eid: z(SLOTS * 4)?,
            u_slot: z(SLOTS * T * 4)?,
            a_gate: z(SLOTS * MI * 2)?,
            a_up: z(SLOTS * MI * 2)?,
            a_act: z(SLOTS * MI * 2)?,
            aq2: z(SLOTS * MI_PAD / 2)?,
            as2: z(SLOTS * MI_PAD / 16)?,
            ag2: z(SLOTS * 4)?,
            expert_out: z(SLOTS * H * 2)?,
            sg: z(T * SHARED * 2)?,
            su: z(T * SHARED * 2)?,
            sact: z(T * SHARED * 2)?,
            shared_out: z(T * H * 2)?,
            moe_out: z(T * H * 2)?,
        })
    }
}

pub struct Bufs {
    pub layers: Vec<Layer>,
    pub gate_t: Table,
    pub up_t: Table,
    pub down_t: Table,
    pub s: Scratch,
}

impl Bufs {
    /// 2026-10-21: `layers` layer pairs drawn from `seed`. `router` scales the router weights:
    /// the gate routes by its input (so a stale read moves the experts); the bench passes 0, so
    /// each layer's experts are fixed by its bias and every replay streams the same bytes.
    pub fn new(g: &dyn GpuBackend, seed: u64, layers: usize, router: f32) -> Result<Self> {
        let r = &mut Rng(seed | 1);
        let mut ep = Nvfp4Pool::new(g, r, EXPERT_POOL)?;
        let mut dp = Nvfp4Pool::new(g, r, DENSE_POOL)?;
        let gate_t = table(g, &mut ep, MI, H)?;
        let up_t = table(g, &mut ep, MI, H)?;
        let down_t = table(g, &mut ep, H, MI)?;
        let mut ls = Vec::with_capacity(layers);
        for _ in 0..layers {
            ls.push(Layer {
                attn: Site::new(g, r)?,
                ffn: Site::new(g, r)?,
                six: SIX.iter().map(|&n| dp.take(n, H)).collect(),
                o: dp.take(H, QKV),
                fb: up(g, &e4m3(r, (QKV * LOW) as usize))?,
                fb_s: rnd_f32(g, r, QKV as usize, 0.002, 0.004)?,
                gb: up(g, &e4m3(r, (QKV * LOW) as usize))?,
                gb_s: rnd_f32(g, r, QKV as usize, 0.002, 0.004)?,
                dt_bias: rnd_f32(g, r, QKV as usize, 0.5, 0.0)?,
                a_log: rnd_f32(g, r, HEADS as usize, 0.5, 0.0)?,
                conv_w: rnd_bf16(g, r, (CD * 4) as usize, 0.5)?,
                o_norm_w: rnd_bf16(g, r, D as usize, 0.5)?,
                conv: rnd_f32(g, r, CONV_BYTES / 4, 0.5, 0.0)?,
                rec: rnd_f32(g, r, REC_BYTES / 4, 0.1, 0.0)?,
                conv_int: zeros(g, CONV_BYTES)?,
                h_int: zeros(g, REC_BYTES)?,
                router: rnd_bf16(g, r, (EXPERTS * H) as usize, router)?,
                router_bias: rnd_f32(g, r, EXPERTS as usize, 0.5, 0.0)?,
                sh_gu: [dp.take(SHARED, H), dp.take(SHARED, H)],
                sh_down: dp.take(H, SHARED),
            });
        }
        Ok(Self {
            layers: ls,
            gate_t,
            up_t,
            down_t,
            s: Scratch::new(g, r)?,
        })
    }
}
