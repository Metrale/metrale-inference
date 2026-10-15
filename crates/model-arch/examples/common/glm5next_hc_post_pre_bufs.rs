// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The buffers of `glm5next_hc_post_pre_tile_gate`: the read-only inputs (block_out,
//! residual, post, comb, the prefix cache's residual rows, the peers' partials) and one arm's
//! outputs, all `ROWS` rows long so a case may start at a row offset.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - [`Arm::reset`] leaves no byte of an arm from the previous case: every buffer is copied
//!   from the inputs or poisoned with 0xAB.

use crate::sites::{H, HC, MIX, Rng, bf16b, f32b, read, up};
use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub const TMAX: usize = 2736;
pub const R0MAX: usize = 64;
pub const PAD: usize = 8;
pub const ROWS: usize = TMAX + R0MAX + PAD;
/// 2026-10-15: Elements per peer slot; the slots hold every row, as the one-shot buffer does.
pub const SLOT: usize = ROWS * H;
pub const PEER_SLOTS: usize = 3;
/// 2026-10-15: Bytes per row of block_out, the streams, post, comb, y, normed and mix.
pub const ROW: [usize; 7] = [
    H * 2,
    HC * H * 4,
    HC * 4,
    HC * HC * 4,
    H * 2,
    H * 2,
    MIX * 4,
];
pub const NAMES: [&str; 7] = ["block_out", "streams", "post", "comb", "y", "normed", "mix"];
pub const DIMS: [u32; 3] = [H as u32, HC as u32, 20];
pub const EPS: [f32; 3] = [1e-5, 1e-6, 1e-5];

/// 2026-10-15: The buffer and row of byte `i` of [`Arm::bytes`] over `hi` rows.
pub fn locate(mut i: usize, hi: usize) -> String {
    for (name, r) in NAMES.iter().zip(ROW) {
        if i < hi * r {
            return format!("{name} row {}", i / r);
        }
        i -= hi * r;
    }
    "past the end".into()
}

/// 2026-10-15: The read-only inputs: block_out, residual, post and comb (`src`, indexed as
/// [`ROW`]), the prefix cache's residual rows, the peers' BF16 partials, the norm weight.
pub struct Inputs {
    pub src: [DevicePtr; 4],
    pub cache: DevicePtr,
    pub peers: DevicePtr,
    pub norm_w: DevicePtr,
}

/// 2026-10-15: One arm's buffers, indexed as [`ROW`].
pub struct Arm(pub [DevicePtr; 7]);

impl Arm {
    pub fn alloc(gpu: &dyn GpuBackend) -> Result<Self> {
        let mut p = [DevicePtr(0); 7];
        for (d, r) in p.iter_mut().zip(ROW) {
            *d = gpu.alloc(ROWS * r)?;
        }
        Ok(Self(p))
    }

    /// 2026-10-15: block_out from the input and, aliased, the streams, post and comb too (the
    /// layer updates them in place); every other buffer poisoned with 0xAB.
    pub fn reset(&self, gpu: &dyn GpuBackend, inp: &Inputs, aliased: bool) -> Result<()> {
        for (i, (d, r)) in self.0.iter().zip(ROW).enumerate() {
            if i == 0 || (aliased && i < 4) {
                gpu.copy_d2d(inp.src[i], *d, ROWS * r)?;
            } else {
                gpu.memset(*d, 0xAB, ROWS * r)?;
            }
        }
        gpu.synchronize(0)
    }

    /// 2026-10-15: Rows `0..hi` of every buffer, `mix` only when `with_mix`.
    pub fn bytes(&self, gpu: &dyn GpuBackend, hi: usize, with_mix: bool) -> Result<Vec<u8>> {
        let n = if with_mix { 7 } else { 6 };
        let mut v = Vec::new();
        for (d, r) in self.0.iter().zip(ROW).take(n) {
            v.extend(read(gpu, *d, hi * r)?);
        }
        Ok(v)
    }
}

/// 2026-10-15: Fresh inputs: residual and cache rows of magnitude `mag` (every 97th value x 40
/// when `mag` > 2), BF16 block_out and peer partials in [-1.5, 1.5), post in [0, 2), comb in
/// [0, 1).
pub fn inputs(gpu: &dyn GpuBackend, rng: &mut Rng, mag: f32, norm_w: DevicePtr) -> Result<Inputs> {
    let stream_rows = |rng: &mut Rng| -> Vec<u8> {
        let mut h: Vec<f32> = (0..ROWS * HC * H).map(|_| rng.unit() * mag).collect();
        if mag > 2.0 {
            h.iter_mut().step_by(97).for_each(|x| *x *= 40.0);
        }
        f32b(&h)
    };
    let mut vals =
        |n: usize, a: f32, b: f32| -> Vec<f32> { (0..n).map(|_| a + b * rng.unit()).collect() };
    let block_out = bf16b(&vals(ROWS * H, 0.0, 1.5));
    let peers = bf16b(&vals(PEER_SLOTS * SLOT, 0.0, 1.5));
    let post = f32b(&vals(ROWS * HC, 1.0, 1.0));
    let comb = f32b(&vals(ROWS * HC * HC, 0.5, 0.5));
    let residual = stream_rows(rng);
    let cache = stream_rows(rng);
    Ok(Inputs {
        src: [
            up(gpu, &block_out)?,
            up(gpu, &residual)?,
            up(gpu, &post)?,
            up(gpu, &comb)?,
        ],
        cache: up(gpu, &cache)?,
        peers: up(gpu, &peers)?,
        norm_w,
    })
}

pub fn free(gpu: &dyn GpuBackend, inp: &Inputs) -> Result<()> {
    for p in inp.src.iter().chain([&inp.cache, &inp.peers]) {
        gpu.free(*p)?;
    }
    Ok(())
}
