// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Byte equality of the two LoRA routes legacy takes for one adapter: the per-row
//! `lora_bgmv` pair (`ops::lora_delta::apply_lora_bgmv`, seq_slot = the
//! adapter's slot) and the installed GEMV pair (`ops::lora_delta::apply_lora_delta`, one row at a
//! time). Legacy runs the pair for an active-adapter request at one row and the bgmv for a routed
//! one and for every multi-row step, so the circuit's single bgmv arm for attention LoRA
//! (FUSIONS.toml `lora_*_bgmv`) is legacy's numerics exactly when these are equal.
//!
//! Owner: model-arch examples (FEATURES workstream).
//! Invariants:
//! - PASS needs every shape equal byte for byte, and the control (the bgmv with another B) to
//!   differ: a comparison that cannot fail proves nothing.
//!
//! Usage: cargo run --release -p metrale-model-arch --example lora_bgmv_pair_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::lora_delta::{
    LoraKernels, LoraPair, LoraRoute, apply_lora_bgmv, apply_lora_delta,
};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-03: `(name, k_in, n_out)` of every projection the dense 27B adapts
/// (`metrale_circuit::lora::ADAPTABLE`): q (with its gate), k, v, o, gate/up, down, GDN out_proj.
const SHAPES: [(&str, usize, usize); 7] = [
    ("q", 5120, 12288),
    ("k", 5120, 1024),
    ("v", 5120, 1024),
    ("o", 6144, 5120),
    ("gate_up", 5120, 17408),
    ("down", 17408, 5120),
    ("gdn_out", 6144, 5120),
];

/// 2026-10-03: `(rank, max_rank, rows)`: a padded and a full rank, one row and multi-row steps.
const CASES: [(usize, usize, usize); 4] = [(8, 16, 1), (16, 16, 1), (16, 64, 3), (64, 64, 8)];

struct Rng(u64);
impl Rng {
    fn bf16(&mut self, lo: f32, hi: f32) -> u16 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let u = ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32;
        ((lo + u * (hi - lo)).to_bits() >> 16) as u16
    }
    fn vec(&mut self, n: usize, mag: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| self.bf16(-mag, mag).to_le_bytes())
            .collect()
    }
}

fn put(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn get(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// 2026-10-03: A padded pool tensor: `rows x cols` with only the first `live` rows (A) or
/// columns (B) non-zero.
fn padded(rng: &mut Rng, rows: usize, cols: usize, live: usize, by_row: bool) -> Vec<u8> {
    let mut out = vec![0u8; rows * cols * 2];
    for r in 0..rows {
        for c in 0..cols {
            if (by_row && r < live) || (!by_row && c < live) {
                let b = rng.bf16(-0.05, 0.05).to_le_bytes();
                out[(r * cols + c) * 2..(r * cols + c) * 2 + 2].copy_from_slice(&b);
            }
        }
    }
    out
}

struct Run<'a> {
    gpu: &'a dyn GpuBackend,
    kernels: &'a LoraKernels,
    stream: u64,
}

impl Run<'_> {
    /// 2026-10-03: The folded outputs of the pair route and of the bgmv route over the same
    /// base, and of the bgmv with `b_other` (the control).
    fn both(
        &self,
        (k_in, n_out): (usize, usize),
        (r, max_rank, rows): (usize, usize, usize),
        seed: u64,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let g = self.gpu;
        let mut rng = Rng(seed);
        let x = put(g, &rng.vec(rows * k_in, 1.0))?;
        let base = rng.vec(rows * n_out, 2.0);
        let a = put(g, &padded(&mut rng, max_rank, k_in, r, true))?;
        let b = put(g, &padded(&mut rng, n_out, max_rank, r, false))?;
        let b_other = put(g, &padded(&mut rng, n_out, max_rank, r, false))?;
        let scale = 0.5f32;
        let pair = LoraPair {
            a: DenseWeight { weight: a },
            b: DenseWeight { weight: b },
            rank: r as u32,
            k_in: k_in as u32,
            n_out: n_out as u32,
            scale,
            max_rank: max_rank as u32,
        };
        let xa = g.alloc(rows * max_rank * 2)?;
        let delta = g.alloc(n_out * 2)?;
        let bytes = rows * n_out * 2;
        let out_pair = put(g, &base)?;
        for row in 0..rows {
            let xr = DevicePtr(x.0 + (row * k_in * 2) as u64);
            let or = DevicePtr(out_pair.0 + (row * n_out * 2) as u64);
            apply_lora_delta(g, self.kernels, &pair, xr, or, 1, xa, delta, self.stream)?;
        }
        let slots = put(g, &vec![0u8; rows * 4])?;
        let bgmv = |b_ptr: DevicePtr| -> Result<Vec<u8>> {
            let route = LoraRoute {
                a_table: put(g, &a.0.to_le_bytes())?,
                b_table: put(g, &b_ptr.0.to_le_bytes())?,
                scale_table: put(g, &scale.to_le_bytes())?,
                k_in: k_in as u32,
                n_out: n_out as u32,
                max_rank: max_rank as u32,
            };
            let out = put(g, &base)?;
            apply_lora_bgmv(
                g,
                self.kernels,
                &route,
                x,
                out,
                slots,
                rows as u32,
                k_in as u32,
                n_out as u32,
                xa,
                self.stream,
            )?;
            g.synchronize(self.stream)?;
            get(g, out, bytes)
        };
        let routed = bgmv(b)?;
        let control = bgmv(b_other)?;
        g.synchronize(self.stream)?;
        Ok((get(g, out_pair, bytes)?, routed, control))
    }
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let run = Run {
        gpu,
        kernels: &LoraKernels::new(gpu)?,
        stream: gpu.create_stream()?,
    };
    let mut failures = 0;
    for (i, (name, k_in, n_out)) in SHAPES.into_iter().enumerate() {
        for (j, case) in CASES.into_iter().enumerate() {
            let seed = 0x10_0000 + (i * 16 + j) as u64;
            let (pair, routed, control) = run.both((k_in, n_out), case, seed)?;
            let differ = pair.iter().zip(&routed).filter(|(a, b)| a != b).count() / 2;
            let control_differs = control != pair;
            let ok = differ == 0 && control_differs;
            println!(
                "{} {name:8} k_in={k_in:5} n_out={n_out:5} r={} max_rank={} rows={}: \
                 {differ} differing elements; control differs: {control_differs}",
                if ok { "PASS" } else { "FAIL" },
                case.0,
                case.1,
                case.2
            );
            failures += usize::from(!ok);
        }
    }
    if failures > 0 {
        bail!("RESULT: FAIL ({failures} cases)");
    }
    println!("RESULT: PASS (bgmv == GEMV pair byte for byte, every shape and case)");
    Ok(())
}
