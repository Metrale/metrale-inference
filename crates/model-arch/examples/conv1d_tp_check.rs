// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-exactness check of the token-parallel prefill conv against the serial
//! `causal_conv1d_update_prefill`: any differing output element fails the run.
//! `CV_DIM` (default 8192) and `CV_SEQ` (default 2700) set the shape.
//! 2026-09-30: It runs the production pair (`ops::conv1d_update_prefill`: the token-parallel
//! kernel, then `causal_conv1d_prefill_state`) `CV_REPS` times (default 50), each from a fresh
//! copy of the incoming state, and also compares the new conv_state.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uni(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * ((self.next() >> 11) as f32 / (1u64 << 53) as f32)
    }
}
fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}
fn up_u16(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn main() -> Result<()> {
    let dim: usize = std::env::var("CV_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let seq: usize = std::env::var("CV_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2700);
    let dconv = 4usize;
    println!("=== causal_conv1d prefill: token-parallel vs serial ===");
    println!("dim={dim} seq_len={seq} d_conv={dconv}");

    let be = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &be;
    let st = g.create_stream()?;
    let mut r = Rng(0x_C0FF_EE11);

    let inp: Vec<u16> = (0..seq * dim).map(|_| bf16(r.uni(-2.0, 2.0))).collect();
    let wt: Vec<u16> = (0..dim * dconv).map(|_| bf16(r.uni(-0.5, 0.5))).collect();
    let bi: Vec<f32> = (0..dim).map(|_| r.uni(-0.1, 0.1)).collect();
    let s0: Vec<f32> = (0..dim * dconv).map(|_| r.uni(-1.0, 1.0)).collect();

    let (pi, pw, pb) = (up_u16(g, &inp)?, up_u16(g, &wt)?, up_f32(g, &bi)?);
    let serial = g.kernel("causal_conv1d", "causal_conv1d_update_prefill")?;
    let tp = g.kernel("causal_conv1d", "causal_conv1d_update_prefill_tp")?;
    let weight = metrale_model_layers::weight_map::DenseWeight { weight: pw };
    // 2026-09-30: One prefill from a fresh copy of `s0`: the output rows and the new conv_state.
    // `tp_k` 0 runs the serial kernel; otherwise the production pair
    // (`ops::conv1d_update_prefill`: the token-parallel kernel, then the state kernel).
    let once = |tp_k: metrale_gpu_runtime::gpu::KernelHandle| -> Result<(Vec<u16>, Vec<u8>)> {
        let ps = up_f32(g, &s0)?;
        let po = g.alloc(seq * dim * 2)?;
        metrale_model_layers::layers::ops::conv1d_update_prefill(
            g,
            serial,
            tp_k,
            ps,
            pi,
            &weight,
            pb,
            po,
            dim as u32,
            dconv as u32,
            seq as u32,
            dim as u32,
            dim as u32,
            st,
        )?;
        g.synchronize(st)?;
        let mut state = vec![0u8; dim * dconv * 4];
        g.copy_d2h(ps, &mut state)?;
        Ok((dn_u16(g, po, seq * dim)?, state))
    };
    let t = std::time::Instant::now();
    let (want, want_state) = once(metrale_gpu_runtime::gpu::KernelHandle(0))?;
    let ta = t.elapsed().as_secs_f64() * 1000.0;
    // 2026-09-30: Repeated, because the race this guards against (the new state written while
    // other blocks still read the old one) shows only when the last block runs first.
    let reps: usize = std::env::var("CV_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let t = std::time::Instant::now();
    let mut bad = 0usize;
    for rep in 0..reps {
        let (out, state) = once(tp)?;
        let d = out.iter().zip(&want).filter(|(x, y)| x != y).count();
        let ds = state
            .chunks(4)
            .zip(want_state.chunks(4))
            .filter(|(x, y)| x != y)
            .count();
        if d != 0 || ds != 0 {
            println!("  rep {rep}: {d} output and {ds} state elements differ");
            bad += 1;
        }
    }
    let tb = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    println!("  serial          {ta:8.3} ms (one run, incl. copies)");
    println!("  token-parallel  {tb:8.3} ms/run (incl. copies)");
    if bad != 0 {
        bail!("NOT bit-identical in {bad} of {reps} runs");
    }
    println!(
        "\nPASS: output and conv_state bit-identical to the serial kernel in {reps} of {reps} runs"
    );
    Ok(())
}
