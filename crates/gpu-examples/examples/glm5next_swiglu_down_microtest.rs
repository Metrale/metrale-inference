// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `glm5next_swiglu_down_pipe_m<M>` (the clamped SwiGLU folded into the input stage
//! of `w4a16_gemv_pipe`) against the unfolded pair the GLM-5.3 shared expert runs today:
//! `glm5next_swiglu_clamp` into the activation buffer, then `w4a16_gemv_pipe_m<M>` over it.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the fold writes the unfolded pair's bytes for every output of every
//!   row, at M 1 and 2, at the shared down's shape (K 768, N 4096) and two wider K, on random
//!   inputs, inputs past the clamp limit on both sides, signed zeros, and at every block cap
//!   (the policy's, 1, 7, 4 waves, uncapped).
//! - Negative controls: a changed gate element of row 1 changes row 1 only; a different limit
//!   changes the output (the limit reaches the stage).
//!
//! Times: both arms captured into a CUDA graph over a rotated 128 MB weight pool, alternating
//! over 5 rounds, median of round medians [min..max], and the per-step saving at 42 shared
//! experts (rank 0).
//!
//! Run (GB10): cargo build --release -p metrale-gpu-examples --features cuda,gpu-examples
//!   --example glm5next_swiglu_down_microtest, then run the binary.

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, bail, ensure};
use fixture::{Pool, ROUNDS, Rng, STREAM_BYTES, bf16_rows, read, segs, stat};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::{self, PipeInput};
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-23: GLM-5.3's clamp limit (`swiglu_limit`) and the shared expert's down shape at TP=3.
const LIMIT: f32 = 10.0;
const N: u32 = 4096;
const SHAPES: [u32; 3] = [768, 2816, 4096];
/// 2026-10-23: Shared experts per step on rank 0 (the fixture's `shared down` count).
const PER_STEP: f64 = 42.0;

struct Bufs {
    gate: DevicePtr,
    up: DevicePtr,
    act: DevicePtr,
    out: DevicePtr,
}

fn swiglu(g: &dyn GpuBackend, k: KernelHandle, b: &Bufs, n: u32, limit: f32, s: u64) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.gate)
        .arg_ptr(b.up)
        .arg_ptr(b.act)
        .arg_u32(n)
        .arg_f32(limit)
        .launch(s)
}

struct Arms {
    swiglu: KernelHandle,
    fold: [KernelHandle; 2],
}

impl Arms {
    fn unfolded(
        &self,
        g: &dyn GpuBackend,
        b: &Bufs,
        w: QuantizedWeight,
        m: u32,
        k: u32,
        cap: u32,
        s: u64,
    ) -> Result<()> {
        swiglu(g, self.swiglu, b, m * k, LIMIT, s)?;
        ops::w4a16_gemv_pipe_at(g, b.act, &segs(&[w], &[b.out], &[N]), m, k, cap, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn folded(
        &self,
        g: &dyn GpuBackend,
        b: &Bufs,
        w: QuantizedWeight,
        m: u32,
        k: u32,
        cap: u32,
        limit: f32,
        s: u64,
    ) -> Result<()> {
        let input = PipeInput::Pair(b.gate, b.up, limit);
        let sg = segs(&[w], &[b.out], &[N]);
        ops::w4a16_gemv_pipe_staged(g, self.fold[m as usize - 1], input, &sg, m, k, cap, s)
    }
}

/// 2026-10-23: Gate and up inputs of `m` rows of `k`: random, past the limit, signed zeros.
fn inputs(rng: &mut Rng, m: usize, k: usize) -> Vec<(&'static str, Vec<u8>, Vec<u8>)> {
    let fill = |f: &dyn Fn(usize) -> u16| -> Vec<u8> {
        (0..m * k).flat_map(|i| f(i).to_le_bytes()).collect()
    };
    // 2026-10-23: +-12.0, +-40.0, +-0.5 and +-0.0 cycling: both clamps, inside, and zeros.
    let edge = [
        0x4140u16, 0xC140, 0x4220, 0xC220, 0x3F00, 0xBF00, 0x0000, 0x8000,
    ];
    vec![
        ("random", bf16_rows(rng, m, k), bf16_rows(rng, m, k)),
        (
            "edges",
            fill(&|i| edge[i % 8]),
            fill(&|i| edge[(i / 8 + i) % 8]),
        ),
        (
            "zeros",
            fill(&|i| if i % 2 == 0 { 0x8000 } else { 0 }),
            bf16_rows(rng, m, k),
        ),
    ]
}

fn byte_gate(
    g: &dyn GpuBackend,
    s: u64,
    a: &Arms,
    b: &Bufs,
    pool: &Pool,
    rng: &mut Rng,
) -> Result<usize> {
    let sms = g.sm_count()?;
    let mut checked = 0;
    for k in SHAPES {
        for m in [1u32, 2] {
            let w = pool.weight(k as usize + m as usize, N, k);
            let policy = ops::w4a16_gemv_pipe_cap(&[N], k, m, sms);
            for (case, gh, uh) in inputs(rng, m as usize, k as usize) {
                g.copy_h2d(&gh, b.gate)?;
                g.copy_h2d(&uh, b.up)?;
                g.memset(b.out, 0xA5, 2 * N as usize * 2)?;
                a.unfolded(g, b, w, m, k, policy, s)?;
                let want = read(g, s, b.out, m as usize * N as usize * 2)?;
                for cap in [policy, 1, 7, sms * 16, u32::MAX] {
                    g.memset(b.out, 0x5A, 2 * N as usize * 2)?;
                    a.folded(g, b, w, m, k, cap, LIMIT, s)?;
                    let got = read(g, s, b.out, want.len())?;
                    if let Some(at) = got.chunks(2).zip(want.chunks(2)).position(|(x, y)| x != y) {
                        bail!(
                            "K={k} M={m} {case} cap={cap}: output {at} (row {}) differs from the unfolded pair",
                            at / N as usize
                        );
                    }
                    checked += want.len() / 2;
                }
            }
        }
    }
    Ok(checked)
}

fn negative_controls(
    g: &dyn GpuBackend,
    s: u64,
    a: &Arms,
    b: &Bufs,
    pool: &Pool,
    rng: &mut Rng,
) -> Result<()> {
    let (k, sms) = (768u32, g.sm_count()?);
    let ku = k as usize;
    let w = pool.weight(3, N, k);
    let cap = ops::w4a16_gemv_pipe_cap(&[N], k, 2, sms);
    let mut gh = bf16_rows(rng, 2, ku);
    g.copy_h2d(&gh, b.gate)?;
    g.copy_h2d(&bf16_rows(rng, 2, ku), b.up)?;
    let run = |limit: f32| -> Result<Vec<u8>> {
        a.folded(g, b, w, 2, k, cap, limit, s)?;
        read(g, s, b.out, 2 * N as usize * 2)
    };
    let base = run(LIMIT)?;
    let half = N as usize * 2;
    ensure!(
        run(LIMIT * 0.5)? != base,
        "negative control: the limit must reach the stage"
    );
    gh[(ku + 5) * 2..(ku + 5) * 2 + 2].copy_from_slice(&0x4000u16.to_le_bytes());
    g.copy_h2d(&gh, b.gate)?;
    let moved = run(LIMIT)?;
    ensure!(
        moved[..half] == base[..half] && moved[half..] != base[half..],
        "negative control: a changed row-1 gate element must change row 1 only"
    );
    println!("negative controls fire: the limit reaches the stage; row isolation");
    Ok(())
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let sms = g.sm_count()?;
    let arms = Arms {
        swiglu: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
        fold: [
            g.kernel("glm5next_swiglu_down", "glm5next_swiglu_down_pipe_m1")?,
            g.kernel("glm5next_swiglu_down", "glm5next_swiglu_down_pipe_m2")?,
        ],
    };
    let mut rng = Rng(0x7377_6967_6c75_646e);
    let pool = Pool::new(&g, &mut rng)?;
    let kmax = *SHAPES.iter().max().unwrap_or(&0) as usize;
    let b = Bufs {
        gate: g.alloc(2 * kmax * 2)?,
        up: g.alloc(2 * kmax * 2)?,
        act: g.alloc(2 * kmax * 2)?,
        out: g.alloc(2 * N as usize * 2)?,
    };
    let checked = byte_gate(&g, s, &arms, &b, &pool, &mut rng)?;
    println!(
        "byte gate: {checked} outputs of the fold equal glm5next_swiglu_clamp + w4a16_gemv_pipe \
         (K {SHAPES:?}, M 1/2, 3 input cases, 5 caps)"
    );
    negative_controls(&g, s, &arms, &b, &pool, &mut rng)?;

    let k = 768u32;
    g.copy_h2d(&bf16_rows(&mut rng, 2, k as usize), b.gate)?;
    g.copy_h2d(&bf16_rows(&mut rng, 2, k as usize), b.up)?;
    let copies = STREAM_BYTES
        .div_ceil(N as usize * k as usize / 2)
        .clamp(4, 2048);
    for m in [1u32, 2] {
        let cap = ops::w4a16_gemv_pipe_cap(&[N], k, m, sms);
        let mut med = [Vec::new(), Vec::new()];
        for _ in 0..ROUNDS {
            for (ai, fold) in [false, true].into_iter().enumerate() {
                let mut i = 0usize;
                let t = timing::time_graph(&g, s, copies, &mut |st| {
                    let w = pool.weight(i, N, k);
                    i += 1;
                    if fold {
                        arms.folded(&g, &b, w, m, k, cap, LIMIT, st)
                    } else {
                        arms.unfolded(&g, &b, w, m, k, cap, st)
                    }
                })?;
                med[ai].push(t.0);
            }
        }
        let (old, new) = (stat(&mut med[0]), stat(&mut med[1]));
        timing::report(&format!("shared swiglu+down K={k}"), m as usize, old, new);
        println!(
            "M={m} per step at {PER_STEP} shared experts: saves {:.1} us",
            (old.0 - new.0) * PER_STEP
        );
    }
    Ok(())
}
