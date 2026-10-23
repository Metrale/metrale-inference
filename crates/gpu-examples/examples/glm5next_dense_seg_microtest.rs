// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `glm5next_dense_seg_pipe_m<M>` (the P3 persistent segment: gate|up GEMV, clamped
//! SwiGLU, down GEMV as three dependent phases of one launch) against the three launches the
//! GLM-5.3 shared expert runs today: `w4a16_gemv_pipe_m<M>` over gate|up, `glm5next_swiglu_clamp`,
//! `w4a16_gemv_pipe_m<M>` over the activation.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the segment writes the three launches' bytes in gate, up, act and out,
//!   at M 1 and 2, at the shared expert's rank widths (inter 768 and 512) and the dense FFN's
//!   (4096), on random inputs and inputs past the clamp limit, over 6 launches in a row (the
//!   control slot is left zero) and through a CUDA graph replayed 3 times.
//! - Negative controls: a different limit changes the output (phase 1 reaches phase 2); a changed
//!   x element of row 1 changes row 1 only.
//!
//! Times: both arms captured into a CUDA graph over a rotated 128 MB weight pool, alternating over
//! 5 rounds, median of round medians [min..max], and the per-step saving at 42 shared experts.
//!
//! Run (GB10): cargo build --release -p metrale-gpu-examples --features cuda,gpu-examples
//!   --example glm5next_dense_seg_microtest, then run the binary.

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, bail, ensure};
use fixture::{Pool, ROUNDS, Rng, bf16_rows, read, segs, stat};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::{self, SegmentCtl};
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-23: GLM-5.3's clamp limit and hidden size; the MLP widths per rank at TP=3.
const LIMIT: f32 = 10.0;
const HIDDEN: u32 = 4096;
const INTERS: [u32; 3] = [768, 512, 4096];
const PER_STEP: f64 = 42.0;

struct Bufs {
    x: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    act: DevicePtr,
    out: DevicePtr,
}

struct W {
    gate: QuantizedWeight,
    up: QuantizedWeight,
    down: QuantizedWeight,
}

struct Arms {
    swiglu: KernelHandle,
    seg: [KernelHandle; 2],
    ctl: SegmentCtl,
}

impl Arms {
    #[allow(clippy::too_many_arguments)]
    fn three(&self, g: &dyn GpuBackend, b: &Bufs, w: &W, m: u32, inter: u32, s: u64) -> Result<()> {
        let sms = g.sm_count()?;
        let gu = segs(&[w.gate, w.up], &[b.gate, b.up], &[inter, inter]);
        ops::w4a16_gemv_pipe(g, b.x, &gu, m, HIDDEN, sms, s)?;
        let n = m * inter;
        KernelLaunch::new(g, self.swiglu)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.gate)
            .arg_ptr(b.up)
            .arg_ptr(b.act)
            .arg_u32(n)
            .arg_f32(LIMIT)
            .launch(s)?;
        let d = segs(&[w.down], &[b.out], &[HIDDEN]);
        ops::w4a16_gemv_pipe(g, b.act, &d, m, inter, sms, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn segment(
        &self,
        g: &dyn GpuBackend,
        b: &Bufs,
        w: &W,
        m: u32,
        inter: u32,
        limit: f32,
        s: u64,
    ) -> Result<()> {
        let gu = segs(&[w.gate, w.up], &[b.gate, b.up], &[inter, inter]);
        let d = segs(&[w.down], &[b.out], &[HIDDEN]);
        ops::w4a16_dense_segment(
            g,
            self.seg[m as usize - 1],
            b.x,
            gu[0],
            gu[1],
            b.act,
            limit,
            d[0],
            m,
            HIDDEN,
            g.sm_count()?,
            self.ctl.take(),
            s,
        )
    }
}

/// 2026-10-23: The four buffers the segment writes, read back, after `fill` poisons them.
fn snapshot(g: &dyn GpuBackend, s: u64, b: &Bufs, m: u32, inter: u32) -> Result<Vec<Vec<u8>>> {
    let mi = (m * inter * 2) as usize;
    let mh = (m * HIDDEN * 2) as usize;
    Ok(vec![
        read(g, s, b.gate, mi)?,
        read(g, s, b.up, mi)?,
        read(g, s, b.act, mi)?,
        read(g, s, b.out, mh)?,
    ])
}

fn poison(g: &dyn GpuBackend, b: &Bufs, v: u8) -> Result<()> {
    let n = 2 * 4096 * 2;
    for p in [b.gate, b.up, b.act, b.out] {
        g.memset(p, v, n)?;
    }
    Ok(())
}

fn compare(what: &str, got: &[Vec<u8>], want: &[Vec<u8>]) -> Result<usize> {
    const NAMES: [&str; 4] = ["gate", "up", "act", "out"];
    let mut n = 0;
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        if let Some(at) = x.chunks(2).zip(y.chunks(2)).position(|(p, q)| p != q) {
            bail!("{what}: {} element {at} differs from the three launches", NAMES[i]);
        }
        n += y.len() / 2;
    }
    Ok(n)
}

fn byte_gate(g: &dyn GpuBackend, s: u64, a: &Arms, b: &Bufs, pool: &Pool, rng: &mut Rng) -> Result<usize> {
    let mut checked = 0;
    for (wi, inter) in INTERS.into_iter().enumerate() {
        let w = W {
            gate: pool.weight(3 * wi, inter, HIDDEN),
            up: pool.weight(3 * wi + 1, inter, HIDDEN),
            down: pool.weight(3 * wi + 2, HIDDEN, inter),
        };
        for m in [1u32, 2] {
            for case in ["random", "large"] {
                let mut x = bf16_rows(rng, m as usize, HIDDEN as usize);
                if case == "large" {
                    // 2026-10-23: x scaled by 2^6 (exponent + 6) so gate/up pass the clamp.
                    for c in x.chunks_mut(2) {
                        let v = u16::from_le_bytes([c[0], c[1]]);
                        let e = (v >> 7) & 0xFF;
                        if e > 0 && e < 0xF0 {
                            c.copy_from_slice(&(v + (6 << 7)).to_le_bytes());
                        }
                    }
                }
                g.copy_h2d(&x, b.x)?;
                poison(g, b, 0xA5)?;
                a.three(g, b, &w, m, inter, s)?;
                let want = snapshot(g, s, b, m, inter)?;
                poison(g, b, 0x5A)?;
                for _ in 0..6 {
                    a.segment(g, b, &w, m, inter, LIMIT, s)?;
                }
                let what = format!("inter={inter} M={m} {case} eager x6");
                checked += compare(&what, &snapshot(g, s, b, m, inter)?, &want)?;
                poison(g, b, 0x3C)?;
                g.begin_capture(s)?;
                a.segment(g, b, &w, m, inter, LIMIT, s)?;
                let graph = g.end_capture(s)?;
                for r in 0..3 {
                    g.launch_graph(graph, s)?;
                    let what = format!("inter={inter} M={m} {case} graph replay {r}");
                    checked += compare(&what, &snapshot(g, s, b, m, inter)?, &want)?;
                }
                g.destroy_graph(graph)?;
            }
        }
    }
    Ok(checked)
}

fn negative_controls(g: &dyn GpuBackend, s: u64, a: &Arms, b: &Bufs, pool: &Pool, rng: &mut Rng) -> Result<()> {
    let inter = 768u32;
    let w = W {
        gate: pool.weight(20, inter, HIDDEN),
        up: pool.weight(21, inter, HIDDEN),
        down: pool.weight(22, HIDDEN, inter),
    };
    let mut x = bf16_rows(rng, 2, HIDDEN as usize);
    g.copy_h2d(&x, b.x)?;
    let run = |limit: f32| -> Result<Vec<u8>> {
        a.segment(g, b, &w, 2, inter, limit, s)?;
        read(g, s, b.out, 2 * HIDDEN as usize * 2)
    };
    let base = run(LIMIT)?;
    ensure!(run(0.25)? != base, "negative control: the limit must reach phase 1");
    let half = HIDDEN as usize * 2;
    let at = (HIDDEN as usize + 5) * 2;
    x[at..at + 2].copy_from_slice(&0x4000u16.to_le_bytes());
    g.copy_h2d(&x, b.x)?;
    let moved = run(LIMIT)?;
    ensure!(
        moved[..half] == base[..half] && moved[half..] != base[half..],
        "negative control: a changed row-1 x element must change row 1 only"
    );
    println!("negative controls fire: the limit reaches phase 1; row isolation through 3 phases");
    Ok(())
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let arms = Arms {
        swiglu: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
        seg: [
            g.kernel("glm5next_dense_seg", "glm5next_dense_seg_pipe_m1")?,
            g.kernel("glm5next_dense_seg", "glm5next_dense_seg_pipe_m2")?,
        ],
        ctl: SegmentCtl::new(&g, 8)?,
    };
    let mut rng = Rng(0x6465_6e73_655f_7365);
    let pool = Pool::new(&g, &mut rng)?;
    let b = Bufs {
        x: g.alloc(2 * HIDDEN as usize * 2)?,
        gate: g.alloc(2 * 4096 * 2)?,
        up: g.alloc(2 * 4096 * 2)?,
        act: g.alloc(2 * 4096 * 2)?,
        out: g.alloc(2 * 4096 * 2)?,
    };
    let checked = byte_gate(&g, s, &arms, &b, &pool, &mut rng)?;
    println!(
        "byte gate: {checked} values (gate, up, act, out) of the segment equal the three launches \
         (inter {INTERS:?}, M 1/2, 2 input cases, eager x6 + graph x3)"
    );
    negative_controls(&g, s, &arms, &b, &pool, &mut rng)?;

    g.copy_h2d(&bf16_rows(&mut rng, 2, HIDDEN as usize), b.x)?;
    for inter in [768u32, 4096] {
        let per = (2 * inter as usize * HIDDEN as usize + HIDDEN as usize * inter as usize) / 2;
        let copies = (96usize << 20).div_ceil(per).clamp(4, 1024);
        for m in [1u32, 2] {
            let mut med = [Vec::new(), Vec::new()];
            for _ in 0..ROUNDS {
                for (ai, seg) in [false, true].into_iter().enumerate() {
                    let mut i = 0usize;
                    let t = timing::time_graph(&g, s, copies, &mut |st| {
                        let w = W {
                            gate: pool.weight(3 * i, inter, HIDDEN),
                            up: pool.weight(3 * i + 1, inter, HIDDEN),
                            down: pool.weight(3 * i + 2, HIDDEN, inter),
                        };
                        i += 1;
                        if seg {
                            arms.segment(&g, &b, &w, m, inter, LIMIT, st)
                        } else {
                            arms.three(&g, &b, &w, m, inter, st)
                        }
                    })?;
                    med[ai].push(t.0);
                }
            }
            let (old, new) = (stat(&mut med[0]), stat(&mut med[1]));
            timing::report(&format!("dense MLP inter={inter}"), m as usize, old, new);
            println!(
                "M={m} inter={inter} per step at {PER_STEP} sites: saves {:.1} us",
                (old.0 - new.0) * PER_STEP
            );
        }
    }
    Ok(())
}
