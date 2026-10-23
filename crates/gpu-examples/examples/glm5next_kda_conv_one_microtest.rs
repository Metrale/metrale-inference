// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `causal_conv1d_update_l2norm_tokens_snap` (`kda_conv_snap.cu`, the
//! `METRALE_GLM_KDA_CONV_ONE` verify conv) against the three kernels a KDA verify with
//! intermediates runs today: `causal_conv1d_window_advance_to` per intermediate,
//! `causal_conv1d_update_l2norm_tokens`, `causal_conv1d_window_advance`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the one launch writes the three kernels' bytes: every output row,
//!   every intermediate window and the advanced live window, at the GLM-5.3 TP=3 rank-0 KDA
//!   shape (conv_dim 8448, q|k channels 5632, head_dim 128) and a ragged one, K 2..=4 rows,
//!   d_conv 4 and 3, random and large-magnitude inputs.
//! - Negative control: a changed element of row 1 leaves row 0 and intermediate 0 and changes
//!   row 1 and the live window.
//!
//! Times: both arms captured into a CUDA graph (40 copies), median [min..max] of 5 rounds.
//!
//! Run (GB10): cargo build --release -p metrale-gpu-examples --features cuda,gpu-examples
//!   --example glm5next_kda_conv_one_microtest, then run the binary.

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, bail, ensure};
use fixture::{ROUNDS, Rng, bf16_rows, read, stat, upload};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const HEAD_DIM: u32 = 128;
const EPS: f32 = 1e-6;
/// 2026-10-23: (conv_dim, q|k channels): rank 0 at TP=3 (22 heads of 128 each for q, k, v) and
/// a dim that is not a multiple of 256.
const SHAPES: [(u32, u32); 2] = [(8448, 5632), (6400, 4352)];

struct K {
    to: KernelHandle,
    tokens: KernelHandle,
    advance: KernelHandle,
    one: KernelHandle,
}

struct Case {
    dim: u32,
    qk: u32,
    d_conv: u32,
    rows: u32,
    state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    snaps: [DevicePtr; 3],
}

fn three(g: &dyn GpuBackend, k: &K, c: &Case, s: u64) -> Result<()> {
    for t in 0..c.rows - 1 {
        KernelLaunch::new(g, k.to)
            .grid([div_ceil(c.dim, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c.state)
            .arg_ptr(c.snaps[t as usize])
            .arg_ptr(c.input)
            .arg_u32(c.dim)
            .arg_u32(c.d_conv)
            .arg_u32(t + 1)
            .launch(s)?;
    }
    KernelLaunch::new(g, k.tokens)
        .grid([div_ceil(c.dim, 256), c.rows, 1])
        .block([256, 1, 1])
        .arg_ptr(c.state)
        .arg_ptr(c.input)
        .arg_ptr(c.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(c.out)
        .arg_u32(c.dim)
        .arg_u32(c.d_conv)
        .arg_u32(c.qk)
        .arg_u32(HEAD_DIM)
        .arg_f32(EPS)
        .launch(s)?;
    KernelLaunch::new(g, k.advance)
        .grid([div_ceil(c.dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c.state)
        .arg_ptr(c.input)
        .arg_u32(c.dim)
        .arg_u32(c.d_conv)
        .arg_u32(c.rows)
        .launch(s)
}

fn one(g: &dyn GpuBackend, k: &K, c: &Case, s: u64) -> Result<()> {
    let snap = |t: u32| {
        if t + 1 < c.rows {
            c.snaps[t as usize]
        } else {
            DevicePtr::NULL
        }
    };
    KernelLaunch::new(g, k.one)
        .grid([div_ceil(c.dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c.state)
        .arg_ptr(c.input)
        .arg_ptr(c.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(c.out)
        .arg_ptr(snap(0))
        .arg_ptr(snap(1))
        .arg_ptr(snap(2))
        .arg_u32(c.rows - 1)
        .arg_u32(c.dim)
        .arg_u32(c.d_conv)
        .arg_u32(c.qk)
        .arg_u32(HEAD_DIM)
        .arg_f32(EPS)
        .arg_u32(c.rows)
        .launch(s)
}

/// 2026-10-23: The bytes an arm leaves: (outputs, intermediates, live window).
fn run(
    g: &dyn GpuBackend,
    s: u64,
    c: &Case,
    state0: &[u8],
    arm: &dyn Fn(u64) -> Result<()>,
) -> Result<(Vec<u8>, Vec<Vec<u8>>, Vec<u8>)> {
    let wb = c.dim as usize * c.d_conv as usize * 4;
    g.copy_h2d(state0, c.state)?;
    g.memset(c.out, 0xA5, 4 * c.dim as usize * 2)?;
    for p in c.snaps {
        g.memset(p, 0x5A, wb)?;
    }
    arm(s)?;
    let out = read(g, s, c.out, c.rows as usize * c.dim as usize * 2)?;
    let snaps = (0..c.rows as usize - 1)
        .map(|t| read(g, s, c.snaps[t], wb))
        .collect::<Result<_>>()?;
    Ok((out, snaps, read(g, s, c.state, wb)?))
}

fn f32_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| ((rng.next() as f32 / u32::MAX as f32 - 0.5) * scale).to_le_bytes())
        .collect()
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let k = K {
        to: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance_to")?,
        tokens: g.kernel("kda_conv_tokens", "causal_conv1d_update_l2norm_tokens")?,
        advance: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance")?,
        one: g.kernel("kda_conv_snap", "causal_conv1d_update_l2norm_tokens_snap")?,
    };
    let mut rng = Rng(0x6b64_6163_6f6e_7631);
    let mut checked = 0usize;
    let mut timed = None;
    for (dim, qk) in SHAPES {
        for d_conv in [4u32, 3] {
            let weight = upload(&g, &bf16_rows(&mut rng, dim as usize, d_conv as usize))?;
            let wb = dim as usize * d_conv as usize * 4;
            for rows in 2u32..=4 {
                let c = Case {
                    dim,
                    qk,
                    d_conv,
                    rows,
                    state: g.alloc(wb)?,
                    input: upload(&g, &bf16_rows(&mut rng, rows as usize, dim as usize))?,
                    weight,
                    out: g.alloc(4 * dim as usize * 2)?,
                    snaps: [g.alloc(wb)?, g.alloc(wb)?, g.alloc(wb)?],
                };
                for (case, scale) in [("random", 4.0f32), ("large", 6.0e4)] {
                    let state0 = f32_bytes(&mut rng, wb / 4, scale);
                    let want = run(&g, s, &c, &state0, &|st| three(&g, &k, &c, st))?;
                    let got = run(&g, s, &c, &state0, &|st| one(&g, &k, &c, st))?;
                    if got.0 != want.0 {
                        let at = got
                            .0
                            .chunks(2)
                            .zip(want.0.chunks(2))
                            .position(|(x, y)| x != y);
                        bail!(
                            "dim={dim} d_conv={d_conv} K={rows} {case}: output differs at {at:?}"
                        );
                    }
                    ensure!(
                        got.1 == want.1,
                        "dim={dim} d_conv={d_conv} K={rows} {case}: an intermediate differs"
                    );
                    ensure!(
                        got.2 == want.2,
                        "dim={dim} d_conv={d_conv} K={rows} {case}: the live window differs"
                    );
                    checked += want.0.len() / 2 + (want.1.len() + 1) * wb / 4;
                    if dim == SHAPES[0].0 && d_conv == 4 && rows == 2 && case == "random" {
                        // 2026-10-23: Negative control: row 1 changed.
                        let mut x = vec![0u8; rows as usize * dim as usize * 2];
                        g.copy_d2h(c.input, &mut x)?;
                        let keep = x.clone();
                        let at = (dim as usize + 300) * 2;
                        x[at..at + 2].copy_from_slice(&0x4280u16.to_le_bytes());
                        g.copy_h2d(&x, c.input)?;
                        let moved = run(&g, s, &c, &state0, &|st| one(&g, &k, &c, st))?;
                        g.copy_h2d(&keep, c.input)?;
                        let half = dim as usize * 2;
                        ensure!(
                            moved.0[..half] == got.0[..half]
                                && moved.0[half..] != got.0[half..]
                                && moved.1[0] == got.1[0]
                                && moved.2 != got.2,
                            "negative control: a changed row-1 input must change row 1 and the \
                             live window only"
                        );
                        println!("negative control fires: row 1 and the live window only");
                        timed = Some((state0.clone(), c));
                        break;
                    }
                }
            }
        }
    }
    println!(
        "byte gate: {checked} values of the one launch equal the three kernels' (2 shapes, d_conv \
         4/3, K 2..=4, 2 input cases)"
    );
    let Some((state0, c)) = timed else {
        bail!("the timed case was not built");
    };
    g.copy_h2d(&state0, c.state)?;
    let mut med = [Vec::new(), Vec::new()];
    for _ in 0..ROUNDS {
        med[0].push(timing::time_graph(&g, s, 40, &mut |st| three(&g, &k, &c, st))?.0);
        med[1].push(timing::time_graph(&g, s, 40, &mut |st| one(&g, &k, &c, st))?.0);
    }
    let (old, new) = (stat(&mut med[0]), stat(&mut med[1]));
    timing::report("kda verify conv (3 -> 1)", 2, old, new);
    println!(
        "per step at 34 KDA layers: saves {:.1} us",
        (old.0 - new.0) * 34.0
    );
    Ok(())
}
