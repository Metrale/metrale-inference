// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tier-1 gate of `METRALE_GLM_DSA_SELECT_FUSED`: `dsa_select_fused_pipe` against
//! the path it replaces, `dsa_index_scores_mma_pipe` + `dsa_topk_pools_radix` + `dsa_sel_fold`,
//! on synthetic prepared pools (BF16 keys and visibility positions, as `dsa_pool_prep_bf16`
//! leaves them).
//!
//! 1. Every selected-pool slot of the fused kernel equals the reference's (outputs pre-filled
//!    with different patterns), and both equal a host selection over the reference's score bits
//!    in the defined order: score descending (-0.0 equal to +0.0), then pool index ascending,
//!    folded (a non-candidate, a slot past `min((q_pos + 1) / kpool, select_k)` or a masked row
//!    is -1). Shapes: random keys; keys repeated in 7 classes (ties at every score); zero keys
//!    (a row scores +0.0 everywhere, or -0.0 where its head weights are all negative); a ramp
//!    (scores rising with the pool, the order that compacts the streaming buffer most); fewer
//!    pools than `select_k`; pools not a multiple of 64; rows not a multiple of 8; shuffled
//!    positions; masked rows; leading invisible pools.
//! 2. The tie shapes must cut through a tie in some row (equal scores selected and not
//!    selected), so the tie-break is exercised, not assumed.
//! 3. Negative controls: one selected pool made invisible, and one unselected candidate given
//!    twice the best key, each change the fused output.
//! 4. Timing at a 32k / 256k / 512k / 1M-token context (2,048 causal rows at the context's end,
//!    median of 7 with min..max): the reference in the row sub-passes a 1M-ceiling scratch
//!    allows (256 x 262,144 scores), then fused in one launch; both outputs compared in full.
//!
//! Owner: model-arch examples.
//! Invariants: exits with an error unless every check holds; the last line reads
//! `RESULT: PASS` only then. Peak device memory is about 0.6 GB.
//!
//! Run (GB10):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_dsa_select_fused_gate

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::select::{SCORES_PIPE_SMEM, SELECT_FUSED_SMEM};
use metrale_model_arch::glm5next_dsa::{
    DSA_ROWS_MODULE, DSA_SCORES_PIPE_MODULE, DSA_SEL_FOLD_MODULE,
};

mod bench;

const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;
/// 2026-10-17: The scores a 1M-ceiling selection scratch holds (`select_scratch_rows`: 8,208
/// rows x 8,192 pools rescaled to 262,144 pools is 256 rows).
const SCRATCH_SCORES: usize = 256 * 262_144;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// 2026-10-17: Round-to-nearest-even FP32 -> BF16 bits (finite inputs).
fn bf16(f: f32) -> u16 {
    let b = f.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let n = std::mem::size_of_val(v);
    // 2026-10-17: SAFETY: `v` is a live slice of `Copy` plain data; the byte view covers exactly
    // its `n` bytes and is only read.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
    let p = gpu.alloc(n.max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn read_i32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// 2026-10-17: `dsa_tk`, the kernels' order-preserving key (-0.0 keyed as +0.0).
fn tk(f: f32) -> u32 {
    let u = if f.to_bits() == 0x8000_0000 {
        0
    } else {
        f.to_bits()
    };
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

struct Kernels {
    pipe: KernelHandle,
    radix: KernelHandle,
    fold: KernelHandle,
    fused: KernelHandle,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Keys {
    Random,
    Dup,
    Zero,
    Ramp,
}

/// 2026-10-17: One shape: `q` rows over `p` pools selecting `k`; `first_key` leading tokens
/// invisible; `shuffled` draws positions at random instead of the causal `4 p - q + r`.
struct Shape {
    q: usize,
    p: usize,
    k: usize,
    keys: Keys,
    first_key: usize,
    shuffled: bool,
}

/// 2026-10-17: Device inputs of one shape, plus the host copies the oracle reads.
struct Io {
    q: DevicePtr,
    w: DevicePtr,
    keys: DevicePtr,
    vis: DevicePtr,
    qpos: DevicePtr,
    qmask: DevicePtr,
    out: DevicePtr,
    cand: DevicePtr,
    sel: DevicePtr,
    rows: usize,
    pools: usize,
    k: usize,
    per: usize,
    h_keys: Vec<u16>,
    h_vis: Vec<i32>,
    h_qpos: Vec<i32>,
    h_qmask: Vec<u8>,
}

fn build(gpu: &dyn GpuBackend, sh: &Shape, per: usize, rng: &mut Rng) -> Result<Io> {
    let (rows, pools) = (sh.q, sh.p);
    let s = pools * KP;
    let h_keys: Vec<u16> = match sh.keys {
        Keys::Random => (0..pools * D).map(|_| bf16(rng.unit())).collect(),
        Keys::Dup => {
            let base: Vec<u16> = (0..7 * D).map(|_| bf16(rng.unit())).collect();
            (0..pools * D)
                .map(|i| base[(i / D % 7) * D + i % D])
                .collect()
        }
        Keys::Zero => vec![0; pools * D],
        Keys::Ramp => (0..pools * D)
            .map(|i| bf16((i / D) as f32 / pools as f32))
            .collect(),
    };
    let h_vis: Vec<i32> = (0..pools)
        .map(|p| {
            let end = KP * p + KP - 1;
            let ok = end >= sh.first_key && !rng.next().is_multiple_of(100);
            if ok { end as i32 } else { -1 }
        })
        .collect();
    let h_qpos: Vec<i32> = (0..rows)
        .map(|r| {
            if sh.shuffled {
                (rng.next() % s as u64) as i32
            } else {
                (s - rows + r) as i32
            }
        })
        .collect();
    let h_qmask: Vec<u8> = (0..rows)
        .map(|_| u8::from(!rng.next().is_multiple_of(33)))
        .collect();
    let ramp = sh.keys == Keys::Ramp;
    let q: Vec<f32> = (0..rows * H * D)
        .map(|_| if ramp { 1.0 } else { rng.unit() })
        .collect();
    // 2026-10-17: a zero-key row scores -0.0 only when every head weight is negative (a signed-zero
    // sum is +0.0 otherwise), so odd rows of the zero-key shape take negative weights.
    let neg = sh.keys == Keys::Zero;
    let w: Vec<f32> = (0..rows * H)
        .map(|i| {
            if ramp {
                1.0 / H as f32
            } else if neg && (i / H) % 2 == 1 {
                -(rng.unit().abs() + 0.01) * 0.2
            } else {
                rng.unit() * 0.2
            }
        })
        .collect();
    Ok(Io {
        q: up(gpu, &q)?,
        w: up(gpu, &w)?,
        keys: up(gpu, &h_keys)?,
        vis: up(gpu, &h_vis)?,
        qpos: up(gpu, &h_qpos)?,
        qmask: up(gpu, &h_qmask)?,
        out: gpu.alloc(per * pools * 4)?,
        cand: gpu.alloc(per * pools)?,
        sel: gpu.alloc(rows * sh.k * 4)?,
        rows,
        pools,
        k: sh.k,
        per,
        h_keys,
        h_vis,
        h_qpos,
        h_qmask,
    })
}

fn scale() -> f32 {
    (D as f32).powf(-0.5)
}

/// 2026-10-17: The reference over rows [r0, r0 + n): scores into the scratch, then the radix
/// select and the fold into those rows of `io.sel`.
fn reference_pass(gpu: &dyn GpuBackend, k: &Kernels, io: &Io, r0: usize, n: usize) -> Result<()> {
    let sel = io.sel.offset(r0 * io.k * 4);
    let (qpos, qmask) = (io.qpos.offset(r0 * 4), io.qmask.offset(r0));
    KernelLaunch::new(gpu, k.pipe)
        .grid([n.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SCORES_PIPE_SMEM)
        .arg_ptr(io.q.offset(r0 * H * D * 4))
        .arg_ptr(io.keys)
        .arg_ptr(io.w.offset(r0 * H * 4))
        .arg_ptr(io.vis)
        .arg_ptr(qpos)
        .arg_ptr(io.out)
        .arg_ptr(io.cand)
        .arg_u32(n as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_f32(scale())
        .launch(0)?;
    KernelLaunch::new(gpu, k.radix)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.out)
        .arg_ptr(sel)
        .arg_u32(n as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(io.k as u32)
        .launch(0)?;
    KernelLaunch::new(gpu, k.fold)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(sel)
        .arg_ptr(io.cand)
        .arg_ptr(qpos)
        .arg_ptr(qmask)
        .arg_u32(io.pools as u32)
        .arg_u32(KP as u32)
        .arg_u32(io.k as u32)
        .launch(0)?;
    Ok(())
}

fn reference(gpu: &dyn GpuBackend, k: &Kernels, io: &Io) -> Result<()> {
    let mut r0 = 0;
    while r0 < io.rows {
        let n = io.per.min(io.rows - r0);
        reference_pass(gpu, k, io, r0, n)?;
        r0 += n;
    }
    Ok(())
}

fn fused(gpu: &dyn GpuBackend, k: &Kernels, io: &Io) -> Result<()> {
    KernelLaunch::new(gpu, k.fused)
        .grid([io.rows.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SELECT_FUSED_SMEM)
        .arg_ptr(io.q)
        .arg_ptr(io.keys)
        .arg_ptr(io.w)
        .arg_ptr(io.vis)
        .arg_ptr(io.qpos)
        .arg_ptr(io.qmask)
        .arg_ptr(io.sel)
        .arg_u32(io.rows as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32(io.k as u32)
        .arg_f32(scale())
        .launch(0)?;
    Ok(())
}

fn selected(
    gpu: &dyn GpuBackend,
    io: &Io,
    fill: u8,
    f: impl Fn() -> Result<()>,
) -> Result<Vec<i32>> {
    gpu.memset(io.sel, fill, io.rows * io.k * 4)?;
    f()?;
    read_i32(gpu, io.sel, io.rows * io.k)
}

/// 2026-10-17: The host selection over one row's reference scores, in the defined order, folded;
/// and whether the cut went through a tie (an equal score on both sides of it).
fn oracle_row(io: &Io, r: usize, scores: &[f32]) -> (Vec<i32>, bool) {
    let cand = |p: usize| io.h_vis[p] >= 0 && io.h_vis[p] <= io.h_qpos[r];
    let key = |p: usize| {
        if cand(p) {
            tk(scores[p])
        } else {
            tk(-f32::MAX)
        }
    };
    let mut order: Vec<usize> = (0..io.pools).collect();
    let cmp = |a: &usize, b: &usize| key(*b).cmp(&key(*a)).then(a.cmp(b));
    let take = io.k.min(io.pools);
    if take < io.pools {
        order.select_nth_unstable_by(take - 1, cmp);
    }
    order[..take].sort_by(cmp);
    let tie = take < io.pools && {
        let cut = key(order[take - 1]);
        order[take..].iter().any(|&p| key(p) == cut && cand(p)) && cand(order[take - 1])
    };
    let row_k = ((io.h_qpos[r] + 1) as usize / KP).min(io.k);
    let on = io.h_qmask[r] != 0;
    let sel = (0..io.k)
        .map(|j| {
            let p = if j < take { order[j] } else { usize::MAX };
            if on && j < row_k && p != usize::MAX && cand(p) {
                p as i32
            } else {
                -1
            }
        })
        .collect();
    (sel, tie)
}

/// 2026-10-17: Rows where `a` and `b` differ.
fn rows_differ(a: &[i32], b: &[i32], k: usize) -> usize {
    a.chunks(k).zip(b.chunks(k)).filter(|(x, y)| x != y).count()
}

fn free(gpu: &dyn GpuBackend, io: &Io) -> Result<()> {
    for p in [
        io.q, io.w, io.keys, io.vis, io.qpos, io.qmask, io.out, io.cand, io.sel,
    ] {
        gpu.free(p)?;
    }
    Ok(())
}

fn equivalence(gpu: &dyn GpuBackend, k: &Kernels, sh: &Shape, rng: &mut Rng) -> Result<()> {
    let io = build(gpu, sh, sh.q, rng)?;
    let want = selected(gpu, &io, 0x00, || reference(gpu, k, &io))?;
    let got = selected(gpu, &io, 0x5A, || fused(gpu, k, &io))?;
    gpu.synchronize(0)?;
    let mut sb = vec![0u8; io.rows * io.pools * 4];
    gpu.copy_d2h(io.out, &mut sb)?;
    let scores: Vec<f32> = sb
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (mut bad_ref, mut ties) = (0usize, 0usize);
    for r in 0..io.rows {
        let (o, tie) = oracle_row(&io, r, &scores[r * io.pools..(r + 1) * io.pools]);
        bad_ref += usize::from(o[..] != want[r * io.k..(r + 1) * io.k]);
        ties += usize::from(tie);
    }
    let bad = rows_differ(&got, &want, io.k);
    let live = want.iter().filter(|&&p| p >= 0).count();
    let negz = scores.iter().filter(|s| s.to_bits() == 0x8000_0000).count();
    let (mut vis_ctl, mut key_ctl) = (true, true);
    if sh.keys == Keys::Random && sh.p > sh.k {
        // 2026-10-17: 3. Row 0's first selected pool made invisible; then an unselected candidate
        // of row 0 given twice the key of that row's best pool.
        let r = (0..io.rows).find(|&r| want[r * io.k] >= 0).unwrap_or(0);
        let best = want[r * io.k].max(0) as usize;
        let mut vis = io.h_vis.clone();
        vis[best] = -1;
        gpu.copy_h2d(&i32_bytes(&vis), io.vis)?;
        vis_ctl = selected(gpu, &io, 0x5A, || fused(gpu, k, &io))? != got;
        gpu.copy_h2d(&i32_bytes(&io.h_vis), io.vis)?;
        let row = &want[r * io.k..(r + 1) * io.k];
        let other = (0..io.pools).find(|p| {
            io.h_vis[*p] >= 0 && io.h_vis[*p] <= io.h_qpos[r] && !row.contains(&(*p as i32))
        });
        if let Some(o) = other {
            let mut keys = io.h_keys.clone();
            for d in 0..D {
                keys[o * D + d] =
                    bf16(2.0 * f32::from_bits((io.h_keys[best * D + d] as u32) << 16));
            }
            let kb: Vec<u8> = keys.iter().flat_map(|h| h.to_le_bytes()).collect();
            gpu.copy_h2d(&kb, io.keys)?;
            key_ctl = selected(gpu, &io, 0x5A, || fused(gpu, k, &io))? != got;
        } else {
            key_ctl = false;
        }
    }
    println!(
        "Q={:5} P={:6} k={:3} {:?}{} first_key={}: fused rows differ {bad}, reference vs host \
         {bad_ref}, {live} live slots, {ties} rows cut through a tie, {negz} scores -0.0; \
         controls vis={vis_ctl} key={key_ctl}",
        sh.q,
        sh.p,
        sh.k,
        sh.keys,
        if sh.shuffled { " shuffled" } else { "" },
        sh.first_key
    );
    free(gpu, &io)?;
    ensure!(bad == 0, "fused differs from the reference");
    ensure!(bad_ref == 0, "the reference differs from the host order");
    ensure!(live > 0, "no live selected slot");
    ensure!(
        vis_ctl && key_ctl,
        "a negative control left the fused output unchanged"
    );
    if matches!(sh.keys, Keys::Dup | Keys::Zero | Keys::Ramp) {
        ensure!(ties > 0, "a tie shape never cut through a tie");
    }
    if sh.keys == Keys::Zero {
        ensure!(negz > 0, "the zero-key shape produced no -0.0 score");
    }
    Ok(())
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        pipe: gpu.kernel(DSA_SCORES_PIPE_MODULE, "dsa_index_scores_mma_pipe")?,
        radix: gpu.kernel(DSA_ROWS_MODULE, "dsa_topk_pools_radix")?,
        fold: gpu.kernel(DSA_SEL_FOLD_MODULE, "dsa_sel_fold")?,
        fused: gpu.kernel(DSA_SCORES_PIPE_MODULE, "dsa_select_fused_pipe")?,
    };
    let sh = |q, p, k, keys, first_key, shuffled| Shape {
        q,
        p,
        k,
        keys,
        first_key,
        shuffled,
    };
    let shapes = [
        sh(2048, 8192, 512, Keys::Random, 0, false),
        sh(8192, 8192, 512, Keys::Random, 0, false),
        sh(37, 300, 512, Keys::Random, 5, true),
        sh(300, 4096, 64, Keys::Random, 0, true),
        sh(1000, 4100, 512, Keys::Dup, 0, true),
        sh(64, 65_536, 512, Keys::Dup, 0, true),
        sh(256, 2000, 512, Keys::Zero, 0, false),
        sh(512, 8192, 512, Keys::Ramp, 0, false),
        sh(203, 1030, 512, Keys::Random, 1001, true),
    ];
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    for s in &shapes {
        equivalence(&gpu, &k, s, &mut rng)?;
    }
    for ctx in [32_768, 262_144, 524_288, 1_048_576] {
        bench::bench(&gpu, &k, ctx, Keys::Random, &mut rng)?;
    }
    bench::bench(&gpu, &k, 1_048_576, Keys::Ramp, &mut rng)?;
    println!("RESULT: PASS");
    Ok(())
}
