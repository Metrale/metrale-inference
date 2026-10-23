// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte gate and microbench of `dsa_index_scores_heads_paged` (one lane per index
//! head, `METRALE_GLM_DSA_SCORES_HEADS`) against `dsa_index_scores_warp_paged`, the captured
//! decode's ceiling scores over the persisted pool keys: GLM-5.3's indexer (32 heads of 128,
//! pools of 4 tokens), 1 or 2 query rows (a decode row, a verify's rows), a 1M-token ceiling
//! (262,144 pools as the row stride), the live pools in each row's device geometry.
//! 2026-10-22: And `dsa_index_scores_heads_window` (its straight-line walk,
//! `METRALE_GLM_DSA_SCORES_HEADS_WINDOW`) against the same reference.
//! 2026-10-23: And their BF16-key instances (`dsa_scores_heads_bf16`,
//! `METRALE_GLM_DSA_POOL_KEYS_BF16`) over the keys rounded to BF16 (RN) in the same slots,
//! against `dsa_index_scores_warp_paged` over those keys widened back to FP32: the same bytes.
//! And `dsa_pool_keys_store_rows_bf16` against the RN rounding of `dsa_pool_keys_store_rows`.
//!
//! 1. Identity: for each context and row count, scores and candidacy bytes over the whole
//!    ceiling rows (untouched slots included, both prefilled with the same sentinel) must be
//!    identical, on random keys and on adversarial pools (zero, signed-zero, huge, tiny, and
//!    sign-cancelling keys; negative and zero head weights; rows whose position leaves
//!    pools out); 2026-10-22: the block table carries entries past the sequence's blocks
//!    that index far outside the key pool, so a read of one faults.
//! 2. Timing: each kernel's launch captured 20 times in a CUDA graph over rotating copies of
//!    the keys (128 MB, so they stream from DRAM as in the serve), replayed; us per launch and
//!    the persisted keys' GB/s.
//!
//! Owner: model-arch examples.
//! Invariants: returns an error unless every compared byte matches.
//!
//! Run (GB10, < 1 GB of device memory):
//!   cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!     --example glm5next_dsa_ceiling_scores_gate

#[path = "common/dsa_mma_paged_gate.rs"]
mod mma_paged;
#[path = "common/dsa_pool_keys_store_gate.rs"]
mod store;

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;
const SEL: usize = 2048 / KP;
const MAX_POOLS: usize = 262_144;
/// 2026-10-21: Tokens per KV block and the persisted keys' f32 stride between blocks (padded
/// past the block's 16 keys, so a dense-layout assumption would read the wrong bytes).
const BS: usize = 64;
const PK_BLK: usize = (BS / KP) * D + 64;
const REPS: usize = 20;
/// 2026-10-21: Bytes of key copies a timed graph cycles through (several times GB10's L2).
const L2_DEFEAT_BYTES: usize = 128 << 20;
/// 2026-10-22: Block-table entries past the sequence's blocks, each far past the key pool.
const BT_TAIL: usize = 8;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.unit() + 1.0) * 0.5 * n as f32) as usize % n.max(1)
    }
}

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let n = std::mem::size_of_val(v);
    // 2026-10-21: SAFETY: `v` is a live slice of `Copy` plain data; the byte view covers exactly
    // its `n` bytes and is only read.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
    let p = gpu.alloc(n.max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-21: One sequence's persisted keys, block table, queries and per-row geometry.
struct Case {
    q: DevicePtr,
    pk: DevicePtr,
    /// 2026-10-21: Copies of `pk` the timed launches rotate over, so the keys come from DRAM as
    /// in the serve (11 layers' keys do not stay in L2), not from L2.
    pk_copies: Vec<DevicePtr>,
    /// 2026-10-23: The keys rounded to BF16 in the same slots, and widened back to FP32.
    pk16: DevicePtr,
    pk16_copies: Vec<DevicePtr>,
    pkw: DevicePtr,
    w: DevicePtr,
    bt: DevicePtr,
    qpos: DevicePtr,
    geom: DevicePtr,
    out: DevicePtr,
    cand: DevicePtr,
    rows: usize,
    pools: usize,
    /// 2026-10-23: The context, and the BF16 keys as dsa_index_scores_mma_pipe takes them
    /// (`[pools, D]` contiguous) with each pool's visibility (its last token).
    s: usize,
    keys16: DevicePtr,
    vis: DevicePtr,
}

/// 2026-10-21: Pool `p`'s key under the adversarial mix (every 9th pool special), else random.
fn key(rng: &mut Rng, p: usize, d: usize) -> f32 {
    match (p % 97, p % 9) {
        (0, _) => 0.0,
        (1, _) => -0.0,
        (2, _) => 3.0e19 * if d.is_multiple_of(2) { 1.0 } else { -1.0 },
        (4, _) => 3.0e38 * if d.is_multiple_of(3) { 1.0 } else { -1.0 },
        (3, _) => 1.0e-39 * (d as f32 + 1.0),
        (_, 4) => {
            if d < 64 {
                0.5
            } else {
                -0.5
            }
        }
        (_, 5) => rng.unit() * 1.0e-20,
        _ => rng.unit(),
    }
}

/// 2026-10-23: `__float2bfloat16_rn`'s bits of `x` (round to nearest even; a NaN stays a quiet
/// NaN of its sign).
fn bf16_rn(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// 2026-10-23: The persisted keys `pk` as `dsa_pool_keys_store_rows_bf16` lays them out (each
/// key slot of each block (`PK_BLK` f32 apart, `BS / KP` slots of `D`) holds its key rounded
/// to BF16 in its first half, the rest zero), and those keys widened back to FP32 in the FP32
/// layout (the block padding kept).
fn bf16_slots(pk: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut packed = vec![0f32; pk.len()];
    let mut wide = pk.to_vec();
    for blk in 0..pk.len() / PK_BLK {
        for slot in 0..BS / KP {
            let base = blk * PK_BLK + slot * D;
            for d in 0..D {
                let h = bf16_rn(pk[base + d]);
                wide[base + d] = f32::from_bits((h as u32) << 16);
                let word = &mut packed[base + d / 2];
                let bits = word.to_bits();
                *word = f32::from_bits(if d % 2 == 0 {
                    (bits & 0xFFFF_0000) | h as u32
                } else {
                    (bits & 0xFFFF) | ((h as u32) << 16)
                });
            }
        }
    }
    (packed, wide)
}

fn build(gpu: &dyn GpuBackend, rng: &mut Rng, s: usize, rows: usize, short: bool) -> Result<Case> {
    let pools = s / KP;
    let nblk = s.div_ceil(BS);
    let mut perm: Vec<i32> = (0..nblk as i32).collect();
    for i in (1..nblk).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    let mut pk = vec![f32::NAN; nblk * PK_BLK];
    for p in 0..pools {
        let t = p * KP;
        let base = perm[t / BS] as usize * PK_BLK + ((t % BS) / KP) * D;
        for d in 0..D {
            pk[base + d] = key(rng, p, d);
        }
    }
    // 2026-10-21: Head 3 of every row all positive and head 5 all negative, so a zero key makes
    // every product of the head the same signed zero.
    let q: Vec<f32> = (0..rows * H * D)
        .map(|i| match (i / D) % H {
            3 => rng.unit().abs(),
            5 => -rng.unit().abs(),
            _ if i % 251 == 0 => -0.0,
            _ => rng.unit(),
        })
        .collect();
    let w: Vec<f32> = (0..rows * H)
        .map(|i| match i % 13 {
            0 => 0.0,
            1 => -0.0,
            _ => rng.unit() * 0.2,
        })
        .collect();
    // 2026-10-21: Row r of a verify of `rows` rows: its own length S - rows + r + 1; `short`
    // places the query a few tokens before its length, so the last pools are not candidates.
    let mut geom = Vec::new();
    let mut qpos = Vec::new();
    for r in 0..rows {
        let sr = s - (rows - 1 - r);
        let np = sr / KP;
        geom.extend_from_slice(&[
            sr as i32,
            sr.div_ceil(KP) as i32,
            np as i32,
            np.min(SEL) as i32,
            2048,
        ]);
        qpos.push(sr as i32 - 1 - if short { 6 + r as i32 } else { 0 });
    }
    let mut bt = perm.clone();
    bt.extend((0..BT_TAIL).map(|i| i32::MAX - i as i32));
    let pk_dev = up(gpu, &pk)?;
    let copies = (L2_DEFEAT_BYTES / (pk.len() * 4)).clamp(1, 16);
    let mut pk_copies = Vec::new();
    for _ in 0..copies {
        pk_copies.push(up(gpu, &pk)?);
    }
    let (pk16, pkw) = bf16_slots(&pk);
    let mut keys16 = vec![0u16; pools * D];
    for p in 0..pools {
        let t = p * KP;
        let base = perm[t / BS] as usize * PK_BLK + ((t % BS) / KP) * D;
        for d in 0..D {
            keys16[p * D + d] = bf16_rn(pk[base + d]);
        }
    }
    let vis: Vec<i32> = (0..pools).map(|p| (p * KP + KP - 1) as i32).collect();
    let mut pk16_copies = Vec::new();
    for _ in 0..copies {
        pk16_copies.push(up(gpu, &pk16)?);
    }
    Ok(Case {
        q: up(gpu, &q)?,
        pk: pk_dev,
        pk_copies,
        pk16: up(gpu, &pk16)?,
        pk16_copies,
        pkw: up(gpu, &pkw)?,
        w: up(gpu, &w)?,
        bt: up(gpu, &bt)?,
        qpos: up(gpu, &qpos)?,
        geom: up(gpu, &geom)?,
        out: gpu.alloc(rows * MAX_POOLS * 4)?,
        cand: gpu.alloc(rows * MAX_POOLS)?,
        rows,
        pools,
        s,
        keys16: up(gpu, &keys16)?,
        vis: up(gpu, &vis)?,
    })
}

fn launch(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    c: &Case,
    pk: DevicePtr,
    grid: [u32; 3],
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(c.q)
        .arg_ptr(pk)
        .arg_ptr(c.w)
        .arg_ptr(c.bt)
        .arg_ptr(c.qpos)
        .arg_ptr(c.out)
        .arg_ptr(c.cand)
        .arg_u32(c.rows as u32)
        .arg_u32(MAX_POOLS as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32(0)
        .arg_u32(BS as u32)
        .arg_u32(PK_BLK as u32)
        .arg_f32((D as f32).powf(-0.5))
        .arg_ptr(c.geom)
        .launch(stream)
}

/// 2026-10-21: Scores and candidacy of a launch over the whole ceiling rows, from the same
/// sentinel fill.
fn run(gpu: &dyn GpuBackend, k: KernelHandle, c: &Case, grid: [u32; 3]) -> Result<Vec<u8>> {
    run_on(gpu, k, c, c.pk, grid)
}

/// 2026-10-23: [`run`] over the keys at `pk`.
fn run_on(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    c: &Case,
    pk: DevicePtr,
    grid: [u32; 3],
) -> Result<Vec<u8>> {
    gpu.memset(c.out, 0x5a, c.rows * MAX_POOLS * 4)?;
    gpu.memset(c.cand, 0xa5, c.rows * MAX_POOLS)?;
    launch(gpu, k, c, pk, grid, 0)?;
    let mut v = read(gpu, c.out, c.rows * MAX_POOLS * 4)?;
    v.extend(read(gpu, c.cand, c.rows * MAX_POOLS)?);
    Ok(v)
}

/// 2026-10-21: us per launch: REPS launches (`f(i, stream)`) captured in one graph, the median
/// of 5 replays.
fn graph_us(gpu: &dyn GpuBackend, f: &dyn Fn(usize, u64) -> Result<()>) -> Result<f64> {
    let s = gpu.create_stream()?;
    f(0, s)?;
    gpu.synchronize(s)?;
    gpu.begin_capture(s)?;
    for i in 0..REPS {
        f(i, s)?;
    }
    let g = gpu.end_capture(s)?;
    let mut v = Vec::new();
    for _ in 0..5 {
        gpu.synchronize(s)?;
        let t0 = std::time::Instant::now();
        gpu.launch_graph(g, s)?;
        gpu.synchronize(s)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    gpu.destroy_graph(g)?;
    v.sort_by(f64::total_cmp);
    Ok(v[2])
}

fn first_diff(a: &[u8], b: &[u8], rows: usize) -> String {
    let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
    let sb = rows * MAX_POOLS * 4;
    if i < sb {
        format!(
            "score row {} pool {}",
            i / 4 / MAX_POOLS,
            (i / 4) % MAX_POOLS
        )
    } else {
        format!(
            "candidacy row {} pool {}",
            (i - sb) / MAX_POOLS,
            (i - sb) % MAX_POOLS
        )
    }
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let old = gpu.kernel("dsa_scores_warp", "dsa_index_scores_warp_paged")?;
    let heads = gpu.kernel("dsa_scores_warp", "dsa_index_scores_heads_paged")?;
    let window = gpu.kernel("dsa_scores_heads_window", "dsa_index_scores_heads_window")?;
    let news = [("heads", heads), ("window", window)];
    let mma = gpu.kernel("dsa_scores_mma_paged", "dsa_index_scores_mma_paged_bf16")?;
    let pipe = gpu.kernel("dsa_index_scores_pipe", "dsa_index_scores_mma_pipe")?;
    let news16 = [
        (
            "heads_bf16",
            gpu.kernel("dsa_scores_heads_bf16", "dsa_index_scores_heads_paged_bf16")?,
        ),
        (
            "window_bf16",
            gpu.kernel(
                "dsa_scores_heads_bf16",
                "dsa_index_scores_heads_window_bf16",
            )?,
        ),
    ];
    let sms = gpu.sm_count()?;
    let grid_new = |_c: &Case| [sms, 1, 1];
    let grid_old = |c: &Case| [c.pools.div_ceil(8).clamp(1, 192) as u32, c.rows as u32, 1];
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    store::gate(&gpu, &mut rng)?;
    println!("{sms} SMs");
    let ctxs: Vec<usize> = metrale_gpu_examples::env_var("GLM_CTX")?
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1_024, 16_384, 65_536, 131_072, 262_144, 524_288]);
    // 2026-10-21: Identity at odd lengths and short rows too, then the timed contexts.
    for (s, short) in [(9usize, false), (37, true), (1_027, true), (4_099, false)]
        .into_iter()
        .chain(ctxs.iter().map(|&s| (s, false)))
        .chain([(65_539usize, true)])
    {
        for rows in [1usize, 2, 3] {
            let c = build(&gpu, &mut rng, s, rows, short)?;
            let a = run(&gpu, old, &c, grid_old(&c))?;
            for (name, new) in news {
                for g in [grid_new(&c), [1, 1, 1], [3 * sms, 1, 1]] {
                    let b = run(&gpu, new, &c, g)?;
                    ensure!(
                        a == b,
                        "S={s} rows={rows} short={short} grid {g:?}: {name} scores differ from \
                         dsa_index_scores_warp_paged at {}",
                        first_diff(&a, &b, rows)
                    );
                }
            }
            let aw = run_on(&gpu, old, &c, c.pkw, grid_old(&c))?;
            for (name, new) in news16 {
                for g in [grid_new(&c), [1, 1, 1], [3 * sms, 1, 1]] {
                    let b = run_on(&gpu, new, &c, c.pk16, g)?;
                    ensure!(
                        aw == b,
                        "S={s} rows={rows} short={short} grid {g:?}: {name} scores differ from \
                         dsa_index_scores_warp_paged over the widened keys at {}",
                        first_diff(&aw, &b, rows)
                    );
                }
            }
            mma_paged::check(
                &gpu,
                mma,
                pipe,
                &c,
                &[grid_new(&c), [1, 1, 1], [3 * sms, 1, 1]],
            )?;
            println!(
                "S={s:7} rows={rows} short={short:5}: identical (grids {sms}, 1, {}; FP32 and BF16 keys; mma = pipe)",
                3 * sms
            );
            if ctxs.contains(&s) && rows <= 2 && !short {
                let pk = |i: usize| c.pk_copies[i % c.pk_copies.len()];
                let to = graph_us(&gpu, &|i, st| {
                    launch(&gpu, old, &c, pk(i), grid_old(&c), st)
                })?;
                let tn = graph_us(&gpu, &|i, st| {
                    launch(&gpu, heads, &c, pk(i), grid_new(&c), st)
                })?;
                let tw = graph_us(&gpu, &|i, st| {
                    launch(&gpu, window, &c, pk(i), grid_new(&c), st)
                })?;
                let pk16 = |i: usize| c.pk16_copies[i % c.pk16_copies.len()];
                let k16 = news16[if rows == 1 { 0 } else { 1 }];
                let t16 = graph_us(&gpu, &|i, st| {
                    launch(&gpu, k16.1, &c, pk16(i), grid_new(&c), st)
                })?;
                let tm = graph_us(&gpu, &|i, st| {
                    launch(&gpu, mma, &c, pk16(i), grid_new(&c), st)
                })?;
                let gb = (c.pools * D * 4) as f64 / 1e3;
                println!(
                    "  TIME S={s:7} rows={rows}: warp_paged {to:8.1} us ({:5.1} GB/s) | heads {tn:8.1} us ({:5.1} GB/s) | window {tw:8.1} us ({:5.1} GB/s) | window/heads {:.3}",
                    gb / to,
                    gb / tn,
                    gb / tw,
                    tw / tn
                );
                println!(
                    "  TIME S={s:7} rows={rows}: {} {t16:8.1} us ({:5.1} GB/s of BF16 keys) | vs {} {:.3}",
                    k16.0,
                    gb / 2.0 / t16,
                    if rows == 1 { "heads" } else { "window" },
                    t16 / if rows == 1 { tn } else { tw }
                );
                println!(
                    "  TIME S={s:7} rows={rows}: mma_paged_bf16 {tm:8.1} us ({:5.1} GB/s of BF16 keys) | vs window {:.3}",
                    gb / 2.0 / tm,
                    tm / tw
                );
            }
            for p in [
                c.q, c.pk, c.pk16, c.pkw, c.w, c.bt, c.qpos, c.geom, c.out, c.cand, c.keys16, c.vis,
            ]
            .into_iter()
            .chain(c.pk_copies.iter().copied())
            .chain(c.pk16_copies.iter().copied())
            {
                gpu.free(p)?;
            }
        }
    }
    println!(
        "PASS: dsa_index_scores_heads_paged and dsa_index_scores_heads_window are \
         dsa_index_scores_warp_paged's bytes; their BF16-key instances are its bytes over the \
         widened keys"
    );
    Ok(())
}
