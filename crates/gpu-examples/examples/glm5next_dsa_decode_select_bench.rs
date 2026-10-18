// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Microbench of one decode row's token selection at a long context, as a captured
//! decode step launches it: `select_tokens` with `DsaSelectLaunch::Ceiling` at a 1,048,576-token
//! ceiling, the live extents in `geom_dev` from `dsa_write_geom` (compress, scores, top-k,
//! expand). An MTP verify of K rows runs K of these per DSA layer, one after another.
//!
//! Per sequence length (the first argument, comma-separated): the row at the last token of a
//! random paged indexer cache laid out as the serve lays it (64-token blocks of keys then gates
//! in an FP8 V pool, a shuffled block table, no validity bytes); `check` also runs the exact launch and requires
//! the selected tokens to be byte-identical, and prints their FNV-1a (the reference a decode
//! lever must reproduce). Timing: median of 7 synchronous runs, spread = max - min. The per-kernel
//! split comes from running it under `nsys profile` with `nocheck` (only ceiling launches).
//!
//! Owner: model-arch examples.
//! Invariants: exits with an error when a `check` comparison differs or nothing is selected.
//!
//! Run (GB10, < 1 GB of device memory):
//!   cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!     --example glm5next_dsa_decode_select_bench -- 262144,524288,1048576 check

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::paged::{IndexerRowsDev, PagedIndexerLayout};
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, contiguous_pool_count,
    select_tokens, topk_tile,
};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// 2026-10-17: The 1M serve's `--max-seq-len` (the decode row's ceiling launch).
const CEILING: usize = 1_048_576;

/// 2026-10-17: GLM-5.3's indexer geometry from its config.json (the attention fields are not
/// read by the selection).
fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: CEILING,
    }
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn f32s(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(f32::to_le_bytes).collect()
}

fn bf16s(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(|f| half::bf16::from_f32(f).to_le_bytes())
        .collect()
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// 2026-10-17: The serve's KV block (tokens) and an FP8 V block's bytes (512 per token).
const BLOCK: usize = 64;
const BLOCK_BYTES: usize = BLOCK * 512;

/// 2026-10-17: A `CEILING`-token random paged cache (blocks in a shuffled order) and one query
/// row; `q_pos` and `geom_dev` are set per length.
fn inputs(gpu: &dyn GpuBackend, c: &Glm5NextDsaConfig) -> Result<DsaSelectInputs> {
    let (d, h) = (c.index_head_dim, c.index_heads);
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let blocks = CEILING / BLOCK;
    let pool = up(gpu, &bf16s((0..blocks * BLOCK_BYTES / 2).map(|_| r.unit())))?;
    let mut bt: Vec<i32> = (0..blocks as i32).collect();
    for i in (1..blocks).rev() {
        let j = ((r.unit() * 0.5 + 0.5) * (i + 1) as f32) as usize;
        bt.swap(i, j.min(i));
    }
    let bt = up(
        gpu,
        &bt.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>(),
    )?;
    let layout = PagedIndexerLayout::new(BLOCK, BLOCK_BYTES, d)?;
    Ok(DsaSelectInputs {
        rows: IndexerRowsDev::paged(&layout, pool, bt)?,
        ape: up(gpu, &f32s((0..c.index_kpool * d).map(|_| r.unit() * 0.1)))?,
        q: up(gpu, &f32s((0..h * d).map(|_| r.unit())))?,
        weights: up(gpu, &f32s((0..h).map(|_| r.unit().abs() * 0.2)))?,
        q_pos: gpu.alloc(16)?,
        q_mask: up(gpu, &[1u8; 16])?,
        first_key: 0,
        geom_dev: gpu.alloc(5 * 4)?,
    })
}

/// 2026-10-17: (median, max - min) ms of 7 synchronous runs.
fn time_ms(gpu: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<(f64, f64)> {
    let mut v = Vec::new();
    for _ in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        f()?;
        gpu.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    v.sort_by(f64::total_cmp);
    Ok((v[3], v[6] - v[0]))
}

fn tokens(gpu: &dyn GpuBackend, s: &DsaSelectScratch, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(s.tokens(), &mut b)?;
    Ok(b)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [seqs, mode] = args.as_slice() else {
        bail!("usage: glm5next_dsa_decode_select_bench <seq,seq,...> <check|nocheck>");
    };
    let check = match mode.as_str() {
        "check" => true,
        "nocheck" => false,
        m => bail!("mode {m}: expected check or nocheck"),
    };
    let seqs: Vec<usize> = seqs
        .split(',')
        .map(|s| s.parse().context("sequence length"))
        .collect::<Result<_>>()?;
    let set = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(0, &set.modules)?;
    let c = cfg();
    let kern = Glm5NextDsaKernels::resolve(&gpu)?;
    let inp = inputs(&gpu, &c)?;
    let scratch = DsaSelectScratch::alloc(
        &gpu,
        &c,
        &DsaSelectGeometry::plan(&c, CEILING, 1)?,
        metrale_gpu_examples::glm_levers()?.dsa.select,
    )?;
    let sl = gpu.alloc(16)?;
    let n = c.out_width() * 4;
    let launch = DsaSelectLaunch::Ceiling {
        max_pools: contiguous_pool_count(c.index_kpool, CEILING),
    };
    println!("ceiling {CEILING} tokens, one decode row, paged cache");
    let mut bad = 0usize;
    for &seq in &seqs {
        ensure!(seq > 0 && seq <= CEILING, "seq {seq} outside 1..={CEILING}");
        let geom = DsaSelectGeometry::plan(&c, seq, 1)?;
        gpu.copy_h2d(&(seq as i32 - 1).to_le_bytes(), inp.q_pos)?;
        gpu.copy_h2d(&(seq as i32).to_le_bytes(), sl)?;
        KernelLaunch::new(&gpu, kern.write_geom)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(sl)
            .arg_ptr(inp.geom_dev)
            .arg_u32(c.index_kpool as u32)
            .arg_u32(c.index_topk as u32)
            .arg_u32(topk_tile() as u32)
            .launch(0)?;
        let run = || select_tokens(&gpu, &kern, &c, &geom, &inp, &scratch, launch, 0);
        gpu.memset(scratch.tokens(), 0xEE, n)?;
        run()?;
        let got = tokens(&gpu, &scratch, n)?;
        let picked = got
            .chunks(4)
            .filter(|t| i32::from_le_bytes([t[0], t[1], t[2], t[3]]) >= 0)
            .count();
        ensure!(picked > 0, "seq {seq}: nothing selected");
        let same = if check {
            let exact = DsaSelectInputs {
                geom_dev: DevicePtr::NULL,
                ..inp
            };
            gpu.memset(scratch.tokens(), 0x11, n)?;
            select_tokens(
                &gpu,
                &kern,
                &c,
                &geom,
                &exact,
                &scratch,
                DsaSelectLaunch::Exact,
                0,
            )?;
            let want = tokens(&gpu, &scratch, n)?;
            bad += usize::from(want != got);
            if want == got { "same" } else { "DIFFER" }
        } else {
            "unchecked"
        };
        let (med, spread) = time_ms(&gpu, &run)?;
        println!(
            "seq {seq} pools {} selected {picked} tokens fnv {:016x} exact {same} \
             ceiling select {med:.3} ms (spread {spread:.3})",
            geom.n_pools,
            fnv(&got)
        );
    }
    ensure!(
        bad == 0,
        "{bad} lengths: the ceiling selection differs from the exact one"
    );
    println!("RESULT: PASS");
    Ok(())
}
