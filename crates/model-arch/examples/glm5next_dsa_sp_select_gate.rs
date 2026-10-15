// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Tier-1 gate and microbench of `METRALE_GLM_DSA_SP_SELECT` on one GPU: the three
//! ranks of a split pass are run one after another on a selection scratch planned as the serve
//! plans it (8,208 prefill rows at a 65,536-token ceiling), from a random flat indexer cache.
//!
//! 1. Bytes: per shape, the unsplit `select_tokens_in_passes` token rows; then for each rank,
//!    with the output pre-filled with a sentinel, `select_tokens_rows` over its two zigzag
//!    slabs (the first compressing the pools), as `sp_select::run::run_split` launches them.
//!    Each rank writes no row it does not own, and the rows the ranks own, put together (what
//!    the two in-place all-gathers do: a byte copy), equal the unsplit rows: 0 rows differ.
//! 2. Negative control: rank 1's first slab selected from inputs shifted by one row differs
//!    from the unsplit rows. Informational: the same slab in 1-row sub-passes (another scores
//!    kernel), which `sp_select::run::plan` refuses to split into.
//! 3. Timing (median of 7 synchronous runs, spread = max - min): the unsplit pass and each
//!    rank's two slabs; the split cost is the slowest rank's.
//! 4. The table: a measured (k, P) point splits iff `unsplit - split - xAG > 2 * spread`
//!    (spread the larger of the two). xAG is the two exchanges' exposed cost from the 3-box
//!    send/recv sweep's linear fit, `XAG_LAT_MS + XAG_MS_PER_MB * MB per peer` each; for each k
//!    the smallest measured P from which every larger measured P splits, else never.
//! 5. The 32k prompt's four full chunks again on a scratch planned at a 524,288-token ceiling
//!    (the 512k serve's): the same bytes as at 65,536, the timings, and the projected 32k delta
//!    over 11 DSA layers at both ceilings, the gathers exposed and hidden.
//! 6. 2026-10-15: The selected-pools payload (`METRALE_GLM_DSA_SP_SELECT_SEL`) on the same
//!    chunks at both ceilings (`common/glm5next_dsa_sp_select_sel.rs`): its bytes, its
//!    unfolded-input negative control, the fold launch, and both payloads' projections.
//!
//! Owner: model-arch examples.
//! Invariants: exits with an error unless every byte check and the negative control hold; the
//! last line reads `RESULT: PASS` only then. Requires `METRALE_GLM_DSA_SCORES_MMA=1`,
//! `METRALE_GLM_DSA_SCORES_PIPE=1` and `METRALE_GLM_DSA_TOPK_RADIX=1` (the serve's kernels).
//!
//! Run (GB10, < 1.5 GB of device memory):
//!   METRALE_GLM_DSA_SCORES_MMA=1 METRALE_GLM_DSA_SCORES_PIPE=1 METRALE_GLM_DSA_TOPK_RADIX=1 \
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_dsa_sp_select_gate

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::layer::select_scratch_rows;
use metrale_model_arch::glm5next_dsa::paged::IndexerRowsDev;
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectScratch, select_tokens_in_passes,
    select_tokens_rows,
};
use metrale_model_arch::glm5next_dsa::sp_select::{MinPoolsTable, owned, run};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// 2026-10-15: The serve's `--max-seq-len` and prefill rows (the recipe's MSL 65536, 8208 rows).
const CEILING: usize = 65_536;
const CEILING_512K: usize = 524_288;
/// 2026-10-15: GLM-5.3's DSA layers (each runs one selection pass per prefill chunk).
const DSA_LAYERS: f64 = 11.0;
/// 2026-10-15: The 32k prompt's full chunks (rows, pools) as the serve runs them (nsys, rank 0):
/// one of 8208 rows, then 8192-row chunks; its 287-row tail never splits.
const C32K: [(usize, usize); 4] = [(8208, 2052), (8192, 4100), (8192, 6148), (8192, 8196)];
const MAX_ROWS: usize = 8208;
const WORLD: usize = 3;
/// 2026-10-15: Linear fit of the 3-box grouped send/recv sweep (one exchange to both peers,
/// NCCL, 4 x 200 Gb/s RoCE): 0.25 ms at 2.02 MB, 0.64-0.70 at 5.60, 1.21-1.36 at 11.21,
/// 2.34-2.55 at 22.41 MB per peer.
const XAG_LAT_MS: f64 = 0.027;
const XAG_MS_PER_MB: f64 = 0.110;
const SENTINEL: u8 = 0xEE;

#[path = "common/glm5next_dsa_sp_select_sel.rs"]
mod sel;

/// 2026-10-15: GLM-5.3's indexer geometry (the attention fields are not read by the selection).
fn cfg(max_context: usize) -> Glm5NextDsaConfig {
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
        max_context,
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

/// 2026-10-15: Every input but `q_pos`, at the largest shape: a `CEILING`-token cache (all keys
/// valid) and `MAX_ROWS` query rows.
fn inputs(gpu: &dyn GpuBackend, c: &Glm5NextDsaConfig) -> Result<DsaSelectInputs> {
    let (d, h) = (c.index_head_dim, c.index_heads);
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let k = up(gpu, &bf16s((0..CEILING * d).map(|_| r.unit())))?;
    let gate = up(gpu, &bf16s((0..CEILING * d).map(|_| r.unit())))?;
    Ok(DsaSelectInputs {
        rows: IndexerRowsDev::flat(k, gate, up(gpu, &vec![1u8; CEILING])?),
        ape: up(gpu, &f32s((0..c.index_kpool * d).map(|_| r.unit() * 0.1)))?,
        q: up(gpu, &f32s((0..MAX_ROWS * h * d).map(|_| r.unit())))?,
        weights: up(gpu, &f32s((0..MAX_ROWS * h).map(|_| r.unit().abs() * 0.2)))?,
        q_pos: gpu.alloc(MAX_ROWS * 4)?,
        q_mask: up(gpu, &vec![1u8; MAX_ROWS])?,
        first_key: 0,
        geom_dev: DevicePtr::NULL,
    })
}

fn tokens(gpu: &dyn GpuBackend, s: &DsaSelectScratch, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(s.tokens(), &mut b)?;
    Ok(b)
}

/// 2026-10-15: (median, max - min) ms of 7 synchronous runs.
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

struct Point {
    k: usize,
    pools: usize,
    unsplit: (f64, f64),
    split: (f64, f64),
    xag: f64,
    /// 2026-10-15: FNV-1a of the unsplit token rows.
    bytes: u64,
}

impl Point {
    fn net(&self) -> f64 {
        self.split.0 + self.xag - self.unsplit.0
    }
    fn hidden(&self) -> f64 {
        self.split.0 - self.unsplit.0
    }
    fn wins(&self) -> bool {
        -self.net() > 2.0 * self.unsplit.1.max(self.split.1)
    }
}

/// 2026-10-15: One shape: `k` causal rows ending at `4 * pools` tokens.
#[allow(clippy::too_many_arguments)]
fn shape(
    gpu: &dyn GpuBackend,
    kern: &Glm5NextDsaKernels,
    c: &Glm5NextDsaConfig,
    inp: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    one_row: Option<&DsaSelectScratch>,
    k: usize,
    pools: usize,
) -> Result<Point> {
    let seq = pools * c.index_kpool;
    let pos: Vec<u8> = (0..k)
        .flat_map(|r| ((seq - k + r) as i32).to_le_bytes())
        .collect();
    gpu.copy_h2d(&pos, inp.q_pos)?;
    let geom = DsaSelectGeometry::plan(c, seq, k)?;
    let rb = c.out_width() * 4;
    let n = k * rb;
    let per = scratch.rows_per_pass(&geom);

    gpu.memset(scratch.tokens(), 0x11, n)?;
    let passes = select_tokens_in_passes(gpu, kern, c, &geom, inp, scratch, 0)?;
    let want = tokens(gpu, scratch, n)?;
    let selected = want
        .chunks(4)
        .filter(|t| i32::from_le_bytes([t[0], t[1], t[2], t[3]]) >= 0)
        .count();
    ensure!(selected > 0, "k {k} P {pools}: nothing was selected");

    let mut assembled = vec![SENTINEL; n];
    let mut foreign = 0;
    let mut rank_passes = Vec::new();
    for rank in 0..WORLD {
        let own = owned(k, WORLD, rank)?;
        gpu.memset(scratch.tokens(), SENTINEL, n)?;
        let a = select_tokens_rows(gpu, kern, c, &geom, inp, scratch, &[own[0]], true, 0)?;
        let b = select_tokens_rows(gpu, kern, c, &geom, inp, scratch, &[own[1]], false, 0)?;
        rank_passes.push(a + b);
        let got = tokens(gpu, scratch, n)?;
        for (r, row) in got.chunks(rb).enumerate() {
            if own.iter().any(|&(lo, hi)| (lo..hi).contains(&r)) {
                assembled[r * rb..(r + 1) * rb].copy_from_slice(row);
            } else if row.iter().any(|&x| x != SENTINEL) {
                foreign += 1;
            }
        }
    }
    let differ = |a: &[u8], b: &[u8]| {
        a.chunks(rb)
            .zip(b.chunks(rb))
            .filter(|(x, y)| x != y)
            .count()
    };
    let bad = differ(&assembled, &want);

    // 2026-10-15: 2. The shifted-input control on rank 1's first slab.
    let s1 = owned(k, WORLD, 1)?[0];
    gpu.memset(scratch.tokens(), SENTINEL, n)?;
    let shifted = inp.from_row(1, c);
    select_tokens_rows(gpu, kern, c, &geom, &shifted, scratch, &[s1], true, 0)?;
    let ctl = tokens(gpu, scratch, n)?;
    let ctl_rows = differ(&ctl[s1.0 * rb..s1.1 * rb], &want[s1.0 * rb..s1.1 * rb]);
    let one_row_rows = match one_row {
        Some(o) => {
            gpu.memset(o.tokens(), SENTINEL, n)?;
            select_tokens_rows(gpu, kern, c, &geom, inp, o, &[s1], true, 0)?;
            let got = tokens(gpu, o, n)?;
            Some(differ(
                &got[s1.0 * rb..s1.1 * rb],
                &want[s1.0 * rb..s1.1 * rb],
            ))
        }
        None => None,
    };

    // 2026-10-15: 3. Timing.
    let unsplit = time_ms(gpu, &|| {
        select_tokens_in_passes(gpu, kern, c, &geom, inp, scratch, 0).map(|_| ())
    })?;
    let mut split = (0.0f64, 0.0f64);
    let mut ranks = Vec::new();
    for rank in 0..WORLD {
        let own = owned(k, WORLD, rank)?;
        let t = time_ms(gpu, &|| {
            select_tokens_rows(gpu, kern, c, &geom, inp, scratch, &[own[0]], true, 0)?;
            select_tokens_rows(gpu, kern, c, &geom, inp, scratch, &[own[1]], false, 0)?;
            Ok(())
        })?;
        ranks.push(t.0);
        if t.0 > split.0 {
            split = t;
        }
    }
    let slab_rows = (0..WORLD)
        .flat_map(|r| owned(k, WORLD, r).unwrap_or_default())
        .map(|(lo, hi)| hi - lo)
        .max()
        .unwrap_or(0);
    let mb = (slab_rows * rb) as f64 / 1e6;
    let xag = 2.0 * (XAG_LAT_MS + XAG_MS_PER_MB * mb);
    let p = Point {
        k,
        pools,
        unsplit,
        split,
        xag,
        bytes: want.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        }),
    };
    println!(
        "k={k:5} P={pools:5} per={per:5} passes={passes} rank_passes={rank_passes:?}: rows differ \
         {bad}, foreign rows written {foreign}, control rows differ {ctl_rows}/{}, 1-row \
         sub-pass rows differ {}",
        s1.1 - s1.0,
        one_row_rows.map_or("-".into(), |r| format!("{r}/{}", s1.1 - s1.0)),
    );
    println!(
        "  unsplit {:8.3} ms (spread {:.3}) | split max {:8.3} ms (spread {:.3}; ranks \
         {:.3}/{:.3}/{:.3}) | xAG exposed {:.3} ms ({mb:.2} MB/peer x2) | net {:+.3} ms -> {}",
        p.unsplit.0,
        p.unsplit.1,
        p.split.0,
        p.split.1,
        ranks[0],
        ranks[1],
        ranks[2],
        p.xag,
        p.net(),
        if p.wins() { "SPLIT" } else { "keep" }
    );
    ensure!(
        bad == 0 && foreign == 0,
        "k {k} P {pools}: the split selection differs from the unsplit one"
    );
    ensure!(
        ctl_rows > 0,
        "k {k} P {pools}: the shifted-input control left the rows unchanged"
    );
    Ok(p)
}

/// 2026-10-15: The serve's selection scratch (`Glm5NextDsaWorkspace::new`) at a ceiling: the
/// ceiling's geometry at `select_scratch_rows` rows, the output at `MAX_ROWS`.
fn serve_scratch(gpu: &dyn GpuBackend, c: &Glm5NextDsaConfig) -> Result<DsaSelectScratch> {
    let full = DsaSelectGeometry::plan(c, c.max_context, MAX_ROWS)?;
    let rows = select_scratch_rows(MAX_ROWS, c.max_context, c.index_kpool, 16);
    println!("scratch rows {rows} at {} tokens", c.max_context);
    DsaSelectScratch::alloc_rows(
        gpu,
        c,
        &DsaSelectGeometry {
            q_rows: rows,
            ..full
        },
        MAX_ROWS,
    )
}

/// 2026-10-15: The 32k prompt's projected selection delta in ms (gathers exposed, hidden) over
/// its full chunks the table splits, 11 layers each.
fn c32k_delta(points: &[&Point], t: &MinPoolsTable) -> (f64, f64) {
    points
        .iter()
        .filter(|p| t.split(p.k, p.pools))
        .fold((0.0, 0.0), |(e, h), p| {
            (e + p.net() * DSA_LAYERS, h + p.hidden() * DSA_LAYERS)
        })
}

fn main() -> Result<()> {
    for v in [
        "METRALE_GLM_DSA_SCORES_MMA",
        "METRALE_GLM_DSA_SCORES_PIPE",
        "METRALE_GLM_DSA_TOPK_RADIX",
    ] {
        ensure!(
            std::env::var(v).as_deref() == Ok("1"),
            "{v}=1 is required (the serve's selection kernels)"
        );
    }
    let set = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(0, &set.modules)?;
    let c = cfg(CEILING);
    let kern = Glm5NextDsaKernels::resolve(&gpu)?;
    let inp = inputs(&gpu, &c)?;
    let scratch = serve_scratch(&gpu, &c)?;
    let small = DsaSelectGeometry::plan(&c, 4096, MAX_ROWS)?;
    let one_row = DsaSelectScratch::alloc_rows(
        &gpu,
        &c,
        &DsaSelectGeometry { q_rows: 1, ..small },
        MAX_ROWS,
    )?;
    println!("world {WORLD}");
    let ks = [1024usize, 2048, 4096, 8192, 8208];
    let ps = [1024usize, 2052, 4100, 6148, 8196, 16384];
    let mut points = Vec::new();
    for &k in &ks {
        for &p in &ps {
            if k > p * c.index_kpool {
                continue;
            }
            let o = (k == 1024 && p == 1024).then_some(&one_row);
            points.push(shape(&gpu, &kern, &c, &inp, &scratch, o, k, p)?);
        }
    }
    let mut table = Vec::new();
    for &k in &ks {
        let at: Vec<&Point> = points.iter().filter(|p| p.k == k).collect();
        let min = (0..at.len())
            .find(|&i| at[i..].iter().all(|p| p.wins()))
            .map_or(u32::MAX as usize, |i| at[i].pools);
        table.push((k, min));
    }
    let t = MinPoolsTable::new(table)?;
    println!(
        "table (gather exposed): METRALE_GLM_DSA_SP_SELECT={}",
        t.canonical()
    );
    for (k, p) in C32K.into_iter().chain([(287, 8268), (4096, 1024)]) {
        let geom = DsaSelectGeometry::plan(&c, p * c.index_kpool, k)?;
        let per = scratch.rows_per_pass(&geom);
        let d = run::plan(Some(&t), WORLD, 0, k, p, true, per)?;
        println!(
            "  plan k={k} P={p} per={per}: {}",
            if d.is_some() { "split" } else { "unsplit" }
        );
    }

    // 2026-10-15: 5. The 32k chunks on the 512k serve's scratch.
    let c512 = cfg(CEILING_512K);
    let scratch512 = serve_scratch(&gpu, &c512)?;
    let mut at512 = Vec::new();
    for (k, p) in C32K {
        let q = shape(&gpu, &kern, &c512, &inp, &scratch512, None, k, p)?;
        let base = points
            .iter()
            .find(|x| x.k == k && x.pools == p)
            .context("a 32k chunk outside the grid")?;
        ensure!(
            q.bytes == base.bytes,
            "k {k} P {p}: the 524,288-ceiling selection differs from the 65,536 one"
        );
        at512.push(q);
    }
    for (name, pts) in [
        (
            "64k",
            C32K.iter()
                .map(|&(k, p)| points.iter().find(|x| x.k == k && x.pools == p))
                .collect::<Option<Vec<_>>>()
                .context("c32k points")?,
        ),
        ("512k", at512.iter().collect()),
    ] {
        let (e, h) = c32k_delta(&pts, &t);
        let un: f64 = pts.iter().map(|p| p.unsplit.0 * DSA_LAYERS).sum();
        println!(
            "c32k projection at {name}: full-chunk selection {un:.1} ms unsplit; delta {e:+.1} ms \
             (gathers exposed), {h:+.1} ms (gathers hidden)"
        );
    }
    sel::sel_gate(&gpu, &kern, &c, &inp, &scratch, "64k")?;
    sel::sel_gate(&gpu, &kern, &c512, &inp, &scratch512, "512k")?;
    println!("RESULT: PASS");
    Ok(())
}
