// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The host-path leg: the whole selection launcher with `METRALE_GLM_DSA_SELECT_FUSED`
//! off and on, on the same inputs. The levers are read once per process, so the gate re-runs
//! itself (`--host-child <dir>`) once per setting, with `METRALE_GLM_DSA_SCORES_MMA=1` and
//! `METRALE_GLM_DSA_SCORES_PIPE=1`, `METRALE_GLM_DSA_SELECT_FUSED` 0 / 1 and
//! `METRALE_GLM_DSA_TOPK_RADIX` 0 / 1 (the unfused reference under both top-k kernels: a fused
//! pass ignores the radix lever, so its rows must equal both).
//!
//! Each child builds a flat indexer cache (random, or tie: every 7th pool the same keys and
//! gates, so equal pool keys tie at every score) and, per shape:
//! - A: `select_tokens` (one exact pass) into a scratch sized for every row.
//! - B: `select_tokens_rows` over three row ranges, into a scratch whose scores hold 64 rows
//!   (sub-passes at row offsets; under the fused lever only the ranges split it).
//! - I: `select_tokens_in_passes` on that scratch.
//! - F: `select_rows_folded` over the three ranges (each range's selected pools at its own rows,
//!   `row_sel`), checking it writes no token byte (the fold returns before any expansion),
//!   then `expand_folded` over the whole pass.
//!
//! Every output is pre-filled with its own pattern. The child requires A = B = I = F byte for
//! byte, live tokens in A, and B and F to run at least three sub-passes; it writes A's token
//! rows, F's selected pools and its readback (fused taken, rows per pass, sub-passes). The
//! parent requires fused taken exactly in the fused children and 0 rows differing between every
//! child and the unfused radix child, for both the token rows and the folded selected pools.
//!
//! Owner: model-arch examples.
//! Invariants: the parent errors unless every child exits 0 and every comparison holds; the
//! scratch directory is removed on success.

use anyhow::{Context, bail};
use metrale_model_arch::glm5next_dsa::paged::IndexerRowsDev;
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, expand_folded,
    select_rows_folded, select_tokens, select_tokens_in_passes, select_tokens_rows,
};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

use super::*;

/// 2026-10-17: (label, query rows, tokens, `index_topk`, tie cache). `index_topk / 4` is the
/// pass's `select_k` where the context has that many pools: 512, 1 (`index_topk` 4), 257 (1028)
/// and 300 (1200 over exactly 300 pools); P > 704 shapes overflow one warp buffer; a token count
/// not a multiple of 4 leaves a partial trailing pool for the expansion's tail.
const SHAPES: [(&str, usize, usize, usize, bool); 6] = [
    ("tie_p3001_k512", 300, 12_007, 2048, true),
    ("rand_p5000_k512", 200, 20_001, 2048, false),
    ("rand_p800_k1", 96, 3_202, 4, false),
    ("tie_p2000_k257", 96, 8_003, 1028, true),
    ("rand_p300_k300", 120, 1_200, 1200, false),
    ("rand_p513_k512", 120, 2_052, 2048, false),
];

/// 2026-10-17: (FUSED, TOPK_RADIX) per child; index 1 (unfused, radix) is the reference.
const CHILDREN: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
const REF: usize = 1;
const SCORE_ROWS: usize = 64;

fn cfg(max_context: usize, index_topk: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: H,
        index_head_dim: D,
        index_kpool: KP,
        index_topk,
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

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn inputs(
    gpu: &dyn GpuBackend,
    q: usize,
    s: usize,
    tie: bool,
    rng: &mut Rng,
) -> Result<DsaSelectInputs> {
    let bf = |f: f32| half::bf16::from_f32(f).to_bits();
    let base: Vec<u16> = (0..2 * 7 * KP * D).map(|_| bf(rng.unit())).collect();
    let (k, g): (Vec<u16>, Vec<u16>) = (0..s * D)
        .map(|i| {
            if tie {
                let j = ((i / D / KP) % 7 * KP + i / D % KP) * D + i % D;
                (base[j], base[7 * KP * D + j])
            } else {
                (bf(rng.unit()), bf(rng.unit()))
            }
        })
        .unzip();
    let valid: Vec<u8> = (0..s)
        .map(|_| u8::from(tie || !rng.next().is_multiple_of(100)))
        .collect();
    let qpos: Vec<i32> = (0..q).map(|r| (s - q + r) as i32).collect();
    let qmask: Vec<u8> = (0..q).map(|r| u8::from(r % 29 != 3)).collect();
    let qv: Vec<f32> = (0..q * H * D).map(|_| rng.unit()).collect();
    let w: Vec<f32> = (0..q * H).map(|_| rng.unit() * 0.2).collect();
    let ape: Vec<f32> = (0..KP * D).map(|_| rng.unit() * 0.1).collect();
    Ok(DsaSelectInputs {
        rows: IndexerRowsDev::flat(up(gpu, &k)?, up(gpu, &g)?, up(gpu, &valid)?),
        ape: up(gpu, &ape)?,
        q: up(gpu, &qv)?,
        weights: up(gpu, &w)?,
        q_pos: up(gpu, &qpos)?,
        q_mask: up(gpu, &qmask)?,
        first_key: 0,
        geom_dev: DevicePtr::NULL,
    })
}

/// 2026-10-17: One child: every shape's legs, written under `dir`.
pub(super) fn child(dir: &std::path::Path) -> Result<()> {
    let set = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(0, &set.modules)?;
    let kern = Glm5NextDsaKernels::resolve(&gpu)?;
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    for &(label, q, s, topk, tie) in &SHAPES {
        let c = cfg(s, topk);
        let inp = inputs(&gpu, q, s, tie, &mut rng)?;
        let geom = DsaSelectGeometry::plan(&c, s, q)?;
        let sel = metrale_gpu_examples::glm_levers()?.dsa.select;
        let full = DsaSelectScratch::alloc(&gpu, &c, &geom, sel)?;
        let small = DsaSelectScratch::alloc_rows(
            &gpu,
            &c,
            &DsaSelectGeometry {
                q_rows: SCORE_ROWS,
                ..geom
            },
            q,
            sel,
        )?;
        let nt = q * c.out_width() * 4;
        let ns = q * geom.select_k * 4;
        let ranges = [
            (0, q / 3 + 1),
            (q / 3 + 1, 2 * q / 3 + 3),
            (2 * q / 3 + 3, q),
        ];
        gpu.memset(full.tokens(), 0x11, nt)?;
        select_tokens(
            &gpu,
            &kern,
            &c,
            &geom,
            &inp,
            &full,
            DsaSelectLaunch::Exact,
            0,
        )?;
        let a = read(&gpu, full.tokens(), nt)?;
        gpu.memset(small.tokens(), 0x22, nt)?;
        let pb = select_tokens_rows(&gpu, &kern, &c, &geom, &inp, &small, &ranges, true, 0)?;
        let b = read(&gpu, small.tokens(), nt)?;
        gpu.memset(small.tokens(), 0x33, nt)?;
        let pi = select_tokens_in_passes(&gpu, &kern, &c, &geom, &inp, &small, 0)?;
        let i = read(&gpu, small.tokens(), nt)?;
        gpu.memset(small.selected(), 0x44, ns)?;
        gpu.memset(small.tokens(), 0x55, nt)?;
        let pf = select_rows_folded(&gpu, &kern, &c, &geom, &inp, &small, &ranges, true, 0)?;
        let untouched = read(&gpu, small.tokens(), nt)?.iter().all(|&x| x == 0x55);
        let sel = read(&gpu, small.selected(), ns)?;
        expand_folded(&gpu, &kern, &c, &geom, &inp, &small, 0)?;
        let f = read(&gpu, small.tokens(), nt)?;
        let rb = c.out_width() * 4;
        let differ = |x: &[u8]| {
            a.chunks(rb)
                .zip(x.chunks(rb))
                .filter(|(u, v)| u != v)
                .count()
        };
        let live = a.chunks_exact(4).filter(|w| w != &[0xFF; 4]).count();
        let meta = format!(
            "fused_on={} per={} passes_rows={pb} passes_in={pi} passes_fold={pf} \
             untouched={untouched} live={live} differ_b={} differ_i={} differ_f={}",
            u8::from(small.fused_on(&geom)),
            small.rows_per_pass(&geom),
            differ(&b),
            differ(&i),
            differ(&f)
        );
        println!("  child {label}: k={} {meta}", geom.select_k);
        ensure!(
            differ(&b) + differ(&i) + differ(&f) == 0,
            "{label}: legs differ: {meta}"
        );
        ensure!(untouched, "{label}: the folded pass wrote token rows");
        ensure!(live > 0 && pb >= 3 && pf >= 3, "{label}: {meta}");
        std::fs::write(dir.join(format!("{label}.tokens")), &a)?;
        std::fs::write(dir.join(format!("{label}.sel")), &sel)?;
        std::fs::write(dir.join(format!("{label}.meta")), meta)?;
        full.free(&gpu)?;
        small.free(&gpu)?;
        for p in [
            inp.rows.k,
            inp.rows.gate,
            inp.rows.valid,
            inp.ape,
            inp.q,
            inp.weights,
        ] {
            gpu.free(p)?;
        }
        gpu.free(inp.q_pos)?;
        gpu.free(inp.q_mask)?;
    }
    Ok(())
}

/// 2026-10-17: Rows of `w` bytes where `a` and `b` differ (all rows when the lengths differ).
fn rows_apart(a: &[u8], b: &[u8], w: usize) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len()) / w;
    }
    a.chunks(w).zip(b.chunks(w)).filter(|(x, y)| x != y).count()
}

/// 2026-10-17: The parent: one child per setting, then the cross-setting comparisons.
pub(super) fn leg() -> Result<()> {
    let root = std::env::temp_dir().join(format!("dsa_fused_host_{}", std::process::id()));
    let exe = std::env::current_exe()?;
    let mut dirs = Vec::new();
    for (n, &(fu, rx)) in CHILDREN.iter().enumerate() {
        let dir = root.join(format!("c{n}"));
        std::fs::create_dir_all(&dir)?;
        println!("host leg: child FUSED={fu} TOPK_RADIX={rx}");
        let st = std::process::Command::new(&exe)
            .arg("--host-child")
            .arg(&dir)
            .env("METRALE_GLM_DSA_SCORES_MMA", "1")
            .env("METRALE_GLM_DSA_SCORES_PIPE", "1")
            .env("METRALE_GLM_DSA_SELECT_FUSED", fu.to_string())
            .env("METRALE_GLM_DSA_TOPK_RADIX", rx.to_string())
            .env_remove("METRALE_GLM_DSA_SCORES_TILED")
            .status()?;
        if !st.success() {
            bail!("host leg child FUSED={fu} TOPK_RADIX={rx} failed: {st}");
        }
        dirs.push(dir);
    }
    for &(label, _, s, topk, _) in &SHAPES {
        let load = |d: &std::path::Path, ext: &str| std::fs::read(d.join(format!("{label}.{ext}")));
        let (rt, rs) = (load(&dirs[REF], "tokens")?, load(&dirs[REF], "sel")?);
        let c = cfg(s, topk);
        let (rb, sk) = (
            c.out_width() * 4,
            DsaSelectGeometry::plan(&c, s, 1)?.select_k,
        );
        for (n, &(fu, rx)) in CHILDREN.iter().enumerate() {
            let meta = String::from_utf8(load(&dirs[n], "meta")?)?;
            let dt = rows_apart(&rt, &load(&dirs[n], "tokens")?, rb);
            let ds = rows_apart(&rs, &load(&dirs[n], "sel")?, 4 * sk);
            println!(
                "host {label} FUSED={fu} RADIX={rx}: token rows differ {dt}, folded rows differ \
                 {ds} vs unfused radix; {meta}"
            );
            ensure!(
                meta.starts_with(&format!("fused_on={fu} ")),
                "{label}: the fused lever did not take the expected path ({meta})"
            );
            ensure!(dt == 0 && ds == 0, "{label}: FUSED={fu} RADIX={rx} differs");
        }
    }
    std::fs::remove_dir_all(&root)?;
    Ok(())
}
