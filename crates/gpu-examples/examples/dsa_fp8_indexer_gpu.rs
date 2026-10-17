// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: GPU gate of the FP8 indexer cache (`--indexer-cache-dtype fp8`): one DSA layer on
//! synthetic weights through the real kernels, a BF16 paged arm against an FP8 paged arm on the
//! same token stream, then the kernels alone at long context.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error unless
//! - A (decode, eager and on the replay-safe path): after the prefill sub-chunk and after every
//!   decode step, the KV latent is byte-identical across the arms, every complete pool's FP8
//!   entry and scale is the host twin's quantization of the BF16 compression of the BF16 arm's
//!   rows, and every open row in the ring is the BF16 row. With a selection wide enough to
//!   keep every pool the layer outputs agree within [`FULL_REL_L2`]; at the narrow width they
//!   are reported.
//! - C: the batched decode (`decode_rows`) and verify (`decode_spans`, after a one-row rewind)
//!   keep the same row invariants on every sequence.
//! - D: an FP8 prefix hit on whole pools gives outputs byte-identical to the cold FP8 run, and
//!   a hit that ends inside a pool is refused.
//! - E (`common/dsa_fp8_kernels.rs`): see `long_context_gate`.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example dsa_fp8_indexer_gpu \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::fp8_rows::{Fp8IndexerLayout, IndexerRowFormat};
use metrale_model_arch::glm5next_dsa::layer::{DsaRowSpan, Glm5NextDsaLayer};
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

#[path = "common/glm5next_dsa_paged_rig.rs"]
pub(crate) mod rig;
use rig::*;
#[path = "common/dsa_fp8_kernels.rs"]
mod fp8k;

/// 2026-10-16: Ring sets of the FP8 arms: a one-row rewind (decode, the verify gate).
const SETS: usize = 1;
/// 2026-10-16: Provisional: the layer-output agreement when the selection keeps every pool
/// (the arms then differ only in the order the selected pools are attended).
const FULL_REL_L2: f64 = 1e-2;

/// 2026-10-16: One sequence's state, KV cache and block table.
struct Arm {
    st: Box<dyn LayerState>,
    kv: PagedKvCache,
    bt: Vec<u32>,
}

impl Arm {
    /// 2026-10-16: Blocks scrambled inside a pool three times larger (as the paged parity gate).
    fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig, fp8: bool, salt: usize) -> Result<Self> {
        let (format, v) = if fp8 {
            let v =
                Fp8IndexerLayout::block_bytes(BLOCK, cfg.index_head_dim, cfg.index_kpool, SETS)?;
            (IndexerRowFormat::Fp8 { tail_sets: SETS }, Some(v))
        } else {
            (IndexerRowFormat::Bf16, None)
        };
        Ok(Self {
            st: Box::new(Glm5NextDsaState::paged_as(cfg, format)?),
            kv: kv_with_v(gpu, 3 * MB, v)?,
            bt: (0..MB)
                .map(|b| ((b * 7 + salt) % (3 * MB)) as u32)
                .collect(),
        })
    }
    fn dsa(&self) -> Result<&Glm5NextDsaState> {
        dsa(self.st.as_ref())
    }
    /// 2026-10-16: `k` rows at `pos` of this arm's sequence; returns the layer output.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        l: &Glm5NextDsaLayer,
        fwd: &Fwd,
        gpu: &dyn GpuBackend,
        x: &[u8],
        pos: usize,
        k: usize,
        capture: bool,
    ) -> Result<Vec<u8>> {
        let (st, kv, bt) = (self.st.as_mut(), &mut self.kv, &mut self.bt);
        decode(l, fwd, gpu, st, kv, bt, x, pos, k, capture)
    }
}

/// 2026-10-16: `k` rows at `pos` through `decode_k`; returns the layer output.
#[allow(clippy::too_many_arguments)]
fn decode(
    l: &Glm5NextDsaLayer,
    fwd: &Fwd,
    gpu: &dyn GpuBackend,
    st: &mut dyn LayerState,
    cache: &mut PagedKvCache,
    bt: &mut Vec<u32>,
    x: &[u8],
    pos: usize,
    k: usize,
    capture: bool,
) -> Result<Vec<u8>> {
    let h = up(gpu, x)?;
    let m = (capture && k == 1)
        .then(|| meta(gpu, &[(pos, bt.as_slice())]))
        .transpose()?;
    let ctx = fwd.ctx(gpu, capture && k == 1, k == 1, m);
    l.decode_k(h, k, st, cache, pos, bt, &ctx, stream(gpu), k > 1)?;
    read(gpu, h, k * HIDDEN * 2)
}

/// 2026-10-16: One sequence of one arm as the row checks read it.
#[derive(Clone, Copy)]
struct Side<'a> {
    st: &'a Glm5NextDsaState,
    kv: &'a PagedKvCache,
    bt: &'a [u32],
}

impl Arm {
    fn side(&self) -> Result<Side<'_>> {
        Ok(Side {
            st: self.dsa()?,
            kv: &self.kv,
            bt: &self.bt,
        })
    }
}

fn dsa(st: &dyn LayerState) -> Result<&Glm5NextDsaState> {
    st.as_any()
        .downcast_ref::<Glm5NextDsaState>()
        .context("DSA state")
}

/// 2026-10-16: The row invariants of A and C on one sequence of each arm (see the module docs).
fn check_rows(
    what: &str,
    gpu: &dyn GpuBackend,
    l: &Glm5NextDsaLayer,
    bf: Side,
    f8: Side,
) -> Result<usize> {
    let len = f8.st.len();
    ensure!(
        bf.st.len() == len,
        "{what}: the arms hold {} and {len} rows",
        bf.st.len()
    );
    let bt_b = up_i32(gpu, &bf.bt.iter().map(|&b| b as i32).collect::<Vec<_>>())?;
    let keys = fp8k::compress_bf16(
        gpu,
        &l.select_kernels,
        &l.indexer_rows(bf.st, bf.kv, bt_b)?,
        l.weights.ape,
        len,
    )?;
    gpu.free(bt_b)?;
    let lay = l.fp8_layout(f8.st, f8.kv)?;
    let pool = read(
        gpu,
        f8.kv.v_pool_ptr(0),
        f8.kv.num_blocks() * lay.block_stride_bytes,
    )?;
    let blay = l.paged_layout(bf.kv)?;
    let bpool = read(
        gpu,
        bf.kv.v_pool_ptr(0),
        bf.kv.num_blocks() * blay.block_stride_bytes,
    )?;
    let d2 = l.cfg.index_head_dim * 2;
    let row = |x: usize| -> Result<(Vec<u8>, Vec<u8>)> {
        let o = blay.key_row_bytes(bf.bt, x)?;
        let g = o + blay.gate_offset_bytes();
        Ok((bpool[o..o + d2].to_vec(), bpool[g..g + d2].to_vec()))
    };
    fp8k::check_pool(what, &lay, &pool, f8.bt, len, &keys, &row)
}

fn latent(a: &Arm, gpu: &dyn GpuBackend, pos: usize) -> Result<Vec<u8>> {
    let slot = a.bt[pos / BLOCK] as usize * BLOCK + pos % BLOCK;
    read(gpu, a.kv.k_pool_ptr(0).offset(slot * 512), 512)
}

/// 2026-10-16: Relative L2 distance of two BF16 vectors.
fn rel_l2(a: &[u8], b: &[u8]) -> f64 {
    let f = |c: &[u8]| bf16::from_le_bytes([c[0], c[1]]).to_f64();
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        num += (f(x) - f(y)).powi(2);
        den += f(x).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// 2026-10-16: Output agreement over a gate: the worst relative L2 and the byte-identical count.
#[derive(Default)]
struct Agree {
    worst: f64,
    same: usize,
    total: usize,
}
impl Agree {
    fn add(&mut self, a: &[u8], b: &[u8]) {
        self.worst = self.worst.max(rel_l2(a, b));
        self.same += usize::from(a == b);
        self.total += 1;
    }
    fn finish(&self, what: &str, full: bool) -> Result<()> {
        println!(
            "  {what}: outputs byte-identical {}/{}, worst rel-L2 {:.6}",
            self.same, self.total, self.worst
        );
        ensure!(
            !full || self.worst <= FULL_REL_L2,
            "{what}: with every pool selected the outputs differ by rel-L2 {} (bound {FULL_REL_L2})",
            self.worst
        );
        Ok(())
    }
}

/// 2026-10-16: Gate A: a prefill sub-chunk then decode, BF16 against FP8.
fn decode_gate(
    gpu: &dyn GpuBackend,
    fwd: &Fwd,
    l: &Glm5NextDsaLayer,
    rng: &mut Lcg,
    capture: bool,
    full: bool,
) -> Result<()> {
    let (mut bf, mut f8) = (
        Arm::new(gpu, &l.cfg, false, 3)?,
        Arm::new(gpu, &l.cfg, true, 5)?,
    );
    let chunk = 14;
    let x = rng.bf16_bytes(chunk * HIDDEN, 1.0);
    let mut agree = Agree::default();
    let a = bf.run(l, fwd, gpu, &x, 0, chunk, false)?;
    let b = f8.run(l, fwd, gpu, &x, 0, chunk, false)?;
    agree.add(&a, &b);
    let tag = format!("A capture={capture} full={full}");
    check_rows(&format!("{tag} prefill"), gpu, l, bf.side()?, f8.side()?)?;
    let mut pools = 0;
    for pos in chunk..LEN {
        let x = rng.bf16_bytes(HIDDEN, 1.0);
        let a = bf.run(l, fwd, gpu, &x, pos, 1, capture)?;
        let b = f8.run(l, fwd, gpu, &x, pos, 1, capture)?;
        agree.add(&a, &b);
        same(
            &format!("{tag} pos {pos} KV latent"),
            &latent(&bf, gpu, pos)?,
            &latent(&f8, gpu, pos)?,
        )?;
        pools = check_rows(&format!("{tag} pos {pos}"), gpu, l, bf.side()?, f8.side()?)?;
    }
    println!(
        "  {tag}: {} steps, {pools} pools: latents identical, entries == twin, ring == BF16",
        LEN - chunk
    );
    agree.finish(&tag, full)
}

/// 2026-10-16: Gate C: C sequences of different lengths in one `decode_rows` per arm, each arm's
/// sequences in disjoint scrambled blocks of one shared cache; then a one-row rewind and three
/// verify rows per sequence in one `decode_spans`.
fn rows_gate(
    gpu: &dyn GpuBackend,
    fwd: &Fwd,
    l: &Glm5NextDsaLayer,
    rng: &mut Lcg,
    capture: bool,
    full: bool,
) -> Result<()> {
    let c = 4;
    let lens: Vec<usize> = (0..c).map(|r| 17 + r * 31).collect();
    let n = 3 * c * MB;
    let v = Fp8IndexerLayout::block_bytes(BLOCK, l.cfg.index_head_dim, l.cfg.index_kpool, SETS)?;
    let mut kvs = [kv(gpu, n)?, kv_with_v(gpu, n, Some(v))?];
    // 2026-10-16: `i * 7 + salt mod n` is a bijection (7 and 120 are coprime): disjoint tables.
    let tables: Vec<Vec<Vec<u32>>> = [5usize, 11]
        .iter()
        .map(|salt| {
            (0..c)
                .map(|r| {
                    (0..MB)
                        .map(|b| (((r * MB + b) * 7 + salt) % n) as u32)
                        .collect()
                })
                .collect()
        })
        .collect();
    let format = [
        IndexerRowFormat::Bf16,
        IndexerRowFormat::Fp8 { tail_sets: SETS },
    ];
    let mut states: Vec<Vec<Box<dyn LayerState>>> = vec![Vec::new(), Vec::new()];
    for (r, &len) in lens.iter().enumerate() {
        let xs: Vec<Vec<u8>> = (0..len).map(|_| rng.bf16_bytes(HIDDEN, 1.0)).collect();
        for side in 0..2 {
            let mut st: Box<dyn LayerState> =
                Box::new(Glm5NextDsaState::paged_as(&l.cfg, format[side])?);
            let mut bt = tables[side][r].clone();
            for (pos, x) in xs.iter().enumerate() {
                decode(
                    l,
                    fwd,
                    gpu,
                    st.as_mut(),
                    &mut kvs[side],
                    &mut bt,
                    x,
                    pos,
                    1,
                    false,
                )?;
            }
            states[side].push(st);
        }
    }
    let tag = format!("C capture={capture} full={full}");
    let mut agree = Agree::default();
    for k in [1usize, 3] {
        let x = rng.bf16_bytes(k * c * HIDDEN, 1.0);
        let mut outs = Vec::new();
        for side in 0..2 {
            let h = up(gpu, &x)?;
            let rows: Vec<(usize, &[u32])> = (0..c)
                .flat_map(|r| (0..k).map(move |t| (r, t)))
                .map(|(r, t)| (lens[r] + t, tables[side][r].as_slice()))
                .collect();
            let m = meta(gpu, &rows)?;
            let ctx = fwd.ctx(gpu, capture, true, Some(m));
            let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                states[side].iter_mut().map(|b| b.as_mut()).collect();
            if k == 1 {
                l.decode_rows(
                    h,
                    &mut refs,
                    &lens,
                    &mut kvs[side],
                    &m,
                    0,
                    &ctx,
                    stream(gpu),
                )?;
            } else {
                // 2026-10-16: Every state is one row ahead: the verify rewinds it one row.
                let spans: Vec<DsaRowSpan> = lens
                    .iter()
                    .map(|&first_pos| DsaRowSpan { first_pos, rows: k })
                    .collect();
                l.decode_spans(
                    h,
                    &mut refs,
                    &spans,
                    &mut kvs[side],
                    &m,
                    0,
                    &ctx,
                    stream(gpu),
                )?;
            }
            outs.push(read(gpu, h, k * c * HIDDEN * 2)?);
        }
        for (a, b) in outs[0].chunks(HIDDEN * 2).zip(outs[1].chunks(HIDDEN * 2)) {
            agree.add(a, b);
        }
        for r in 0..c {
            let side = |s: usize| -> Result<Side> {
                Ok(Side {
                    st: dsa(states[s][r].as_ref())?,
                    kv: &kvs[s],
                    bt: &tables[s][r],
                })
            };
            check_rows(&format!("{tag} k={k} seq {r}"), gpu, l, side(0)?, side(1)?)?;
        }
    }
    println!("  {tag}: decode_rows, then rewind + decode_spans x3: entries == twin, ring == BF16");
    agree.finish(&tag, full)
}

/// 2026-10-16: Gate D: an FP8 prefix hit on whole pools against the cold FP8 run, and a hit
/// that ends inside a pool, refused.
fn prefix_gate(gpu: &dyn GpuBackend, fwd: &Fwd, l: &Glm5NextDsaLayer, rng: &mut Lcg) -> Result<()> {
    let mut a = Arm::new(gpu, &l.cfg, true, 0)?;
    let shared = 5;
    let start = shared * BLOCK;
    let xs: Vec<Vec<u8>> = (0..LEN).map(|_| rng.bf16_bytes(HIDDEN, 1.0)).collect();
    let mut a_out = Vec::new();
    for (pos, x) in xs.iter().enumerate() {
        a_out.push(a.run(l, fwd, gpu, x, pos, 1, false)?);
    }
    let used: std::collections::BTreeSet<u32> = a.bt.iter().copied().collect();
    let mut free = (0..3 * MB as u32).filter(|b| !used.contains(b));
    let mut bt: Vec<u32> = a.bt[..shared].to_vec();
    bt.extend((shared..MB).map(|_| free.next().expect("a free block")));
    let fresh = || Glm5NextDsaState::paged_as(&l.cfg, IndexerRowFormat::Fp8 { tail_sets: SETS });
    let mut b = Arm {
        st: Box::new(fresh()?),
        kv: std::mem::replace(&mut a.kv, kv(gpu, 1)?),
        bt,
    };
    // 2026-10-16: The refusal first: a fresh state adopting `start + 1` rows (inside a pool).
    let (mut bad, mut bt_bad, at) = (fresh()?, b.bt.clone(), start + 1);
    let refused = decode(
        l,
        fwd,
        gpu,
        &mut bad,
        &mut b.kv,
        &mut bt_bad,
        &xs[at],
        at,
        1,
        false,
    );
    match refused {
        Err(e) if format!("{e:#}").contains("ends inside a pool") => {}
        Err(e) => bail!("D: the mid-pool hit failed for another reason: {e:#}"),
        Ok(_) => bail!("D: a prefix hit ending inside a pool was accepted"),
    }
    for pos in start..LEN {
        let out = b.run(l, fwd, gpu, &xs[pos], pos, 1, false)?;
        same(&format!("D prefix hit pos {pos}"), &a_out[pos], &out)?;
    }
    println!(
        "  D: FP8 prefix hit at {start} tokens: {} outputs byte-identical to cold; a hit at {} refused",
        LEN - start,
        start + 1
    );
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let fwd = Fwd {
        buffers: BufferArena::new(&config, 8, 16, 16, 8, &gpu)?,
        config,
        dispatch: GemmDispatch::defaults(),
        derived: DerivedWeights::new(),
        levers: ModelLevers::defaults(),
        stats: ModelStats::new(),
    };
    let mut rng = Lcg(0x6c6d_5e9d_0016);
    println!("GATE: FP8 indexer cache against BF16");
    for full in [false, true] {
        let mut c = cfg();
        if full {
            // 2026-10-16: Every pool of the context fits the selection.
            c.index_topk = c.max_context / c.index_kpool;
        }
        let l = layer(&gpu, &c, &mut rng)?;
        ensure!(
            l.select_kernels.fp8.ready(),
            "this build has no FP8 indexer kernels"
        );
        for capture in [false, true] {
            decode_gate(&gpu, &fwd, &l, &mut rng, capture, full)?;
            rows_gate(&gpu, &fwd, &l, &mut rng, capture, full)?;
        }
        if !full {
            prefix_gate(&gpu, &fwd, &l, &mut rng)?;
        }
    }
    let l = layer(&gpu, &cfg(), &mut rng)?;
    fp8k::long_context_gate(&gpu, &l.select_kernels, &mut rng)?;
    println!("PASS");
    Ok(())
}
