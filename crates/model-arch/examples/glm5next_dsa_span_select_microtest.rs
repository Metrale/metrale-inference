// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of `METRALE_GLM_DSA_SPAN_SELECT` at the checkpoint's
//! indexer geometry (32 index heads of 128, kpool 4, top-2048 tokens) and the TP=3 per-rank
//! attention (22 heads), on paged indexer caches whose lengths (2.3k..5k tokens) pass the
//! top-2048 budget, so each row's selection drops keys.
//!
//! * `decode_spans_with`: speculative verify spans `(sequences, rows)` = (1, 2), (4, 2), (8, 2)
//!   and (4, 4) (2, 8, 16 and 16 rows) on the replay-safe path a captured decode runs. The span
//!   arm (`indexer_rows`, `span_select`: one store and one selection launch per stage for every
//!   row) against the per-row path the serve runs today (`indexer_rows` on and off).
//! * `decode_k_with`: one sequence's verify of k = 2 and 4 rows with row metadata, `span_select`
//!   on against off. Run once with and once without `METRALE_GLM_DSA_VERIFY_ROWS=1`, the other
//!   arm's two serve paths.
//! * Mutation control: the span arm again with one row's `seq_len` one lower; its compare
//!   against the per-row arm must FAIL, or the gate exits with an error.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every arm leaves byte-identical layer outputs, latent KV pools
//!   and indexer pools, or if the mutation control passes; prints the GPU time of one mixer call
//!   per arm otherwise (11 copies, the decode's DSA layers, captured into one graph).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_span_select_microtest \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_paged_rig.rs"]
mod rig;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::layer::{DsaRowSpan, Glm5NextDsaLayer};
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::{AttnMetadataDev, LayerState};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use rig::{BLOCK, Fwd, Lcg, kv, layer, meta_mb, read, same, stream, up, up_i32};

/// 2026-10-14: Verify spans: (sequences, rows per sequence).
const SPANS: [(usize, usize); 4] = [(1, 2), (4, 2), (8, 2), (4, 4)];
const VERIFY_K: [usize; 2] = [2, 4];
const CTX: usize = 8192;
const MBX: usize = CTX / BLOCK;
const COPIES: usize = 11;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 22,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: CTX,
    }
}

/// 2026-10-14: One arm: its own KV pools (the same history bytes as every other arm's), its
/// paged states at the history lengths, and its input rows.
struct Arm {
    kv: PagedKvCache,
    states: Vec<Box<dyn LayerState>>,
    hidden: DevicePtr,
}

/// 2026-10-14: The bytes one case starts from, shared by its arms.
struct Case {
    c: Glm5NextDsaConfig,
    lens: Vec<usize>,
    x: Vec<u8>,
    kh: Vec<u8>,
    vh: Vec<u8>,
    blocks: usize,
}

impl Case {
    fn new(
        g: &dyn GpuBackend,
        c: &Glm5NextDsaConfig,
        rng: &mut Lcg,
        seqs: usize,
        rows: usize,
    ) -> Result<Self> {
        let lens: Vec<usize> = (0..seqs).map(|i| 2300 + 389 * i).collect();
        let probe = kv(g, seqs * MBX)?;
        let (kb, vb) = pools(&probe);
        drop(probe);
        let kh = (0..kb)
            .map(|_| (((rng.next() + 1.0) * 127.5) as u8) & 0xF7)
            .collect();
        Ok(Self {
            c: *c,
            x: rng.bf16_bytes(seqs * rows * c.hidden, 1.0),
            vh: rng.bf16_bytes(vb / 2, 1.0),
            kh,
            lens,
            blocks: seqs * MBX,
        })
    }

    fn tables(&self) -> Vec<Vec<u32>> {
        (0..self.lens.len())
            .map(|i| (0..MBX).map(|b| (i * MBX + b) as u32).collect())
            .collect()
    }

    /// 2026-10-14: Metadata rows sequence-major: row j of sequence s at position `lens[s] + j`.
    fn meta(&self, g: &dyn GpuBackend, rows: usize) -> Result<AttnMetadataDev> {
        let tables = self.tables();
        let r: Vec<(usize, &[u32])> = (0..self.lens.len())
            .flat_map(|s| (0..rows).map(move |j| (s, j)))
            .map(|(s, j)| (self.lens[s] + j, tables[s].as_slice()))
            .collect();
        meta_mb(g, &r, MBX)
    }

    fn arm(&self, g: &dyn GpuBackend) -> Result<Arm> {
        let cache = kv(g, self.blocks)?;
        g.copy_h2d(&self.kh, cache.k_pool_ptr(0))?;
        g.copy_h2d(&self.vh, cache.v_pool_ptr(0))?;
        let states = self
            .lens
            .iter()
            .map(|&len| {
                let mut st = Glm5NextDsaState::paged(&self.c)?;
                st.advance(len)?;
                Ok(Box::new(st) as Box<dyn LayerState>)
            })
            .collect::<Result<_>>()?;
        Ok(Arm {
            kv: cache,
            states,
            hidden: up(g, &self.x)?,
        })
    }

    /// 2026-10-14: `a` against `b`: layer output, latent pool and indexer pool.
    fn compare(&self, g: &dyn GpuBackend, what: &str, a: &Arm, b: &Arm) -> Result<()> {
        let (kb, vb) = pools(&a.kv);
        same(
            &format!("{what} output"),
            &read(g, a.hidden, self.x.len())?,
            &read(g, b.hidden, self.x.len())?,
        )?;
        same(
            &format!("{what} latent pool"),
            &read(g, a.kv.k_pool_ptr(0), kb)?,
            &read(g, b.kv.k_pool_ptr(0), kb)?,
        )?;
        same(
            &format!("{what} indexer pool"),
            &read(g, a.kv.v_pool_ptr(0), vb)?,
            &read(g, b.kv.v_pool_ptr(0), vb)?,
        )
    }
}

fn pools(kv: &PagedKvCache) -> (usize, usize) {
    (
        kv.num_blocks() * kv.k_block_stride_bytes_for_layer(0),
        kv.num_blocks() * kv.v_block_stride_bytes_for_layer(0),
    )
}

/// 2026-10-14: `meta` with row `row`'s `seq_len` one lower.
fn mutated(g: &dyn GpuBackend, case: &Case, rows: usize, row: usize) -> Result<AttnMetadataDev> {
    let m = case.meta(g, rows)?;
    let mut lens: Vec<i32> = (0..case.lens.len())
        .flat_map(|s| (0..rows).map(move |j| (case.lens[s] + j + 1) as i32))
        .collect();
    lens[row] -= 1;
    Ok(AttnMetadataDev {
        seq_len: up_i32(g, &lens)?,
        ..m
    })
}

/// 2026-10-14: The control must make the compare fail; a pass means the gate cannot see a
/// one-row change.
fn expect_differs(r: Result<()>, what: &str) -> Result<()> {
    match r {
        Ok(()) => bail!("{what}: mutation control PASSED the compare; the gate is blind"),
        Err(e) => {
            println!("{what}: mutation control detected ({e})");
            Ok(())
        }
    }
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = stream(g);
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let fwd = Fwd {
        buffers: BufferArena::new(&config, 8, 16, 16, 8, g)?,
        config,
        dispatch: GemmDispatch::defaults(),
        derived: DerivedWeights::new(),
        levers: ModelLevers::defaults(),
        stats: ModelStats::new(),
    };
    let c = cfg();
    let mut rng = Lcg(0x6c6d_d5a0_0514);
    let l: Glm5NextDsaLayer = layer(g, &c, &mut rng)?;
    println!(
        "GATE+BENCH: DSA span select (METRALE_GLM_DSA_SPAN_SELECT) vs per row; \
         METRALE_GLM_DSA_VERIFY_ROWS={}",
        std::env::var("METRALE_GLM_DSA_VERIFY_ROWS").unwrap_or_default()
    );
    for (seqs, rows) in SPANS {
        spans(g, &fwd, &l, &mut rng, seqs, rows, s)?;
    }
    for k in VERIFY_K {
        verify(g, &fwd, &l, &mut rng, k, s)?;
    }
    println!("PASS: span select is byte-identical to per row; both mutation controls detected");
    Ok(())
}

/// 2026-10-14: `decode_spans_with` over `seqs` spans of `rows` rows: per row with the indexer
/// unbatched (a) and batched (b), against the span arm (c); then the mutation control.
fn spans(
    g: &dyn GpuBackend,
    fwd: &Fwd,
    l: &Glm5NextDsaLayer,
    rng: &mut Lcg,
    seqs: usize,
    rows: usize,
    s: u64,
) -> Result<()> {
    let case = Case::new(g, &l.cfg, rng, seqs, rows)?;
    let m = case.meta(g, rows)?;
    let span: Vec<DsaRowSpan> = case
        .lens
        .iter()
        .map(|&first_pos| DsaRowSpan { first_pos, rows })
        .collect();
    let run = |arm: &mut Arm, m: &AttnMetadataDev, idx: bool, sel: bool, s: u64| -> Result<()> {
        let ctx = fwd.ctx(g, true, true, Some(*m));
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            arm.states.iter_mut().map(|b| b.as_mut()).collect();
        l.decode_spans_with(
            arm.hidden,
            &mut refs,
            &span,
            &mut arm.kv,
            m,
            0,
            &ctx,
            s,
            idx,
            sel,
        )
    };
    let (mut a, mut b, mut c) = (case.arm(g)?, case.arm(g)?, case.arm(g)?);
    run(&mut a, &m, false, false, s)?;
    run(&mut b, &m, true, false, s)?;
    run(&mut c, &m, true, true, s)?;
    let n = seqs * rows;
    case.compare(g, &format!("spans {seqs}x{rows} per-row vs span"), &a, &c)?;
    case.compare(
        g,
        &format!("spans {seqs}x{rows} indexer-rows vs span"),
        &b,
        &c,
    )?;
    let mm = mutated(g, &case, rows, n / 2)?;
    let mut d = case.arm(g)?;
    run(&mut d, &mm, true, true, s)?;
    expect_differs(
        case.compare(g, &format!("spans {seqs}x{rows} mutated"), &b, &d),
        &format!("spans {seqs}x{rows}"),
    )?;
    // 2026-10-14: Each call rewinds the states to the span starts (`check_lockstep`), so
    // repeated calls time the same step.
    let old = timing::time_graph(g, s, COPIES, &mut |s| run(&mut b, &m, true, false, s))?;
    let new = timing::time_graph(g, s, COPIES, &mut |s| run(&mut c, &m, true, true, s))?;
    timing::report(&format!("DSA spans {seqs}x{rows}"), n, old, new);
    Ok(())
}

/// 2026-10-14: `decode_k_with` over one sequence's `k`-row verify with row metadata, span
/// select off (a) against on (b); then the mutation control.
fn verify(
    g: &dyn GpuBackend,
    fwd: &Fwd,
    l: &Glm5NextDsaLayer,
    rng: &mut Lcg,
    k: usize,
    s: u64,
) -> Result<()> {
    let case = Case::new(g, &l.cfg, rng, 1, k)?;
    let m = case.meta(g, k)?;
    let len = case.lens[0];
    let table = case.tables().remove(0);
    let run = |arm: &mut Arm, m: AttnMetadataDev, sel: bool, s: u64| -> Result<()> {
        let ctx = fwd.ctx(g, true, true, Some(m));
        let mut bt = table.clone();
        l.decode_k_with(
            arm.hidden,
            k,
            arm.states[0].as_mut(),
            &mut arm.kv,
            len,
            &mut bt,
            &ctx,
            s,
            false,
            true,
            sel,
        )
    };
    let (mut a, mut b) = (case.arm(g)?, case.arm(g)?);
    run(&mut a, m, false, s)?;
    run(&mut b, m, true, s)?;
    case.compare(g, &format!("verify k={k} per-row vs span"), &a, &b)?;
    let mut d = case.arm(g)?;
    run(&mut d, mutated(g, &case, k, k - 1)?, true, s)?;
    expect_differs(
        case.compare(g, &format!("verify k={k} mutated"), &a, &d),
        &format!("verify k={k}"),
    )?;
    let old = timing::time_graph(g, s, COPIES, &mut |s| run(&mut a, m, false, s))?;
    let new = timing::time_graph(g, s, COPIES, &mut |s| run(&mut b, m, true, s))?;
    timing::report(&format!("DSA verify k={k}"), k, old, new);
    Ok(())
}
