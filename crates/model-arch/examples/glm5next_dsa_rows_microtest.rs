// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte gate and microbench of the GLM-5.3 DSA mixer's batched decode at the
//! checkpoint's indexer geometry (32 index heads of 128, kpool 4, top-2048 tokens) and the
//! TP=3 per-rank attention (22 heads), over R = 1, 4 and 16 sequences on paged indexer caches
//! of different lengths (up to ~3.9k tokens of an 8k context), on the replay-safe path a
//! captured decode runs: `decode_spans_with` per row (`indexer_rows` false) against the batched
//! indexer (`true`: staged projections, one store and one selection launch per stage).
//! 2026-10-09: Then a prefill sub-chunk (T = 77, 198) of one sequence on the host path, per row
//! against the batched indexer (`decode_k_with`), byte for byte, with its wall time.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless both arms leave byte-identical layer outputs, latent KV pools
//!   and indexer pools; prints the GPU time of one mixer call per arm otherwise (11 copies, the
//!   decode's DSA layers, captured into one graph; `common/graph_timing.rs`).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_rows_microtest \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_paged_rig.rs"]
mod rig;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result};
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
use rig::{BLOCK, Fwd, Lcg, kv, layer, layer_rows, meta_mb, read, same, stream, up};

const ROWS: [usize; 3] = [1, 4, 16];
/// 2026-10-09: The context the indexer cache and selection ceiling are sized for.
const CTX: usize = 8192;
const MBX: usize = CTX / BLOCK;
/// 2026-10-09: Copies per timed graph: the decode's 11 DSA layers.
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

/// 2026-10-09: One arm: its own KV pools (filled with the same history bytes as the other
/// arm's), its paged states at the history lengths, and its input rows.
struct Arm {
    kv: PagedKvCache,
    states: Vec<Box<dyn LayerState>>,
    hidden: DevicePtr,
}

/// 2026-10-09: Pool bytes: FP8 latent codes without the NaN pattern, and BF16 indexer rows.
fn history_bytes(rng: &mut Lcg, k_bytes: usize, v_bytes: usize) -> (Vec<u8>, Vec<u8>) {
    let k = (0..k_bytes)
        .map(|_| (((rng.next() + 1.0) * 127.5) as u8) & 0xF7)
        .collect();
    (k, rng.bf16_bytes(v_bytes / 2, 1.0))
}

fn pools(kv: &PagedKvCache) -> (usize, usize) {
    (
        kv.num_blocks() * kv.k_block_stride_bytes_for_layer(0),
        kv.num_blocks() * kv.v_block_stride_bytes_for_layer(0),
    )
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
    let mut rng = Lcg(0x6c6d_d5a0_0016);
    let l: Glm5NextDsaLayer = layer(g, &c, &mut rng)?;
    println!(
        "GATE+BENCH: batched DSA indexer (store + selection) vs per row, paged, captured path"
    );
    for r in ROWS {
        let lens: Vec<usize> = (0..r).map(|i| 700 + 211 * i).collect();
        let tables: Vec<Vec<u32>> = (0..r)
            .map(|i| (0..MBX).map(|b| (i * MBX + b) as u32).collect())
            .collect();
        let rows: Vec<(usize, &[u32])> = (0..r).map(|i| (lens[i], tables[i].as_slice())).collect();
        let m: AttnMetadataDev = meta_mb(g, &rows, MBX)?;
        let x = rng.bf16_bytes(r * c.hidden, 1.0);
        let probe = kv(g, r * MBX)?;
        let (kb, vb) = pools(&probe);
        let (kh, vh) = history_bytes(&mut rng, kb, vb);
        drop(probe);
        let arm = || -> Result<Arm> {
            let cache = kv(g, r * MBX)?;
            g.copy_h2d(&kh, cache.k_pool_ptr(0))?;
            g.copy_h2d(&vh, cache.v_pool_ptr(0))?;
            let states = lens
                .iter()
                .map(|&len| {
                    let mut st = Glm5NextDsaState::paged(&c)?;
                    st.advance(len)?;
                    Ok(Box::new(st) as Box<dyn LayerState>)
                })
                .collect::<Result<_>>()?;
            Ok(Arm {
                kv: cache,
                states,
                hidden: up(g, &x)?,
            })
        };
        let (mut a, mut b) = (arm()?, arm()?);
        let spans: Vec<DsaRowSpan> = lens
            .iter()
            .map(|&first_pos| DsaRowSpan { first_pos, rows: 1 })
            .collect();
        let ctx = fwd.ctx(g, true, true, Some(m));
        let run = |arm: &mut Arm, batched: bool, s: u64| -> Result<()> {
            let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                arm.states.iter_mut().map(|b| b.as_mut()).collect();
            l.decode_spans_with(
                arm.hidden,
                &mut refs,
                &spans,
                &mut arm.kv,
                &m,
                0,
                &ctx,
                s,
                batched,
            )
        };
        run(&mut a, false, s)?;
        run(&mut b, true, s)?;
        same(
            &format!("R={r} output"),
            &read(g, a.hidden, r * c.hidden * 2)?,
            &read(g, b.hidden, r * c.hidden * 2)?,
        )?;
        same(
            &format!("R={r} latent pool"),
            &read(g, a.kv.k_pool_ptr(0), kb)?,
            &read(g, b.kv.k_pool_ptr(0), kb)?,
        )?;
        same(
            &format!("R={r} indexer pool"),
            &read(g, a.kv.v_pool_ptr(0), vb)?,
            &read(g, b.kv.v_pool_ptr(0), vb)?,
        )?;
        // 2026-10-09: Each call rewinds the states one row (`check_lockstep`) and rewrites the
        // same rows, so repeated calls time the same step.
        let old = timing::time_graph(g, s, COPIES, &mut |s| run(&mut a, false, s))?;
        let new = timing::time_graph(g, s, COPIES, &mut |s| run(&mut b, true, s))?;
        timing::report("DSA mixer (decode_spans)", r, old, new);
    }
    prefill(g, &fwd, &c, &mut rng)?;
    println!("PASS: the batched DSA indexer is byte-identical to per row");
    Ok(())
}

/// 2026-10-09: A prefill sub-chunk of T tokens of one sequence on a paged cache after a
/// 700-token history, on the host path the serve's prefill runs: `decode_k_with` per row
/// against the batched indexer, byte for byte (output, latent pool, indexer pool), and the
/// wall time of one call each (eager, as the prefill runs; median of 7 after 2 warm-ups).
fn prefill(g: &dyn GpuBackend, fwd: &Fwd, c: &Glm5NextDsaConfig, rng: &mut Lcg) -> Result<()> {
    let l = layer_rows(g, c, rng, 512)?;
    let s = stream(g);
    let len = 700;
    for t in [77usize, 198] {
        let table: Vec<u32> = (0..MBX as u32).collect();
        let x = rng.bf16_bytes(t * c.hidden, 1.0);
        let probe = kv(g, MBX)?;
        let (kb, vb) = pools(&probe);
        let (kh, vh) = history_bytes(rng, kb, vb);
        drop(probe);
        let arm = || -> Result<Arm> {
            let cache = kv(g, MBX)?;
            g.copy_h2d(&kh, cache.k_pool_ptr(0))?;
            g.copy_h2d(&vh, cache.v_pool_ptr(0))?;
            let mut st = Glm5NextDsaState::paged(c)?;
            st.advance(len)?;
            Ok(Arm {
                kv: cache,
                states: vec![Box::new(st)],
                hidden: up(g, &x)?,
            })
        };
        let (mut a, mut b) = (arm()?, arm()?);
        let ctx = fwd.ctx(g, false, false, None);
        let run = |arm: &mut Arm, batched: bool| -> Result<()> {
            let mut bt = table.clone();
            g.copy_h2d(&x, arm.hidden)?;
            l.decode_k_with(
                arm.hidden,
                t,
                arm.states[0].as_mut(),
                &mut arm.kv,
                len,
                &mut bt,
                &ctx,
                s,
                true,
                batched,
            )
        };
        run(&mut a, false)?;
        run(&mut b, true)?;
        same(
            &format!("prefill T={t} output"),
            &read(g, a.hidden, t * c.hidden * 2)?,
            &read(g, b.hidden, t * c.hidden * 2)?,
        )?;
        same(
            &format!("prefill T={t} latent pool"),
            &read(g, a.kv.k_pool_ptr(0), kb)?,
            &read(g, b.kv.k_pool_ptr(0), kb)?,
        )?;
        same(
            &format!("prefill T={t} indexer pool"),
            &read(g, a.kv.v_pool_ptr(0), vb)?,
            &read(g, b.kv.v_pool_ptr(0), vb)?,
        )?;
        let wall = |arm: &mut Arm, batched: bool| -> Result<(f64, f64, f64)> {
            let mut v = Vec::new();
            for i in 0..9 {
                g.synchronize(s)?;
                let t0 = std::time::Instant::now();
                run(arm, batched)?;
                g.synchronize(s)?;
                if i >= 2 {
                    v.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            }
            v.sort_by(f64::total_cmp);
            Ok((v[v.len() / 2], v[0], v[v.len() - 1]))
        };
        let old = wall(&mut a, false)?;
        let new = wall(&mut b, true)?;
        timing::report("DSA prefill chunk (decode_k)", t, old, new);
    }
    Ok(())
}
