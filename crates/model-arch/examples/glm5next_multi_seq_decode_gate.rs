// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Byte-identity gate for the GLM-5.3 batched multi-sequence decode mixers: KDA
//! `decode_rows` and DSA `decode_rows` over C sequences against each sequence decoded alone
//! (`Glm5NextKdaLayer::decode`, `Glm5NextDsaLayer::decode_k` at k = 1 on a decode step), at
//! C = 2, 4 and 16, on synthetic weights through the real kernels.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless, for every C and every row, the batched output, the updated
//!   KDA recurrent and conv state, and the DSA indexer row and KV latent the step wrote are
//!   byte-identical to the single-sequence run's.
//!
//! The sequences differ in length and history, so a row that read another row's state,
//! metadata row or selection would differ. The DSA arm runs twice: on the host-offset path
//! (`graph_capture` false) and on the replay-safe path a captured graph bakes (`graph_capture`
//! true, launched eagerly), each against the single-sequence run on the same path.
//!
//! The mHC, norm and MLP launches of a batched layer are covered by their own byte gates
//! (`glm5next_moe_row_batch_microtest`, `dense_gemv_bf16_batchm_microtest`); the whole model
//! by the serve-level A/B in the campaign report (batched against `METRALE_HC_PERSEQ_DECODE=1`).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_multi_seq_decode_gate \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use metrale_model_arch::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWeights, Glm5NextDsaWorkspace,
};
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use metrale_model_layers::weight_map::DenseWeight;

const CS: [usize; 3] = [2, 4, 16];
const HIDDEN: usize = 2048;
const BLOCK: usize = 16;
/// 2026-10-08: Block-table entries per sequence: 6 blocks of 16 hold the longest history (79
/// tokens) plus the decoded one.
const MB: usize = 6;
// 2026-10-09: Local `STREAM` = `gpu.default_stream()`, the non-blocking stream `copy_h2d` uses;
// stream 0 is not ordered after the layer's host copies (`slot`, `q_pos`, `bt`).

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn up(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
fn up_i32(gpu: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    #[allow(non_snake_case)]
    let STREAM = gpu.default_stream();
    gpu.synchronize(STREAM)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn copy(gpu: &dyn GpuBackend, src: DevicePtr, n: usize) -> Result<DevicePtr> {
    let p = gpu.alloc(n)?;
    gpu.copy_d2d(src, p, n)?;
    Ok(p)
}

fn same(what: &str, a: &[u8], b: &[u8]) -> Result<()> {
    if a != b {
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
        bail!("{what}: batched differs from single-sequence at byte {first}");
    }
    Ok(())
}

/// 2026-10-08: KDA: C states drawn independently; each sequence decoded alone from a copy,
/// then all C in one `decode_rows` from another copy.
fn kda_gate(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    #[allow(non_snake_case)]
    let STREAM = gpu.default_stream();
    let cfg = Glm5NextKdaConfig {
        hidden: HIDDEN,
        heads: 8,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-6,
        l2_eps: 1e-6,
        chunk: 32,
    };
    let (qkv, hd) = (cfg.qkv_dim(), cfg.head_dim);
    let w = |rng: &mut Lcg, n: usize| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: up_bf16(gpu, &rng.vec(n, 0.03))?,
        })
    };
    let weights = Glm5NextKdaWeights {
        q_proj: w(rng, qkv * HIDDEN)?,
        k_proj: w(rng, qkv * HIDDEN)?,
        v_proj: w(rng, qkv * HIDDEN)?,
        conv: w(rng, cfg.conv_dim() * cfg.conv_kernel)?,
        f_a: w(rng, hd * HIDDEN)?,
        f_b: w(rng, qkv * hd)?,
        dt_bias: up_f32(gpu, &rng.vec(qkv, 0.5))?,
        a_log: up_f32(gpu, &rng.vec(cfg.heads, 0.5))?,
        b_proj: w(rng, cfg.heads * HIDDEN)?,
        g_a: w(rng, hd * HIDDEN)?,
        g_b: w(rng, qkv * hd)?,
        o_norm: DenseWeight {
            weight: up_bf16(gpu, &vec![1.0; hd])?,
        },
        o_proj: w(rng, HIDDEN * qkv)?,
    };
    let layer = Glm5NextKdaLayer::new(0, cfg, weights, Glm5NextKdaKernels::resolve(gpu)?)?;
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, 16)?;
    let (cb, hb) = (cfg.conv_state_elems() * 4, cfg.recurrent_state_elems() * 4);
    for c in CS {
        let init: Vec<KdaSeqState> = (0..c)
            .map(|_| {
                Ok(KdaSeqState {
                    conv: up_f32(gpu, &rng.vec(cb / 4, 0.5))?,
                    recurrent: up_f32(gpu, &rng.vec(hb / 4, 0.1))?,
                })
            })
            .collect::<Result<_>>()?;
        let hidden = up_bf16(gpu, &rng.vec(c * HIDDEN, 1.0))?;
        let fork = |s: &KdaSeqState| -> Result<KdaSeqState> {
            Ok(KdaSeqState {
                conv: copy(gpu, s.conv, cb)?,
                recurrent: copy(gpu, s.recurrent, hb)?,
            })
        };
        let alone: Vec<KdaSeqState> = init.iter().map(fork).collect::<Result<_>>()?;
        let batch: Vec<KdaSeqState> = init.iter().map(fork).collect::<Result<_>>()?;
        let mut outs = Vec::with_capacity(c);
        for (r, st) in alone.iter().enumerate() {
            layer.decode(gpu, hidden.offset(r * HIDDEN * 2), st, &ws, STREAM)?;
            outs.push(read(gpu, ws.final_out, HIDDEN * 2)?);
        }
        layer.decode_rows(gpu, hidden, &batch, &ws, STREAM)?;
        let rows = read(gpu, ws.final_out, c * HIDDEN * 2)?;
        for r in 0..c {
            let row = &rows[r * HIDDEN * 2..(r + 1) * HIDDEN * 2];
            same(&format!("KDA C={c} row {r} output"), &outs[r], row)?;
            same(
                &format!("KDA C={c} row {r} recurrent state"),
                &read(gpu, alone[r].recurrent, hb)?,
                &read(gpu, batch[r].recurrent, hb)?,
            )?;
            same(
                &format!("KDA C={c} row {r} conv state"),
                &read(gpu, alone[r].conv, cb)?,
                &read(gpu, batch[r].conv, cb)?,
            )?;
        }
        println!("  KDA  C={c:>2}: {c} rows byte-identical (output, recurrent, conv)");
    }
    Ok(())
}

fn dsa_layer(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    rng: &mut Lcg,
) -> Result<Glm5NextDsaLayer> {
    let (h, ql, kvl, d) = (
        cfg.hidden,
        cfg.q_lora_rank,
        cfg.kv_lora_rank,
        cfg.index_head_dim,
    );
    let lh = cfg.local_heads;
    let ape = up_f32(gpu, &rng.vec(cfg.index_kpool * d, 0.1))?;
    let mut bf = |n: usize, s: f32| up_bf16(gpu, &rng.vec(n, s));
    let weights = Glm5NextDsaWeights {
        q_a_proj: bf(ql * h, 0.03)?,
        q_a_layernorm: up_bf16(gpu, &vec![1.0; ql])?,
        q_absorb: bf(lh * kvl * ql, 0.03)?,
        kv_a_proj: bf(kvl * h, 0.03)?,
        kv_a_layernorm: up_bf16(gpu, &vec![1.0; kvl])?,
        o_absorb: bf(h * lh * kvl, 0.03)?,
        wk: bf(d * h, 0.03)?,
        k_norm_weight: up_bf16(gpu, &vec![1.0; d])?,
        k_norm_bias: up_bf16(gpu, &vec![0.0; d])?,
        compress_gate: bf(d * h, 0.03)?,
        wq_b: bf(cfg.index_heads * d * ql, 0.03)?,
        weights_proj: bf(cfg.index_heads * h, 0.03)?,
        ape,
    };
    Ok(Glm5NextDsaLayer {
        cfg: *cfg,
        weights,
        kernels: Glm5NextDsaLayerKernels::resolve(gpu)?,
        select_kernels: Glm5NextDsaKernels::resolve(gpu)?,
        decode_kernel: Glm5NextDsaDecodeKernel::resolve(gpu)?,
        workspace: Glm5NextDsaWorkspace::new(gpu, cfg, 16)?,
        layer_idx: 1,
        attn_layer_idx: 0,
        rms_eps: 1e-6,
        kv_scale: 1.0,
        persist_bt: true,
        indexer_cache: metrale_model_arch::glm5next_dsa::paged::IndexerCache::Flat,
    })
}

/// 2026-10-08: Metadata rows for `seqs` (position, slot, seq_len, block-table row), at fixed
/// strides, as `upload_batch_metadata_fixed` lays them out.
fn meta(gpu: &dyn GpuBackend, seqs: &[(usize, Vec<u32>)]) -> Result<AttnMetadataDev> {
    let positions: Vec<i32> = seqs.iter().map(|(l, _)| *l as i32).collect();
    let slots: Vec<u8> = seqs
        .iter()
        .flat_map(|(l, bt)| ((bt[l / BLOCK] as usize * BLOCK + l % BLOCK) as i64).to_le_bytes())
        .collect();
    let lens: Vec<i32> = seqs.iter().map(|(l, _)| *l as i32 + 1).collect();
    let bts: Vec<i32> = seqs
        .iter()
        .flat_map(|(_, bt)| bt.iter().map(|b| *b as i32))
        .collect();
    let p = up_i32(gpu, &positions)?;
    Ok(AttnMetadataDev {
        positions: p,
        positions_h: p,
        positions_w: p,
        slot: up(gpu, &slots)?,
        seq_len: up_i32(gpu, &lens)?,
        block_table: up_i32(gpu, &bts)?,
        max_blocks_per_seq: MB as u32,
        num_seqs: seqs.len() as u32,
        seq_slot: DevicePtr::NULL,
        moe_row_adapter: DevicePtr::NULL,
    })
}

struct Fwd {
    config: ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
}

impl Fwd {
    fn ctx<'a>(
        &'a self,
        gpu: &'a dyn GpuBackend,
        capture: bool,
        meta: Option<AttnMetadataDev>,
    ) -> ForwardContext<'a> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: meta,
            profile: false,
            comm: None,
            graph_capture: capture,
            decode_step: true,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }
}

/// 2026-10-08: DSA: C sequences of different lengths, each with its own blocks, history
/// written token by token through the single-sequence decode. Then the next token of each
/// sequence alone, and all C in one `decode_rows`, from the same history.
fn dsa_gate(gpu: &dyn GpuBackend, fwd: &Fwd, rng: &mut Lcg, capture: bool) -> Result<()> {
    #[allow(non_snake_case)]
    let STREAM = gpu.default_stream();
    let cfg = Glm5NextDsaConfig {
        hidden: HIDDEN,
        index_heads: 8,
        index_head_dim: 128,
        index_kpool: 4,
        // 2026-10-08: 4 pools out of up to 19, so the top-k really selects.
        index_topk: 16,
        always_select_tail: true,
        local_heads: 4,
        q_lora_rank: 512,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: MB * BLOCK,
    };
    let layer = dsa_layer(gpu, &cfg, rng)?;
    let row = HIDDEN * 2;
    for c in CS {
        let mut kv = PagedKvCache::new(
            KvCacheConfig {
                block_size: BLOCK,
                num_kv_heads: 1,
                head_dim: cfg.kv_lora_rank,
                num_layers: 1,
                dtype: KvCacheDtype::Fp8,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            c * MB,
            gpu,
        )?;
        let seqs: Vec<(usize, Vec<u32>)> = (0..c)
            .map(|r| {
                (
                    17 + (r * 29) % 63,
                    (0..MB).map(|b| (r * MB + b) as u32).collect(),
                )
            })
            .collect();
        let mut states: Vec<Box<dyn LayerState>> = Vec::with_capacity(c);
        for (len, bt) in &seqs {
            let mut st: Box<dyn LayerState> = Box::new(Glm5NextDsaState::alloc(gpu, &cfg)?);
            let mut bt = bt.clone();
            let x = gpu.alloc(row)?;
            for t in 0..*len {
                gpu.copy_h2d(
                    &rng.vec(HIDDEN, 1.0)
                        .iter()
                        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
                        .collect::<Vec<_>>(),
                    x,
                )?;
                let ctx = fwd.ctx(gpu, false, None);
                layer.decode_k(x, 1, st.as_mut(), &mut kv, t, &mut bt, &ctx, STREAM, false)?;
            }
            states.push(st);
        }
        let hidden = up_bf16(gpu, &rng.vec(c * HIDDEN, 1.0))?;
        let slot_bytes = |len: usize, bt: &[u32]| {
            (bt[len / BLOCK] as usize * BLOCK + len % BLOCK) * cfg.kv_lora_rank
        };
        let pool = kv.k_pool_ptr(0);
        let indexer = |st: &dyn LayerState, len: usize| -> Result<Vec<u8>> {
            let s = st
                .as_any()
                .downcast_ref::<Glm5NextDsaState>()
                .context("DSA state")?;
            let mut v = read(
                gpu,
                s.k_normed.offset(s.row_offset(len)),
                cfg.index_head_dim * 2,
            )?;
            v.extend(read(
                gpu,
                s.gate.offset(s.row_offset(len)),
                cfg.index_head_dim * 2,
            )?);
            Ok(v)
        };

        let mut alone = Vec::with_capacity(c);
        for (r, (len, bt)) in seqs.iter().enumerate() {
            let x = copy(gpu, hidden.offset(r * row), row)?;
            let m = meta(gpu, &seqs[r..r + 1])?;
            let ctx = fwd.ctx(gpu, capture, Some(m));
            let mut bt = bt.clone();
            layer.decode_k(
                x,
                1,
                states[r].as_mut(),
                &mut kv,
                *len,
                &mut bt,
                &ctx,
                STREAM,
                false,
            )?;
            alone.push((
                read(gpu, x, row)?,
                indexer(states[r].as_ref(), *len)?,
                read(
                    gpu,
                    pool.offset(slot_bytes(*len, bt.as_slice())),
                    cfg.kv_lora_rank,
                )?,
            ));
            let s = states[r]
                .as_any_mut()
                .downcast_mut::<Glm5NextDsaState>()
                .context("DSA")?;
            s.rewind_to(*len)?;
        }

        let m = meta(gpu, &seqs)?;
        let ctx = fwd.ctx(gpu, capture, Some(m));
        let lens: Vec<usize> = seqs.iter().map(|(l, _)| *l).collect();
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            states.iter_mut().map(|b| b.as_mut()).collect();
        layer.decode_rows(hidden, &mut refs, &lens, &mut kv, &m, 0, &ctx, STREAM)?;
        for (r, (len, bt)) in seqs.iter().enumerate() {
            let tag = format!("DSA capture={capture} C={c} row {r} (len {len})");
            same(
                &format!("{tag} output"),
                &alone[r].0,
                &read(gpu, hidden.offset(r * row), row)?,
            )?;
            same(
                &format!("{tag} indexer row"),
                &alone[r].1,
                &indexer(states[r].as_ref(), *len)?,
            )?;
            same(
                &format!("{tag} KV latent"),
                &alone[r].2,
                &read(gpu, pool.offset(slot_bytes(*len, bt)), cfg.kv_lora_rank)?,
            )?;
        }
        println!(
            "  DSA  C={c:>2} capture={capture}: {c} rows byte-identical (output, indexer, latent)"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    // 2026-10-08: The mixers read only the GPU, the metadata and the flags from the context;
    // the arena exists because `ForwardContext` holds one, at the mock tests' small shape.
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
    let mut rng = Lcg(0x6c6d_5e9d_0001);
    println!("GATE: batched multi-sequence decode mixers vs one sequence at a time");
    kda_gate(&gpu, &mut rng)?;
    dsa_gate(&gpu, &fwd, &mut rng, false)?;
    dsa_gate(&gpu, &fwd, &mut rng, true)?;
    println!("PASS");
    Ok(())
}
