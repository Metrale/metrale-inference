// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Byte-identity gate for the GLM-5.3 batched multi-sequence decode mixers: KDA
//! `decode_rows` and DSA `decode_rows` over C sequences against each sequence decoded alone
//! (`Glm5NextKdaLayer::decode`, `Glm5NextDsaLayer::decode_k` at k = 1 on a decode step), at
//! C = 1 (2026-10-09), 2, 4 and 16, on synthetic weights through the real kernels. The levers
//! that batch rows (`METRALE_GLM_KDA_SEQ_ROWS`, `METRALE_GLM_KDA_ROWS_REG`,
//! `METRALE_GLM_DSA_INDEXER_ROWS`) are read from the environment, so running the gate with them
//! set checks the batched arms against the same single-sequence references.
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

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Context, Result};
use fixture::*;
use half::bf16;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_arch::glm5next_kda::KdaSeqState;
use metrale_model_layers::layer::LayerState;

const CS: [usize; 4] = [1, 2, 4, 16];

/// 2026-10-08: KDA: C states drawn independently; each sequence decoded alone from a copy,
/// then all C in one `decode_rows` from another copy.
fn kda_gate(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let (cfg, layer, ws) = kda_fixture(gpu, rng)?;
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

/// 2026-10-08: DSA: C sequences of different lengths, each with its own blocks, history
/// written token by token through the single-sequence decode. Then the next token of each
/// sequence alone, and all C in one `decode_rows`, from the same history.
fn dsa_gate(gpu: &dyn GpuBackend, fwd: &Fwd, rng: &mut Lcg, capture: bool) -> Result<()> {
    let cfg = dsa_cfg();
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
    let fwd = Fwd::new(&gpu)?;
    let mut rng = Lcg(0x6c6d_5e9d_0001);
    println!("GATE: batched multi-sequence decode mixers vs one sequence at a time");
    kda_gate(&gpu, &mut rng)?;
    dsa_gate(&gpu, &fwd, &mut rng, false)?;
    dsa_gate(&gpu, &fwd, &mut rng, true)?;
    println!("PASS");
    Ok(())
}
