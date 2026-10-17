// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte-identity gate for the GLM-5.3 batched speculative verify mixers: C
//! sequences of k rows each, packed sequence-major and run in groups of
//! `DENSE_GEMV_BATCHM_MAX_M` rows as `Glm5NextLayer::forward_spans` runs them (KDA
//! `decode_rows_then` with each sequence's state repeated over its rows, DSA `decode_spans`),
//! against each sequence's own k-row verify (KDA `decode_k`, DSA `decode_k`), at
//! (C, k) = (2, 8), (3, 5), (4, 5) and (16, 8). (4, 5) puts a sequence across two groups.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every row's output, every sequence's final KDA recurrent and
//!   conv state, its per-row snapshots (snapshot rollback) and its verify record (replay
//!   rollback), and every DSA indexer row and KV latent the rows wrote are byte-identical to
//!   the single-sequence verify's.
//!
//! The mHC, norm and MLP launches of a batched layer are covered by their own byte gates; the
//! group split by `group_spans` by its unit tests.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_verify_spans_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Context, Result};
use fixture::*;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::layer::DsaRowSpan;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_arch::glm5next_kda::{KdaSeqState, KdaVerifyRecord};
use metrale_model_arch::glm5next_layer::{GroupSpan, group_spans};
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

const CASES: [(usize, usize); 4] = [(2, 8), (3, 5), (4, 5), (16, 8)];

/// 2026-10-09: `(base, m, spans)` of every row group of a pass of `ks` rows.
fn groups(ks: &[usize]) -> Vec<(usize, usize, Vec<GroupSpan>)> {
    let total: usize = ks.iter().sum();
    let cap = DENSE_GEMV_BATCHM_MAX_M as usize;
    (0..total)
        .step_by(cap)
        .map(|base| {
            let m = cap.min(total - base);
            (base, m, group_spans(ks, base, m))
        })
        .collect()
}

/// 2026-10-09: One KDA sequence's buffers: state, k - 1 snapshots and a k - 1 row record.
struct KdaSeq {
    st: KdaSeqState,
    snaps: Vec<(DevicePtr, DevicePtr)>,
    record: KdaVerifyRecord,
    record_base: DevicePtr,
}

fn kda_gate(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let (cfg, layer, ws) = kda_fixture(gpu, rng)?;
    let (cb, hb) = (cfg.conv_state_elems() * 4, cfg.recurrent_state_elems() * 4);
    let out_row = HIDDEN * 2;
    for (c, k) in CASES {
        let rec_bytes = (k - 1) * cfg.replay_row_bytes();
        let init: Vec<KdaSeqState> = (0..c)
            .map(|_| {
                Ok(KdaSeqState {
                    conv: up_f32(gpu, &rng.vec(cb / 4, 0.5))?,
                    recurrent: up_f32(gpu, &rng.vec(hb / 4, 0.1))?,
                })
            })
            .collect::<Result<_>>()?;
        let hidden = up_bf16(gpu, &rng.vec(c * k * HIDDEN, 1.0))?;
        let fork = |s: &KdaSeqState| -> Result<KdaSeq> {
            let record_base = gpu.alloc(rec_bytes)?;
            Ok(KdaSeq {
                st: KdaSeqState {
                    conv: copy(gpu, s.conv, cb)?,
                    recurrent: copy(gpu, s.recurrent, hb)?,
                },
                snaps: (0..k - 1)
                    .map(|_| Ok((gpu.alloc(hb)?, gpu.alloc(cb)?)))
                    .collect::<Result<_>>()?,
                record: KdaVerifyRecord::new(&cfg, record_base, rec_bytes),
                record_base,
            })
        };
        let alone: Vec<KdaSeq> = init.iter().map(fork).collect::<Result<_>>()?;
        let batch: Vec<KdaSeq> = init.iter().map(fork).collect::<Result<_>>()?;

        let mut want_out = Vec::with_capacity(c);
        for (s, q) in alone.iter().enumerate() {
            let x = copy(gpu, hidden.offset(s * k * out_row), k * out_row)?;
            layer.decode_k(gpu, x, k, &q.st, &ws, &q.snaps, STREAM)?;
            want_out.push(read(gpu, ws.final_out, k * out_row)?);
            layer.record_verify_rows(gpu, &ws, k - 1, &q.record, STREAM)?;
        }

        let ks = vec![k; c];
        let mut got_out = vec![0u8; c * k * out_row];
        for (base, m, spans) in groups(&ks) {
            let row_states: Vec<KdaSeqState> = spans
                .iter()
                .flat_map(|sp| std::iter::repeat_n(batch[sp.seq].st, sp.rows))
                .collect();
            let x = copy(gpu, hidden.offset(base * out_row), m * out_row)?;
            layer.decode_rows_then(gpu, x, &row_states, &ws, STREAM, |row| {
                let sp = spans
                    .iter()
                    .find(|sp| (sp.row0..sp.row0 + sp.rows).contains(&row))
                    .context("row outside every span")?;
                match batch[sp.seq].snaps.get(sp.t0 + row - sp.row0) {
                    Some(dst) => layer.snapshot_state(gpu, &row_states[row], *dst, STREAM),
                    None => Ok(()),
                }
            })?;
            got_out[base * out_row..(base + m) * out_row].copy_from_slice(&read(
                gpu,
                ws.final_out,
                m * out_row,
            )?);
            for sp in &spans {
                let end = (sp.t0 + sp.rows).min(k - 1);
                if end > sp.t0 {
                    let q = &batch[sp.seq];
                    layer.record_verify_rows_at(
                        gpu,
                        &ws,
                        sp.row0,
                        sp.t0,
                        end - sp.t0,
                        &q.record,
                        STREAM,
                    )?;
                }
            }
        }

        for s in 0..c {
            let tag = format!("KDA C={c} k={k} seq {s}");
            same(
                &format!("{tag} output rows"),
                &want_out[s],
                &got_out[s * k * out_row..(s + 1) * k * out_row],
            )?;
            let (a, b) = (&alone[s], &batch[s]);
            same(
                &format!("{tag} recurrent"),
                &read(gpu, a.st.recurrent, hb)?,
                &read(gpu, b.st.recurrent, hb)?,
            )?;
            same(
                &format!("{tag} conv"),
                &read(gpu, a.st.conv, cb)?,
                &read(gpu, b.st.conv, cb)?,
            )?;
            for t in 0..k - 1 {
                same(
                    &format!("{tag} snapshot {t} recurrent"),
                    &read(gpu, a.snaps[t].0, hb)?,
                    &read(gpu, b.snaps[t].0, hb)?,
                )?;
                same(
                    &format!("{tag} snapshot {t} conv"),
                    &read(gpu, a.snaps[t].1, cb)?,
                    &read(gpu, b.snaps[t].1, cb)?,
                )?;
            }
            same(
                &format!("{tag} verify record"),
                &read(gpu, a.record_base, rec_bytes)?,
                &read(gpu, b.record_base, rec_bytes)?,
            )?;
        }
        println!("  KDA  C={c:>2} k={k}: output, state, snapshots and record byte-identical");
    }
    Ok(())
}

fn dsa_gate(gpu: &dyn GpuBackend, fwd: &Fwd, rng: &mut Lcg, capture: bool) -> Result<()> {
    let cfg = dsa_cfg();
    let layer = dsa_layer(gpu, &cfg, rng)?;
    let row = HIDDEN * 2;
    for (c, k) in CASES {
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
        // 2026-10-09: Histories of 17..79 tokens, so the k verify rows fit the MB blocks.
        let seqs: Vec<(usize, Vec<u32>)> = (0..c)
            .map(|s| {
                (
                    17 + (s * 29) % 63,
                    (0..MB).map(|b| (s * MB + b) as u32).collect(),
                )
            })
            .collect();
        let mut states: Vec<Box<dyn LayerState>> = Vec::with_capacity(c);
        for (len, bt) in &seqs {
            let mut st: Box<dyn LayerState> = Box::new(Glm5NextDsaState::alloc(gpu, &cfg)?);
            let mut bt = bt.clone();
            for t in 0..*len {
                let x = up_bf16(gpu, &rng.vec(HIDDEN, 1.0))?;
                let ctx = fwd.ctx(gpu, false, None);
                layer.decode_k(x, 1, st.as_mut(), &mut kv, t, &mut bt, &ctx, STREAM, false)?;
            }
            states.push(st);
        }
        let hidden = up_bf16(gpu, &rng.vec(c * k * HIDDEN, 1.0))?;
        let rows_of = |s: usize| -> Vec<(usize, Vec<u32>)> {
            (0..k).map(|t| (seqs[s].0 + t, seqs[s].1.clone())).collect()
        };
        let slot_bytes = |pos: usize, bt: &[u32]| {
            (bt[pos / BLOCK] as usize * BLOCK + pos % BLOCK) * cfg.kv_lora_rank
        };
        let pool = kv.k_pool_ptr(0);
        let written = |st: &dyn LayerState, s: usize| -> Result<Vec<u8>> {
            let d = st
                .as_any()
                .downcast_ref::<Glm5NextDsaState>()
                .context("DSA state")?;
            let mut v = Vec::new();
            for (pos, bt) in rows_of(s) {
                v.extend(read(
                    gpu,
                    d.k_normed.offset(d.row_offset(pos)),
                    cfg.index_head_dim * 2,
                )?);
                v.extend(read(
                    gpu,
                    d.gate.offset(d.row_offset(pos)),
                    cfg.index_head_dim * 2,
                )?);
                v.extend(read(
                    gpu,
                    pool.offset(slot_bytes(pos, &bt)),
                    cfg.kv_lora_rank,
                )?);
            }
            Ok(v)
        };

        let mut alone = Vec::with_capacity(c);
        for (s, (len, bt)) in seqs.iter().enumerate() {
            let x = copy(gpu, hidden.offset(s * k * row), k * row)?;
            let m = meta(gpu, &rows_of(s))?;
            let ctx = fwd.ctx(gpu, capture, Some(m));
            let mut bt = bt.clone();
            layer.decode_k(
                x,
                k,
                states[s].as_mut(),
                &mut kv,
                *len,
                &mut bt,
                &ctx,
                STREAM,
                false,
            )?;
            alone.push((read(gpu, x, k * row)?, written(states[s].as_ref(), s)?));
            states[s]
                .as_any_mut()
                .downcast_mut::<Glm5NextDsaState>()
                .context("DSA")?
                .rewind_to(*len)?;
        }

        let all_rows: Vec<(usize, Vec<u32>)> = (0..c).flat_map(&rows_of).collect();
        let m = meta(gpu, &all_rows)?;
        let ctx = fwd.ctx(gpu, capture, Some(m));
        let ks = vec![k; c];
        for (base, _, spans) in groups(&ks) {
            let (s_lo, s_hi) = (spans[0].seq, spans[spans.len() - 1].seq);
            let dsa_spans: Vec<DsaRowSpan> = spans
                .iter()
                .map(|sp| DsaRowSpan {
                    first_pos: seqs[sp.seq].0 + sp.t0,
                    rows: sp.rows,
                })
                .collect();
            let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                states[s_lo..=s_hi].iter_mut().map(|b| b.as_mut()).collect();
            layer.decode_spans(
                hidden.offset(base * row),
                &mut refs,
                &dsa_spans,
                &mut kv,
                &m,
                base,
                &ctx,
                STREAM,
            )?;
        }
        for s in 0..c {
            let tag = format!("DSA capture={capture} C={c} k={k} seq {s}");
            same(
                &format!("{tag} output rows"),
                &alone[s].0,
                &read(gpu, hidden.offset(s * k * row), k * row)?,
            )?;
            same(
                &format!("{tag} indexer rows and KV latents"),
                &alone[s].1,
                &written(states[s].as_ref(), s)?,
            )?;
        }
        println!("  DSA  C={c:>2} k={k} capture={capture}: rows byte-identical");
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
    let mut rng = Lcg(0x6c6d_5e9d_0002);
    println!("GATE: batched verify mixers (sequence spans) vs each sequence's own verify");
    kda_gate(&gpu, &mut rng)?;
    dsa_gate(&gpu, &fwd, &mut rng, false)?;
    dsa_gate(&gpu, &fwd, &mut rng, true)?;
    println!("PASS");
    Ok(())
}
