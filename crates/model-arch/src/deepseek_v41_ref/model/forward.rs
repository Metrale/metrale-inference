// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The forward for one chunk of the DeepSeek-V4.1 tiny graph: embed,
//! expand to `hc_mult` copies, the one-hot initial pre-mix, engram, the
//! blocks with the delayed mixes, collapse, final RMSNorm, f32 logits.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use super::super::attn::AttnWeights;
use super::super::compress::{CompressorWeights, IndexerWeights, attention_any};
use super::super::engram::{EngramTables, gate_and_add};
use super::super::hc::{hc_mixes, hc_post, hc_pre, rms_norm};
use super::super::mat::{OutPrec, Weight, gather_rows, linear};
use super::super::moe::{MoeWeights, moe};
use super::{LayerTrace, ModelCfg, ModelState, ModelWeights, StepTrace};

/// 2026-09-25: The forward for one chunk (`[seqlen]` ids at `start_pos`), text only.
pub fn forward<M: Weight>(
    ids: &[i64],
    start_pos: usize,
    c: &ModelCfg,
    w: &ModelWeights<M>,
    st: &mut ModelState,
    t: &EngramTables,
) -> StepTrace {
    let (dim, hc, s) = (c.dim, c.hc, ids.len());
    let hashes = st.hash.forward(ids, s, start_pos);
    let (nl, cols) = (t.layer_ids.len(), t.n_hash_cols());

    let embed = gather_rows(&w.embed, dim, ids);
    let mut h: Vec<f32> = (0..s)
        .flat_map(|tk| std::iter::repeat_n(embed[tk * dim..(tk + 1) * dim].to_vec(), hc).flatten())
        .collect();
    let mut pre_mix: Vec<f32> = (0..s)
        .flat_map(|_| (0..hc).map(|stream| if stream == 0 { 1.0 } else { 0.0 }))
        .collect();
    let icfg = c.indexer_cfg();
    let mut traces = Vec::with_capacity(c.n_layers);

    for l in 0..c.n_layers {
        let lw = &w.layers[l];
        let hs = &hashes;
        let engram_out = lw.engram.as_ref().map(|e| {
            let hi = e.hash_index;
            let ids_l: Vec<i64> = (0..s)
                .flat_map(|tk| (0..cols).map(move |cc| hs[(tk * nl + hi) * cols + cc]))
                .collect();
            let emb = gather_rows(&e.table, t.head_dim, &ids_l);
            let in_f = cols * t.head_dim;
            let kv = linear(&e.wkv, &emb, s, in_f, dim * (hc + 1), OutPrec::Bf16);
            let out = gate_and_add(&h, &kv, &e.q, &e.k, s, hc, dim, c.eps);
            h = out.clone();
            out
        });
        let h_in = h.clone();
        let pre_mix_in = pre_mix.clone();

        let (attn_pre, attn_post, attn_comb) = hc_mixes(
            &h,
            s,
            hc,
            dim,
            &lw.hc_attn_fn,
            &lw.hc_attn_scale,
            &lw.hc_attn_base,
            c.iters,
            c.hc_eps,
            c.eps,
        );
        let attn_in = rms_norm(
            &hc_pre(&h, &pre_mix, s, hc, dim),
            &lw.attn_norm,
            s,
            dim,
            c.eps,
        );
        let aw = AttnWeights {
            sink: &lw.sink,
            wq_a: &lw.wq_a,
            q_norm: &lw.q_norm,
            wq_b: &lw.wq_b,
            wkv: &lw.wkv,
            kv_norm: &lw.kv_norm,
            wo_a: &lw.wo_a,
            wo_b: &lw.wo_b,
        };
        let comp = lw.comp_wkv.as_ref().map(|wkv| CompressorWeights {
            wkv,
            wgate: lw.comp_wgate.as_ref(),
            norm: lw.comp_norm.as_deref().expect("comp norm"),
        });
        let idxw = lw.idx_wq_b.as_ref().map(|wq_b| IndexerWeights {
            wq_b,
            weights_proj: lw.idx_weights_proj.as_ref().expect("weights_proj"),
            wk: lw.idx_wk.as_ref(),
            k_norm: lw.idx_k_norm.as_deref(),
        });
        let ac = c.attn_cfg(l);
        let attn = attention_any(
            &attn_in,
            s,
            start_pos,
            &aw,
            comp.as_ref(),
            idxw.as_ref(),
            &icfg,
            &ac,
            &st.fcs[l],
            &mut st.layers[l],
            &mut st.shared,
        );
        let h_mid = hc_post(&attn.out, &h, &attn_post, &attn_comb, s, hc, dim);

        let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(
            &h_mid,
            s,
            hc,
            dim,
            &lw.hc_ffn_fn,
            &lw.hc_ffn_scale,
            &lw.hc_ffn_base,
            c.iters,
            c.hc_eps,
            c.eps,
        );
        let ffn_in = rms_norm(
            &hc_pre(&h_mid, &attn_pre, s, hc, dim),
            &lw.ffn_norm,
            s,
            dim,
            c.eps,
        );
        let mw = MoeWeights {
            gate_w: &lw.gate_w,
            gate_bias: &lw.gate_bias,
            experts: lw.experts.iter().map(|(a, b, cc)| (a, b, cc)).collect(),
            shared: (&lw.shared.0, &lw.shared.1, &lw.shared.2),
        };
        let (ffn_out, moe_weights, moe_indices) = moe(&ffn_in, s, &mw, &c.moe);
        let h_out = hc_post(&ffn_out, &h_mid, &ffn_post, &ffn_comb, s, hc, dim);

        h = h_out.clone();
        pre_mix = ffn_pre.clone();
        traces.push(LayerTrace {
            engram_out,
            h_in,
            pre_mix_in,
            attn_pre,
            attn_post,
            attn_comb,
            attn_in,
            attn,
            ffn_pre,
            ffn_post,
            ffn_comb,
            ffn_in,
            moe_weights,
            moe_indices,
            ffn_out,
            h_out,
            shared_compress_kv: st.shared.compress_kv.clone(),
            shared_index_k: st.shared.index_k.clone(),
            shared_topk: st.shared.topk_idxs.clone(),
            shared_topk_w: st.shared.topk,
            shared_cand: st.shared.candidates.clone(),
            shared_cand_w: st.shared.cand_width,
        });
    }

    let h_final = hc_pre(&h, &pre_mix, s, hc, dim);
    let head_in = rms_norm(&h_final, &w.norm, s, dim, c.eps);
    let logits = linear(&w.head, &head_in, s, dim, c.vocab, OutPrec::F32);
    StepTrace {
        tokens: s,
        embed,
        layers: traces,
        h_final,
        head_in,
        logits,
    }
}
