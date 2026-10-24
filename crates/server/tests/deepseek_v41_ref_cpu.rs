// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: The DeepSeek-V4.1 CPU reference on the real checkpoint: the engine's config
//! and tokenizer, the compressed-vocab token map and hash tables the engine derives, and every
//! weight in its stored precision through the memory-mapped checkpoint. It writes the
//! teacher-forced prefill logits and a short greedy continuation, the reference the GPU path
//! is scored against.
//!
//! Inputs: `METRALE_DSV41_CKPT` (the snapshot directory), `METRALE_DSV41_PROMPT` (a UTF-8 file
//! holding the prompt text, encoded after the BOS id) and `METRALE_DSV41_REF_OUT` (an existing
//! directory for the outputs). Outputs there:
//! - `ids.txt`: the prompt ids, then the greedy ids, one per line;
//! - `prefill_logits.f32`: `[prompt_len, vocab]` little-endian f32;
//! - `decode_logits.f32`: `[NEW_TOKENS, vocab]`, the logits that chose each greedy id;
//! - `routes.u16`: `[layers, prompt_len, topk]` little-endian u16, the prefill's routed expert ids;
//! - `summary.txt`: per-position top-5 and the decoded continuation.

use std::fmt::Write as _;
mod dsv41_env;

use std::time::Instant;

use metrale_model_arch::deepseek_v41_ref::checkpoint::model_weights;
use metrale_model_arch::deepseek_v41_ref::engram::EngramTables;
use metrale_model_arch::deepseek_v41_ref::model::{ModelCfg, ModelState, Semantics, forward};
use metrale_model_arch::engram_v41::EngramHashTables;
use metrale_model_weights::fast_weights::MappedSafetensors;
use metrale_server::tokenizer::engram_vocab::CompressedTokenMap;
use tokenizers::Tokenizer;

/// 2026-10-19: Greedy steps after the prefill; each is one single-token decode step.
const NEW_TOKENS: usize = 8;

fn top5(row: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx.into_iter().take(5).map(|i| (i, row[i])).collect()
}

fn le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore = "needs METRALE_DSV41_CKPT, METRALE_DSV41_PROMPT and METRALE_DSV41_REF_OUT; a long CPU run"]
fn reference_prefill_and_greedy_on_the_real_checkpoint() {
    let env = dsv41_env::path;
    let (dir, out) = (env("METRALE_DSV41_CKPT"), env("METRALE_DSV41_REF_OUT"));
    let prompt = std::fs::read_to_string(env("METRALE_DSV41_PROMPT")).expect("prompt file");
    let read = |p: &str| std::fs::read_to_string(dir.join(p)).expect("snapshot file");
    let mc = metrale_config::parse_config(&read("config.json")).expect("engine config");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");

    let mut ids: Vec<i64> = vec![i64::from(mc.bos_token_id)];
    let enc = tok.encode(prompt.as_str(), false).expect("encode");
    ids.extend(enc.get_ids().iter().map(|&i| i64::from(i)));
    let prompt_len = ids.len();

    let map = CompressedTokenMap::build(&tok, mc.engram_compressed_vocab_size as u64)
        .expect("compressed token map");
    let hash = EngramHashTables::from_flat(
        mc.engram_layer_ids.clone(),
        mc.engram_max_ngram_size,
        mc.engram_n_heads,
        mc.engram_pad_token_id,
        map.lookup().iter().map(|&x| i64::from(x)).collect(),
        &mc.engram_multipliers,
        &mc.engram_primes,
        &mc.engram_offsets,
    )
    .expect("engram hash tables");
    let rows = mc
        .engram_num_embeddings
        .iter()
        .map(|&r| usize::try_from(r).unwrap())
        .collect();
    let t = EngramTables::from_hash_tables(hash, mc.engram_head_dim, rows);
    let c = ModelCfg::from_model_config(&mc, prompt_len + NEW_TOKENS, Semantics::REVISION_1017);

    let clock = Instant::now();
    let src = MappedSafetensors::open(&dir).expect("map the checkpoint");
    let w = model_weights(&src, &c, &t).expect("checkpoint weights");
    let mut summary = String::new();
    writeln!(
        summary,
        "weights bound in {:.1} s, {} tensors",
        clock.elapsed().as_secs_f64(),
        src.len()
    )
    .unwrap();

    let mut st = ModelState::new(&c, &t);
    let clock = Instant::now();
    let pre = forward(&ids, 0, &c, &w, &mut st, &t);
    writeln!(
        summary,
        "prefill {prompt_len} tokens in {:.1} s",
        clock.elapsed().as_secs_f64()
    )
    .unwrap();
    assert!(
        pre.logits.iter().all(|v| v.is_finite()),
        "non-finite prefill logits"
    );
    std::fs::write(out.join("prefill_logits.f32"), le_bytes(&pre.logits)).unwrap();
    let routes: Vec<u8> = pre
        .layers
        .iter()
        .flat_map(|l| l.moe_indices.iter())
        .flat_map(|&e| u16::try_from(e).expect("expert id fits u16").to_le_bytes())
        .collect();
    std::fs::write(out.join("routes.u16"), routes).unwrap();
    for (p, row) in pre.logits.chunks(c.vocab).enumerate() {
        writeln!(
            summary,
            "prefill pos {p} id {} top5 {:?}",
            ids[p],
            top5(row)
        )
        .unwrap();
    }

    let mut decode_logits = Vec::with_capacity(NEW_TOKENS * c.vocab);
    let mut last = pre.logits[(prompt_len - 1) * c.vocab..].to_vec();
    for step in 0..NEW_TOKENS {
        let next = top5(&last)[0].0 as i64;
        decode_logits.extend_from_slice(&last);
        ids.push(next);
        if step + 1 == NEW_TOKENS {
            break;
        }
        let clock = Instant::now();
        let s = forward(&[next], prompt_len + step, &c, &w, &mut st, &t);
        assert!(
            s.logits.iter().all(|v| v.is_finite()),
            "non-finite decode logits"
        );
        writeln!(
            summary,
            "decode step {step} in {:.1} s top5 {:?}",
            clock.elapsed().as_secs_f64(),
            top5(&s.logits)
        )
        .unwrap();
        last = s.logits;
    }
    std::fs::write(out.join("decode_logits.f32"), le_bytes(&decode_logits)).unwrap();
    let all: Vec<String> = ids.iter().map(i64::to_string).collect();
    std::fs::write(out.join("ids.txt"), all.join("\n") + "\n").unwrap();
    let generated: Vec<u32> = ids[prompt_len..].iter().map(|&i| i as u32).collect();
    writeln!(
        summary,
        "continuation: {:?}",
        tok.decode(&generated, false).expect("decode")
    )
    .unwrap();
    std::fs::write(out.join("summary.txt"), &summary).unwrap();
    print!("{summary}");
}
