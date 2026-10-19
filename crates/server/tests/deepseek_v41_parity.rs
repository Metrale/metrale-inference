// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: Scores the engine's teacher-forced logits against the CPU reference's
//! (`deepseek_v41_ref_cpu`): per position the KL divergence and whether the arg-max ids agree.
//!
//! Inputs: `METRALE_DSV41_CKPT` (the snapshot directory the reference ran on, for the
//! vocabulary), `METRALE_DSV41_REF_OUT` (the reference's output directory) and
//! `METRALE_DSV41_CANDIDATE` (the engine's rows, `[positions, vocab]` little-endian f32, row `p`
//! the logits after the reference's ids `0..=p`; extra rows past the reference are ignored).
//!
//! The reference's positions: every prefill row, then the decode rows after the first, which
//! repeats the last prefill row.

mod dsv41_env;

use metrale_model_arch::deepseek_v41_ref::parity::score;

fn f32s(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert!(b.len().is_multiple_of(4), "{}: not f32 rows", p.display());
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "needs METRALE_DSV41_CKPT, METRALE_DSV41_REF_OUT and METRALE_DSV41_CANDIDATE"]
fn engine_logits_against_the_cpu_reference() {
    let env = dsv41_env::path;
    let (dir, out) = (env("METRALE_DSV41_CKPT"), env("METRALE_DSV41_REF_OUT"));
    let cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let vocab = metrale_config::parse_config(&cfg)
        .expect("engine config")
        .vocab_size;
    let mut reference = f32s(&out.join("prefill_logits.f32"));
    let decode = f32s(&out.join("decode_logits.f32"));
    assert!(
        reference.len().is_multiple_of(vocab) && decode.len().is_multiple_of(vocab),
        "reference rows are not of {vocab}"
    );
    reference.extend_from_slice(decode.get(vocab..).unwrap_or_default());
    let candidate = f32s(&env("METRALE_DSV41_CANDIDATE"));
    assert!(
        candidate.len() >= reference.len(),
        "{} candidate rows for {} reference positions",
        candidate.len() / vocab,
        reference.len() / vocab
    );
    let p = score(&reference, &candidate[..reference.len()], vocab).expect("score");
    for (i, q) in p.positions.iter().enumerate() {
        println!(
            "pos {i} kl {:.6} top1 ref {} eng {}{}",
            q.kl,
            q.reference_top1,
            q.candidate_top1,
            if q.reference_top1 == q.candidate_top1 {
                ""
            } else {
                " MISMATCH"
            }
        );
    }
    println!(
        "positions {} mean_kl {:.6} max_kl {:.6} top1 {}/{}",
        p.positions.len(),
        p.mean_kl(),
        p.max_kl(),
        p.top1_matches(),
        p.positions.len()
    );
}
