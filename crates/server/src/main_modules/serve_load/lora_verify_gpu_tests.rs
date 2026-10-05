// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: GPU check that an MTP verify applies the active LoRA adapter's attention delta, as
//! non-speculative decode does. Until 2026-10-03 it did not: the verify's K rows run the
//! multi-sequence attention sites, which fold nothing without a slot buffer, and a request that
//! resolves to the active adapter uploaded none.
//!
//! Two attention-only adapters are generated: `a` (active, nonzero) and `z` (the same `A`, a zero
//! `B`: applying it changes nothing). Every sequence prefills under `a` from the same prompt, so
//! the states are equal; then the step runs under `a` or under `z`. Decode must tell them apart
//! (the control: the adapter matters); the verify must too. A repeat of the `a` verify must be
//! byte-identical, or an apparent difference is noise.
//!
//! Every step runs on the same pool slot. Run twice with `METRALE_LORA_GRAPH_REFERENCE=<file>`: first
//! with `METRALE_DEBUG_NO_GRAPH=1` (eager: it writes the steps' logits), then graphed (it requires
//! every step equal to eager's). Until 2026-10-03 the graphed run failed: the slot-keyed graphs
//! replayed the LoRA route they were captured with for the next sequence on the slot.
//! `cargo test --release -p metrale-server --bin met lora_verify_gpu -- --ignored --nocapture`, on
//! a GPU box with `unsloth/Qwen3.8-27B-NVFP4` cached.
//!
//! Owner: server (LoRA).
//! Invariants: none beyond the types.

use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use metrale_model_engine::traits::Model;

use super::load_engine;

const CHECKPOINT: &str = "unsloth/Qwen3.8-27B-NVFP4";
const RANK: usize = 8;

/// 2026-10-03: `unsloth/Qwen3.8-27B-NVFP4`'s attention shapes (checked by the loader, which
/// refuses an adapter tensor of another shape): 64 layers, every 4th full attention; q (with its
/// gate) 2 x 24 x 256 wide, k and v 4 x 256, hidden 5120.
const LAYERS: usize = 64;
const HIDDEN: usize = 5120;
const Q_OUT: usize = 2 * 24 * 256;
const KV_OUT: usize = 4 * 256;
const O_IN: usize = 24 * 256;

fn bf16(x: f32) -> [u8; 2] {
    (((x.to_bits() + 0x7FFF + ((x.to_bits() >> 16) & 1)) >> 16) as u16).to_le_bytes()
}

/// 2026-10-03: A PEFT adapter of q/k/v/o on every full-attention layer; `b_zero` makes `B` zero.
fn write_adapter(dir: &Path, b_zero: bool) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut x: u64 = 0x5eed_1003;
    let mut next = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.08
    };
    let mut tensors: Vec<(String, [usize; 2], Vec<u8>)> = Vec::new();
    for layer in (3..LAYERS).step_by(4) {
        for (m, n_out, k_in) in [
            ("q_proj", Q_OUT, HIDDEN),
            ("k_proj", KV_OUT, HIDDEN),
            ("v_proj", KV_OUT, HIDDEN),
            ("o_proj", HIDDEN, O_IN),
        ] {
            let base =
                format!("base_model.model.model.language_model.layers.{layer}.self_attn.{m}");
            let a: Vec<u8> = (0..RANK * k_in).flat_map(|_| bf16(next())).collect();
            let b: Vec<u8> = (0..n_out * RANK)
                .flat_map(|_| bf16(if b_zero { 0.0 } else { next() }))
                .collect();
            tensors.push((format!("{base}.lora_A.weight"), [RANK, k_in], a));
            tensors.push((format!("{base}.lora_B.weight"), [n_out, RANK], b));
        }
    }
    tensors.sort_by(|p, q| p.0.cmp(&q.0));
    let mut header = serde_json::Map::new();
    let mut at = 0usize;
    for (name, shape, data) in &tensors {
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [at, at + data.len()] }),
        );
        at += data.len();
    }
    let mut h = serde_json::to_vec(&header)?;
    h.resize(h.len().div_ceil(8) * 8, b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (_, _, data) in &tensors {
        out.extend(data);
    }
    std::fs::write(dir.join("adapter_model.safetensors"), out)?;
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_vec(&serde_json::json!({
            "peft_type": "LORA", "r": RANK, "lora_alpha": 2 * RANK, "lora_dropout": 0.0,
            "use_rslora": false, "bias": "none", "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "base_model_name_or_path": CHECKPOINT,
        }))?,
    )?;
    Ok(())
}

fn logits(model: &dyn Model, rows: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; rows * model.vocab_size() * 2];
    model.copy_logits_to_host(model.logits_buffer_ptr(), &mut out)?;
    Ok(out)
}

/// 2026-10-03: Prefill `prompt` under the active adapter, then one step under `slot`: a decode
/// of the prefill's argmax, or a K = 2 verify of it and one draft. The step's logits.
fn step(model: &dyn Model, prompt: &[u32], slot: i32, verify: bool) -> Result<(usize, Vec<u8>)> {
    let mut seq = model.alloc_sequence()?;
    let pool_slot = seq.slot_idx;
    let out = (|| -> Result<Vec<u8>> {
        seq.adapter_slot = -1;
        let p = model.prefill(prompt, &mut seq, 0)?;
        let mut first = vec![0u8; model.vocab_size() * 2];
        model.copy_logits_to_host(p, &mut first)?;
        let t = first
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0 as u32;
        seq.adapter_slot = slot;
        if verify {
            model.decode_verify_graphed(&[t, t], &mut seq, 0)?;
            logits(model, 2)
        } else {
            let p = model.decode(t, &mut seq, 0)?;
            let mut l = vec![0u8; model.vocab_size() * 2];
            model.copy_logits_to_host(p, &mut l)?;
            Ok(l)
        }
    })();
    model.free_sequence(&mut seq)?;
    Ok((pool_slot, out?))
}

/// 2026-10-03: The steps' logits, concatenated in a fixed order.
fn record(steps: &[&[u8]]) -> Vec<u8> {
    steps.concat()
}

#[test]
#[ignore = "GPU: loads unsloth/Qwen3.8-27B-NVFP4 with two generated LoRA adapters"]
fn an_active_adapter_verify_applies_its_attention_lora_as_decode_does() -> Result<()> {
    let eager = std::env::var("METRALE_DEBUG_NO_GRAPH").as_deref() == Ok("1");
    let reference = std::path::PathBuf::from(
        std::env::var("METRALE_LORA_GRAPH_REFERENCE")
            .context("METRALE_LORA_GRAPH_REFERENCE names the eager run's logits file")?,
    );
    // 2026-10-03: The CUDA backend's allocator reports through the Tokio runtime, as under
    // `met serve` and `met circuit diff`.
    let rt = tokio::runtime::Runtime::new()?;
    let _in_runtime = rt.enter();
    let dir = tempfile::tempdir()?;
    let (a, z) = (dir.path().join("a"), dir.path().join("z"));
    write_adapter(&a, false)?;
    write_adapter(&z, true)?;
    let argv = [
        "met",
        "serve",
        CHECKPOINT,
        "--weight-quantization",
        "nvfp4",
        "--kv-cache-dtype",
        "bf16",
        "--lm-head-dtype",
        "bf16",
        "--gpu-memory-utilization",
        "0.85",
        "--max-batch-size",
        "1",
        "--max-seq-len",
        "4096",
        "--speculative",
        "--num-drafts",
        "1",
        "--mtp-quantization",
        "bf16",
        "--no-tui",
    ];
    let mut argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    for (n, d) in [("a", &a), ("z", &z)] {
        argv.extend(["--lora-adapter".into(), format!("{n}={}", d.display())]);
    }
    let crate::cli::Command::Serve(serve) = crate::cli::Cli::try_parse_from(argv)?.command else {
        unreachable!("a serve command")
    };
    if let Err(msg) = crate::cli::validate_serve_args(&serve) {
        anyhow::bail!("{msg}");
    }
    crate::main_modules::serve_flags::publish_kernel_flags(&serve);
    let engine = load_engine(serve)?.context("rank 0")?;
    let model = engine.model.as_ref();
    model.bind_gpu_to_thread()?;
    let prompt: Vec<u32> = (0..48).map(|i| 1000 + 37 * i).collect();
    // 2026-10-03: `a` loads first, so it is slot 0 and active; `z` is slot 1.
    let (active, zero) = (-1, 1);
    // 2026-10-03: Each step allocates and frees, so every one runs on the same pool slot: under
    // graphs, each reuses the graph keys the step before it captured.
    let mut slots = Vec::new();
    let mut run = |slot: i32, verify: bool| -> Result<Vec<u8>> {
        let (s, l) = step(model, &prompt, slot, verify)?;
        slots.push(s);
        Ok(l)
    };
    let decode_a = run(active, false)?;
    let decode_z = run(zero, false)?;
    assert_ne!(
        decode_a, decode_z,
        "control: decode does not see the adapter at all"
    );
    let verify_a = run(active, true)?;
    let verify_a2 = run(active, true)?;
    let verify_z = run(zero, true)?;
    let decode_a2 = run(active, false)?;
    assert!(
        slots.windows(2).all(|w| w[0] == w[1]),
        "the steps ran on different pool slots {slots:?}: no graph is reused"
    );
    assert_eq!(verify_a, verify_a2, "control: the verify is not repeatable");
    assert_eq!(decode_a, decode_a2, "control: the decode is not repeatable");
    assert_ne!(
        verify_a, verify_z,
        "the verify of an active-adapter request ran its attention without the adapter"
    );
    let got = record(&[&decode_a, &decode_z, &verify_a, &verify_z]);
    if eager {
        std::fs::write(&reference, &got)?;
    } else {
        let want = std::fs::read(&reference)
            .with_context(|| format!("the eager run's logits at {}", reference.display()))?;
        assert!(
            got == want,
            "a graphed step differs from the eager step (a slot's graph replayed another \
             adapter's LoRA route)"
        );
    }
    Ok(())
}
