// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Standalone eager native GPT-OSS teacher-forced full forward.
//! Usage: MODEL_DIR MODULES_JSON TOKEN_IDS_JSON OUTPUT_JSON
//! Emits BF16 logits/hidden traces, not a serving or performance certification.
#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("gpt_oss_full_forward requires cuda feature")
}
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    run()
}

#[cfg(feature = "cuda")]
fn run() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
    use metrale_gpu_runtime::{cuda_backend::MetraleCudaBackend, gpu::GpuBackend};
    use metrale_model_arch::weight_loader::gpt_oss::{GptOssCheckpoint, runtime::GptOssLayer};
    use metrale_model_layers::{
        layer::{LayerState, TransformerLayer},
        layers::ops,
        weight_map::DenseWeight,
    };
    use metrale_model_weights::weights::{SafetensorsLoader, WeightLoader};
    use std::io::Write;
    use std::{path::PathBuf, time::Instant};
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 4,
        "expected MODEL_DIR MODULES_JSON TOKEN_IDS_JSON OUTPUT_JSON"
    );
    let model = PathBuf::from(&args[0]);
    let manifest_path = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[3]);
    ensure!(!output.exists(), "refusing to overwrite result receipt");
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    ensure!(
        !tokens.is_empty() && tokens.len() <= 512,
        "harness requires 1..512 explicit token IDs"
    );
    let config =
        metrale_config::parse_config(&std::fs::read_to_string(model.join("config.json"))?)?;
    ensure!(
        tokens.iter().all(|&t| (t as usize) < config.vocab_size),
        "token ID outside vocabulary"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let modules = manifest["modules"]
        .as_array()
        .context("module manifest needs modules array")?;
    let mut loaded = Vec::new();
    for module in modules {
        let name = module["name"].as_str().context("module name missing")?;
        let ptx = module["ptx"].as_str().context("PTX path missing")?;
        let path = manifest_path
            .parent()
            .context("manifest parent missing")?
            .join(ptx);
        // 2026-10-07: Backend API takes static modules; process lifetime owns these bytes.
        let name: &'static str = Box::leak(name.to_owned().into_boxed_str());
        let bytes: &'static [u8] = Box::leak(std::fs::read(path)?.into_boxed_slice());
        loaded.push((name, bytes));
    }
    let gpu = MetraleCudaBackend::new(0, &loaded)?;
    let stream = gpu.default_stream();
    for module in modules {
        for symbol in module["required_symbols"]
            .as_array()
            .context("missing symbols")?
        {
            gpu.kernel(
                module["name"].as_str().unwrap(),
                symbol.as_str().context("invalid symbol")?,
            )?;
        }
    }
    let total = gpu.total_memory()?;
    let minimum_free = total.div_ceil(100) * 15;
    let cache_bytes = tokens.len().div_ceil(16) * 16 * 8 * 64 * 2 * 2 * 24;
    let reserve = minimum_free + cache_bytes + 64 * 1024 * 1024;
    ensure!(
        gpu.free_memory()? > reserve,
        "insufficient memory with 15 percent reserve"
    );
    let load_start = Instant::now();
    let store = SafetensorsLoader::new().load(&model, &gpu, reserve)?;
    let checkpoint = GptOssCheckpoint::bind(&store, &config)?;
    let layers: Vec<_> = checkpoint
        .layers
        .iter()
        .enumerate()
        .map(|(i, w)| GptOssLayer::new(w, &config, i, &gpu))
        .collect::<anyhow::Result<_>>()?;
    let mut states: Vec<Box<dyn LayerState>> = layers
        .iter()
        .map(|l| l.alloc_state(&gpu))
        .collect::<anyhow::Result<_>>()?;
    let mut cache = PagedKvCache::new(
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 64,
            num_layers: 24,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        tokens.len().div_ceil(16),
        &gpu,
    )?;
    let hidden = gpu.alloc(2880 * 2)?;
    let normed = gpu.alloc(2880 * 2)?;
    let logits = gpu.alloc(config.vocab_size * 2)?;
    ensure!(
        gpu.free_memory()? >= minimum_free,
        "memory use exceeds 0.85 cap"
    );
    let norm_kernel = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    let head_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let logits_path = output.with_extension("logits.bf16");
    let hidden_path = output.with_extension("layers.bf16");
    let mut logits_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&logits_path)?;
    let mut hidden_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&hidden_path)?;
    let mut blocks = vec![];
    let mut times = vec![];
    let mut next_ids = vec![];
    for (position, &token) in tokens.iter().enumerate() {
        let start = Instant::now();
        gpu.copy_d2d_async(
            checkpoint.embedding.ptr().offset(token as usize * 2880 * 2),
            hidden,
            2880 * 2,
            stream,
        )?;
        for (layer, state) in layers.iter().zip(states.iter_mut()) {
            layer.forward_token(
                hidden,
                state.as_mut(),
                &mut cache,
                position,
                &mut blocks,
                &gpu,
                stream,
            )?;
            let mut row = vec![0u8; 2880 * 2];
            gpu.copy_d2h_on_stream(hidden, &mut row, stream)?;
            hidden_file.write_all(&row)?;
        }
        ops::rms_norm(
            &gpu,
            norm_kernel,
            hidden,
            &DenseWeight {
                weight: checkpoint.final_norm.ptr(),
            },
            normed,
            1,
            2880,
            config.rms_norm_eps as f32,
            stream,
        )?;
        ops::dense_gemv(
            &gpu,
            head_kernel,
            normed,
            &DenseWeight {
                weight: checkpoint.head.ptr(),
            },
            logits,
            config.vocab_size as u32,
            2880,
            stream,
        )?;
        let mut result = vec![0u8; config.vocab_size * 2];
        gpu.copy_d2h_on_stream(logits, &mut result, stream)?;
        let values: Vec<_> = result
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite native logits at token position {position}"
        );
        let next = greedy_argmax(&values).context("empty vocabulary")?;
        next_ids.push(next);
        logits_file.write_all(&result)?;
        times.push(start.elapsed().as_secs_f64());
        println!(
            "native position={position} input={token} next={next} seconds={:.4}",
            times.last().unwrap()
        );
    }
    logits_file.sync_all()?;
    hidden_file.sync_all()?;
    let receipt = serde_json::json!({"kind":"native_eager_teacher_forced_forward","expected_model_revision":"6cee5e81ee83917806bbde320786a8fb61efebee","tokens":tokens,"layers":24,"load_seconds":load_seconds,"per_token_seconds_including_trace_copies":times,"next_token_ids":next_ids,"logits":{"path":logits_path,"dtype":"BF16","shape":[tokens.len(),config.vocab_size]},"hidden":{"path":hidden_path,"dtype":"BF16","shape":[tokens.len(),24,2880]},"module_manifest":manifest,"gpu_memory_fraction_limit":0.85,"limitations":["host expert ID readback","scalar prefill","trace I/O included in times","not serving or speed certification"]});
    let mut receipt_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?;
    receipt_file.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    receipt_file.sync_all()?;
    for (layer, state) in layers.iter().zip(states.iter_mut()) {
        layer.release_state(state.as_mut(), &gpu)?;
    }
    gpu.free(hidden)?;
    gpu.free(normed)?;
    gpu.free(logits)?;
    Ok(())
}

// 2026-10-07: Match reference argmax's first/lower vocabulary index on ties.
#[cfg(any(test, feature = "cuda"))]
fn greedy_argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .expect("caller validated finite logits")
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    #[test]
    fn argmax_uses_first_index_on_equal_logits() {
        assert_eq!(super::greedy_argmax(&[1.0, 1.0]), Some(0));
        assert_eq!(super::greedy_argmax(&[-2.0, 5.0, 5.0, 4.0]), Some(1));
        assert_eq!(super::greedy_argmax(&[-0.0, 0.0]), Some(0));
        assert_eq!(super::greedy_argmax(&[]), None);
        // Known-bad former implementation picks the last equal maximum.
        let bad = [1.0f32, 1.0]
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        assert_ne!(bad, super::greedy_argmax(&[1.0, 1.0]).unwrap());
    }
}
