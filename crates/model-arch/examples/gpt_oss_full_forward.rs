// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Standalone eager native GPT-OSS teacher-forced full forward.
//! Usage: MODEL_DIR MODULES_JSON TOKEN_IDS_JSON OUTPUT_JSON [GENERATION_JSON_OR_DASH] [DIAGNOSTICS_JSON]
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
        (4..=6).contains(&args.len()),
        "expected MODEL_DIR MODULES_JSON TOKEN_IDS_JSON OUTPUT_JSON [GENERATION_JSON_OR_DASH] [DIAGNOSTICS_JSON]"
    );
    let model = PathBuf::from(&args[0]);
    let manifest_path = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[3]);
    ensure!(!output.exists(), "refusing to overwrite result receipt");
    let mut tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
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
    let mut generation = args
        .get(4)
        .filter(|path| *path != "-")
        .map(|path| -> anyhow::Result<Generation> {
            let policy: GenerationPolicy = serde_json::from_slice(&std::fs::read(path)?)?;
            ensure!(
                (1..=256).contains(&policy.max_new_tokens),
                "generation requires 1..256 tokens"
            );
            ensure!(
                !policy.stop_ids.is_empty()
                    && policy
                        .stop_ids
                        .iter()
                        .all(|&id| (id as usize) < config.vocab_size),
                "explicit in-vocabulary stop IDs required"
            );
            Ok(Generation {
                policy,
                ids: vec![],
                reason: None,
            })
        })
        .transpose()?;
    let diagnostics: Option<DiagnosticsPolicy> = args
        .get(5)
        .map(|path| -> anyhow::Result<_> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) })
        .transpose()?;
    if let Some(policy) = &diagnostics {
        ensure!(
            policy
                .layers
                .iter()
                .chain(&policy.cache_layers)
                .all(|&l| l < 24),
            "diagnostic layer outside model"
        );
        ensure!(
            policy.positions.iter().all(|&p| p < 768),
            "diagnostic position outside harness bound"
        );
        std::fs::create_dir(output.with_extension("diagnostics"))?;
    }
    let mut diagnostic_records = vec![];
    let prompt_tokens = tokens.clone();
    let capacity = tokens.len() + generation.as_ref().map_or(0, |g| g.policy.max_new_tokens);
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
    let cache_bytes = capacity.div_ceil(16) * 16 * 8 * 64 * 2 * 2 * 24;
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
        capacity.div_ceil(16),
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
    let mut position = 0;
    while position < tokens.len() {
        let token = tokens[position];
        let start = Instant::now();
        gpu.copy_d2d_async(
            checkpoint.embedding.ptr().offset(token as usize * 2880 * 2),
            hidden,
            2880 * 2,
            stream,
        )?;
        for (layer_index, (layer, state)) in layers.iter().zip(states.iter_mut()).enumerate() {
            let inspect = diagnostics.as_ref().is_some_and(|p| {
                p.layers.contains(&layer_index) && p.positions.contains(&position)
            });
            let mut incoming = vec![];
            if inspect {
                incoming.resize(2880 * 2, 0);
                gpu.copy_d2h_on_stream(hidden, &mut incoming, stream)?;
            }
            layer.forward_token(
                hidden,
                state.as_mut(),
                &mut cache,
                position,
                &mut blocks,
                &gpu,
                stream,
            )?;
            if inspect {
                let mut snapshots =
                    layer.diagnostic_snapshot(state.as_ref(), position, &gpu, stream)?;
                snapshots.push(
                    metrale_model_arch::weight_loader::gpt_oss::runtime::DiagnosticTensor {
                        name: "incoming_hidden",
                        dtype: "BF16",
                        shape: vec![2880],
                        bytes: incoming,
                    },
                );
                if diagnostics
                    .as_ref()
                    .is_some_and(|p| p.cache_layers.contains(&layer_index))
                {
                    for (name, pool) in [
                        ("cache_keys", cache.k_pool_ptr(layer_index)),
                        ("cache_values", cache.v_pool_ptr(layer_index)),
                    ] {
                        let mut bytes = vec![0; (position + 1) * 8 * 64 * 2];
                        for (logical, &physical) in blocks.iter().enumerate() {
                            let start = logical * cache.block_size();
                            if start > position {
                                break;
                            }
                            let count = cache.block_size().min(position + 1 - start);
                            gpu.copy_d2h_on_stream(
                                pool.offset(
                                    physical as usize
                                        * cache.block_stride_bytes_for_layer(layer_index),
                                ),
                                &mut bytes[start * 1024..(start + count) * 1024],
                                stream,
                            )?;
                        }
                        snapshots.push(
                            metrale_model_arch::weight_loader::gpt_oss::runtime::DiagnosticTensor {
                                name,
                                dtype: "BF16",
                                shape: vec![position + 1, 8, 64],
                                bytes,
                            },
                        );
                    }
                    diagnostic_records.push(serde_json::json!({"position":position,"layer":layer_index,"name":"cache_layout","block_size":cache.block_size(),"physical_blocks":blocks,"block_stride_bytes":cache.block_stride_bytes_for_layer(layer_index),"export_order":"logical token, KV head, dimension","attention_scale":0.125,"sliding_window":if config.layer_types[layer_index] == metrale_config::LayerType::SlidingAttention {128} else {0}}));
                }
                for snapshot in snapshots {
                    let path = output
                        .with_extension("diagnostics")
                        .join(format!("p{position}-l{layer_index}-{}.bin", snapshot.name));
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)?;
                    file.write_all(&snapshot.bytes)?;
                    diagnostic_records.push(serde_json::json!({"position":position,"layer":layer_index,"name":snapshot.name,"dtype":snapshot.dtype,"shape":snapshot.shape,"path":path}));
                }
            }
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
        if position + 1 >= prompt_tokens.len()
            && let Some(generation) = generation.as_mut()
        {
            if generation.record(next as u32) {
                break;
            }
            tokens.push(next as u32);
        }
        position += 1;
    }
    logits_file.sync_all()?;
    hidden_file.sync_all()?;
    let receipt = serde_json::json!({"kind":if generation.is_some() {"native_eager_greedy_generation"} else {"native_eager_teacher_forced_forward"},"diagnostics":diagnostic_records,"prompt_tokens":prompt_tokens,"generation":generation,"expected_model_revision":"6cee5e81ee83917806bbde320786a8fb61efebee","tokens":tokens,"layers":24,"load_seconds":load_seconds,"per_token_seconds_including_trace_copies":times,"next_token_ids":next_ids,"logits":{"path":logits_path,"dtype":"BF16","shape":[tokens.len(),config.vocab_size]},"hidden":{"path":hidden_path,"dtype":"BF16","shape":[tokens.len(),24,2880]},"module_manifest":manifest,"gpu_memory_fraction_limit":0.85,"limitations":["host expert ID readback","scalar prefill","trace I/O included in times","not serving or speed certification"]});
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

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct DiagnosticsPolicy {
    positions: Vec<usize>,
    layers: Vec<usize>,
    #[serde(default)]
    cache_layers: Vec<usize>,
}

// A generated stop token is recorded but never fed back into the cache.
#[cfg(any(test, feature = "cuda"))]
#[derive(serde::Deserialize, serde::Serialize)]
struct GenerationPolicy {
    max_new_tokens: usize,
    stop_ids: Vec<u32>,
}
#[cfg(any(test, feature = "cuda"))]
#[derive(serde::Serialize)]
struct Generation {
    policy: GenerationPolicy,
    ids: Vec<u32>,
    reason: Option<&'static str>,
}
#[cfg(any(test, feature = "cuda"))]
impl Generation {
    fn record(&mut self, id: u32) -> bool {
        assert!(self.reason.is_none(), "generation already stopped");
        self.ids.push(id);
        self.reason = if self.policy.stop_ids.contains(&id) {
            Some("stop_token")
        } else if self.ids.len() == self.policy.max_new_tokens {
            Some("length")
        } else {
            None
        };
        self.reason.is_some()
    }
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
    fn generation_records_stop_and_limit_without_extra_feedback() {
        let mut g = super::Generation {
            policy: super::GenerationPolicy {
                max_new_tokens: 2,
                stop_ids: vec![9],
            },
            ids: vec![],
            reason: None,
        };
        assert!(!g.record(4));
        assert!(g.record(5));
        assert_eq!(g.reason, Some("length"));
        assert_eq!(g.ids, [4, 5]);
        g.ids.clear();
        g.reason = None;
        assert!(g.record(9));
        assert_eq!(g.reason, Some("stop_token"));
        assert_eq!(g.ids, [9]);
    }

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
