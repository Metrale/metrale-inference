// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Isolated scalar/chunk hidden, cache and following-decode exact gate.
//! `MODEL MODULE_MANIFEST TOKEN_IDS OUTPUT_DIR [CAPACITY16|64|128]`. No serving or speed claim.
#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires CUDA")
}
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
    use metrale_gpu_runtime::{cuda_backend::MetraleCudaBackend, gpu::GpuBackend};
    use metrale_model_arch::weight_loader::gpt_oss::{
        GptOssCheckpoint,
        runtime::{GptOssLayer, PrefillScratch},
    };
    use metrale_model_layers::layer::{LayerState, TransformerLayer};
    use metrale_model_weights::weights::{SafetensorsLoader, WeightLoader};
    use std::path::PathBuf;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        (4..=5).contains(&args.len()),
        "MODEL MANIFEST TOKENS OUTPUT [CAPACITY]"
    );
    let capacity = args
        .get(4)
        .map(|v| v.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    ensure!(
        [16, 64, 128].contains(&capacity),
        "diagnostic capacity must be16,64,128"
    );
    let widths: &[usize] = match capacity {
        16 => &[0, 1, 2, 8, 16, 15],
        64 => &[0, 16, 31, 63, 64],
        128 => &[0, 16, 31, 64, 127, 128],
        _ => unreachable!(),
    };
    let model = PathBuf::from(&args[0]);
    let manifest_path = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir(&output)?;
    let mut tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    ensure!(
        (130..=512).contains(&tokens.len()),
        "require window-crossing bounded fixture"
    );
    let prefix = tokens.len();
    tokens.push(2);
    let config =
        metrale_config::parse_config(&std::fs::read_to_string(model.join("config.json"))?)?;
    ensure!(
        tokens.iter().all(|&v| (v as usize) < config.vocab_size),
        "invalid tokens"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let mut loaded = vec![];
    for m in manifest["modules"].as_array().context("modules")? {
        let name: &'static str = Box::leak(
            m["name"]
                .as_str()
                .context("name")?
                .to_owned()
                .into_boxed_str(),
        );
        let bytes: &'static [u8] = Box::leak(
            std::fs::read(
                manifest_path
                    .parent()
                    .unwrap()
                    .join(m["ptx"].as_str().context("ptx")?),
            )?
            .into_boxed_slice(),
        );
        loaded.push((name, bytes));
    }
    let gpu = MetraleCudaBackend::new(0, &loaded)?;
    let stream = gpu.default_stream();
    let reserve = gpu.total_memory()?.div_ceil(100) * 15 + 256 * 1024 * 1024;
    let store = SafetensorsLoader::new().load(&model, &gpu, reserve)?;
    let bound = GptOssCheckpoint::bind(&store, &config)?;
    let layers: Vec<_> = bound
        .layers
        .iter()
        .enumerate()
        .map(|(i, w)| GptOssLayer::new(w, &config, i, &gpu))
        .collect::<anyhow::Result<_>>()?;
    let count = tokens.len();
    let pages = count.div_ceil(16);
    let pool_bytes = pages * 16 * 512 * 2;
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
        pages,
        &gpu,
    )?;
    let hidden = gpu.alloc(capacity * 5760)?;
    let mut scratch = PrefillScratch::new(&gpu, capacity, pages)?;
    let poison: Vec<u8> = (0..pool_bytes / 2)
        .flat_map(|_| 0x7fc0u16.to_le_bytes())
        .collect();
    let mut baseline: Option<(Vec<u8>, Vec<u8>)> = None;
    let mut cases = vec![];
    for &width in widths {
        let mut states: Vec<Box<dyn LayerState>> = layers
            .iter()
            .map(|l| l.alloc_state(&gpu))
            .collect::<anyhow::Result<_>>()?;
        for l in 0..24 {
            gpu.copy_h2d_async(&poison, cache.k_pool_ptr(l), stream)?;
            gpu.copy_h2d_async(&poison, cache.v_pool_ptr(l), stream)?;
        }
        let mut blocks = vec![];
        // 2026-10-07: Admission failures must leave the fresh state usable.
        if width != 0 {
            for invalid in [0, capacity + 1, 129] {
                ensure!(
                    layers[0]
                        .forward_chunk(
                            hidden,
                            states[0].as_mut(),
                            &mut cache,
                            0,
                            invalid,
                            &mut blocks,
                            &mut scratch,
                            &gpu,
                            stream
                        )
                        .is_err(),
                    "invalid chunk admitted"
                );
            }
        }

        if width != 0 {
            for invalid_start in [1, 131072, usize::MAX] {
                ensure!(
                    layers[0]
                        .forward_chunk(
                            hidden,
                            states[0].as_mut(),
                            &mut cache,
                            invalid_start,
                            1,
                            &mut blocks,
                            &mut scratch,
                            &gpu,
                            stream
                        )
                        .is_err(),
                    "invalid start admitted"
                );
            }
        }
        let mut trace = vec![0u8; count * 24 * 5760];
        let mut start = 0;
        while start < count {
            let rows = if width == 0 || start == prefix {
                1
            } else {
                width.min(prefix - start)
            };
            for t in 0..rows {
                gpu.copy_d2d_async(
                    bound
                        .embedding
                        .ptr()
                        .offset(tokens[start + t] as usize * 5760),
                    hidden.offset(t * 5760),
                    5760,
                    stream,
                )?;
            }
            for (li, (layer, state)) in layers.iter().zip(states.iter_mut()).enumerate() {
                if width == 0 || start == prefix {
                    layer.forward_token(
                        hidden,
                        state.as_mut(),
                        &mut cache,
                        start,
                        &mut blocks,
                        &gpu,
                        stream,
                    )?;
                } else {
                    layer.forward_chunk(
                        hidden,
                        state.as_mut(),
                        &mut cache,
                        start,
                        rows,
                        &mut blocks,
                        &mut scratch,
                        &gpu,
                        stream,
                    )?;
                }
                let mut bytes = vec![0u8; rows * 5760];
                gpu.copy_d2h_on_stream(hidden, &mut bytes, stream)?;
                for t in 0..rows {
                    trace[((start + t) * 24 + li) * 5760..((start + t) * 24 + li + 1) * 5760]
                        .copy_from_slice(&bytes[t * 5760..(t + 1) * 5760]);
                }
            }
            start += rows;
        }
        let mut kv = Vec::with_capacity(count * 24 * 512 * 4);
        for l in 0..24 {
            for ptr in [cache.k_pool_ptr(l), cache.v_pool_ptr(l)] {
                let mut pool = vec![0u8; pool_bytes];
                gpu.copy_d2h_on_stream(ptr, &mut pool, stream)?;
                for t in 0..count {
                    let slot = blocks[t / 16] as usize * 16 + t % 16;
                    kv.extend_from_slice(&pool[slot * 1024..(slot + 1) * 1024]);
                }
            }
        }
        let (hidden_mismatches, cache_mismatches) = baseline.as_ref().map_or((0, 0), |(h, k)| {
            (
                trace.iter().zip(h).filter(|(a, b)| a != b).count(),
                kv.iter().zip(k).filter(|(a, b)| a != b).count(),
            )
        });
        let nonfinite = trace
            .chunks_exact(2)
            .filter(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
            .count();
        std::fs::write(output.join(format!("width{width}.hidden.bf16")), &trace)?;
        std::fs::write(output.join(format!("width{width}.cache.bf16")), &kv)?;
        cases.push(serde_json::json!({"width":width,"hidden_byte_mismatches":hidden_mismatches,"cache_byte_mismatches":cache_mismatches,"nonfinite":nonfinite,"following_decode_position":prefix}));
        if baseline.is_none() {
            baseline = Some((trace, kv));
        }
        for (l, s) in layers.iter().zip(states.iter_mut()) {
            l.release_state(s.as_mut(), &gpu)?;
        }
        cache.free_blocks(&blocks);
        eprintln!(
            "chunk width={width} hidden_mismatch={hidden_mismatches} cache_mismatch={cache_mismatches}"
        );
    }
    // 2026-10-07: Stream mismatch must refuse before any launch/free.
    ensure!(
        scratch.release(&gpu, stream.wrapping_add(1)).is_err(),
        "cross-stream scratch release admitted"
    );
    let mut control_state = layers[0].alloc_state(&gpu)?;
    let mut control_blocks = vec![];
    ensure!(
        layers[0]
            .forward_chunk(
                hidden,
                control_state.as_mut(),
                &mut cache,
                0,
                1,
                &mut control_blocks,
                &mut scratch,
                &gpu,
                stream.wrapping_add(1)
            )
            .is_err(),
        "cross-stream scratch reuse admitted"
    );
    // 2026-10-07: Exhaust capacity after one successful chunk, then restore capacity.
    // The failed state must remain poisoned despite the repaired allocator condition.
    let mut reserved = Vec::new();
    for _ in 0..pages - if capacity > 16 { 2 } else { 1 } {
        reserved.push(cache.alloc_block()?);
    }
    for t in 0..16 {
        gpu.copy_d2d_async(
            bound.embedding.ptr().offset(tokens[t] as usize * 5760),
            hidden.offset(t * 5760),
            5760,
            stream,
        )?;
    }
    layers[0].forward_chunk(
        hidden,
        control_state.as_mut(),
        &mut cache,
        0,
        16,
        &mut control_blocks,
        &mut scratch,
        &gpu,
        stream,
    )?;
    ensure!(
        layers[0]
            .forward_chunk(
                hidden,
                control_state.as_mut(),
                &mut cache,
                16,
                capacity,
                &mut control_blocks,
                &mut scratch,
                &gpu,
                stream
            )
            .is_err(),
        "exhausted cache admitted"
    );
    ensure!(
        control_blocks.len() == if capacity > 16 { 2 } else { 1 },
        "allocation control did not fail at intended partial-page boundary"
    );
    cache.free_block(reserved.pop().unwrap());
    let error = layers[0]
        .forward_chunk(
            hidden,
            control_state.as_mut(),
            &mut cache,
            16,
            1,
            &mut control_blocks,
            &mut scratch,
            &gpu,
            stream,
        )
        .unwrap_err();
    ensure!(
        error.to_string().contains("state"),
        "failed state was retried: {error}"
    );
    gpu.synchronize(stream)?;
    layers[0].release_state(control_state.as_mut(), &gpu)?;
    cache.free_blocks(&control_blocks);
    cache.free_blocks(&reserved);
    let passed = cases.iter().all(|r| {
        r["hidden_byte_mismatches"] == 0 && r["cache_byte_mismatches"] == 0 && r["nonfinite"] == 0
    });
    let report = serde_json::json!({"scope":"diagnostic single-sequence chunks; no production admission or speed qualification","capacity":capacity,"scratch_bytes":PrefillScratch::required_bytes(capacity,pages)?,"tokens":tokens,"module_manifest":manifest,"future_cache_fill":"BF16 NaN","admission_controls":"empty/oversize/nonsequential/overflow/stream mismatch/released scratch; allocation failure poisons state","cases":cases,"passed":passed});
    scratch.release(&gpu, stream)?;
    let mut released_state = layers[0].alloc_state(&gpu)?;
    ensure!(
        layers[0]
            .forward_chunk(
                hidden,
                released_state.as_mut(),
                &mut cache,
                0,
                1,
                &mut vec![],
                &mut scratch,
                &gpu,
                stream
            )
            .is_err(),
        "released scratch admitted"
    );
    layers[0].release_state(released_state.as_mut(), &gpu)?;
    gpu.free(hidden)?;
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;

    ensure!(passed, "chunk exact gate failed; raw evidence retained");
    Ok(())
}
