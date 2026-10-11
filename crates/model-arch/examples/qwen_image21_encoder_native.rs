// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Actual dense native text encoder; external token IDs, no final norm.
//! MODEL MODULES_JSON INPUT_IDS_JSON NEW_OUT [LAYERS=36]. No image/VLM claim.
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::{cuda_backend::MetraleCudaBackend, gpu::GpuBackend};
use metrale_model_arch::qwen_image21::encoder::{
    DiagnosticTextBlock, TextBlockWeights, text_rope_coefficients,
};
use metrale_model_layers::layers::ops;
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
mod qwen_support;
use qwen_support::{Modules, Owned, read_tensor, sha};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("model")?);
    let manifest = PathBuf::from(args.next().context("modules")?);
    let ids_path = PathBuf::from(args.next().context("input IDs")?);
    let out = PathBuf::from(args.next().context("new output")?);
    let layers: usize = args.next().unwrap_or_else(|| "36".into()).parse()?;
    ensure!((1..=36).contains(&layers), "invalid layer count");
    let ids: Vec<u32> = serde_json::from_slice(&std::fs::read(&ids_path)?)?;
    ensure!(
        !ids.is_empty()
            && ids.len() <= 4096
            && ids
                .iter()
                .all(|id| *id < 151936 && !(151652..=151656).contains(id)),
        "invalid encoder IDs"
    );
    std::fs::create_dir(&out)?;
    let source = std::fs::read(&manifest)?;
    let modules: Modules = serde_json::from_slice(&source)?;
    let mut ptx = Vec::new();
    let required = [
        "image_modulation",
        "dense_gemm_bf16",
        "embed_from_argmax",
        "rms_norm_vanilla",
        "attn_prefill_h128",
        "residual_add",
    ];
    ensure!(
        modules.modules.len() == required.len(),
        "module count differs"
    );
    for name in required {
        let m = modules
            .modules
            .iter()
            .find(|m| m.name == name)
            .context("missing module")?;
        ensure!(
            Path::new(&m.ptx).components().count() == 1 && !Path::new(&m.ptx).is_absolute(),
            "unsafe PTX path"
        );
        let bytes = std::fs::read(manifest.parent().context("manifest parent")?.join(&m.ptx))?;
        ensure!(sha(&bytes) == m.ptx_sha256, "PTX SHA differs");
        ptx.push((name, &*Box::leak(bytes.into_boxed_slice())));
    }
    let gpu = MetraleCudaBackend::new(0, &ptx)?;
    let stream = gpu.default_stream();
    let total = gpu.total_memory()?;
    let used = total.saturating_sub(gpu.device_free_memory()?);
    ensure!(
        used.saturating_add(3 << 30) <= total / 100 * 85,
        "encoder memory would exceed85percent"
    );
    let component = model.join("text_encoder");
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(component.join("config.json"))?)?;
    let c = &config["text_config"];
    ensure!(
        config["model_type"] == "qwen3_vl"
            && c["hidden_size"] == 4096
            && c["num_hidden_layers"] == 36
            && c["num_attention_heads"] == 32
            && c["num_key_value_heads"] == 8
            && c["intermediate_size"] == 12288
            && c["rope_theta"] == 5000000
            && c["rms_norm_eps"] == 1e-6
            && c["head_dim"] == 128
            && c["hidden_act"] == "silu"
            && c["attention_bias"] == false,
        "encoder config differs"
    );
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        component.join("model.safetensors.index.json"),
    )?)?;
    let index: BTreeMap<String, String> = serde_json::from_value(index["weight_map"].clone())?;
    let (embedding, _) = read_tensor(
        &component,
        &index,
        "model.language_model.embed_tokens.weight",
        &[151936, 4096],
    )?;
    let embedding_hash = sha(&embedding);
    let embedding_gpu = Owned::upload(&gpu, &embedding)?;
    drop(embedding);
    let tokens = ids.len() as u32;
    let ids_gpu = Owned::upload(
        &gpu,
        &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let mut hidden = vec![0u8; ids.len() * 8192];
    let input = Owned::upload(&gpu, &hidden)?;
    ops::batched_embed(
        &gpu,
        gpu.kernel("embed_from_argmax", "batched_embed")?,
        ids_gpu.ptr,
        embedding_gpu.ptr,
        input.ptr,
        tokens,
        4096,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.copy_d2h(input.ptr, &mut hidden)?;
    std::fs::write(out.join("embedding.bf16"), &hidden)?;
    drop(embedding_gpu);
    let (cos, sin) = text_rope_coefficients(tokens)?;
    let cos: Vec<u8> = cos.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    let sin: Vec<u8> = sin.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    std::fs::write(out.join("cos.bf16"), &cos)?;
    std::fs::write(out.join("sin.bf16"), &sin)?;
    let cos = Owned::upload(&gpu, &cos)?;
    let sin = Owned::upload(&gpu, &sin)?;
    let suffixes = [
        ("input_layernorm", vec![4096]),
        ("self_attn.q_proj", vec![4096, 4096]),
        ("self_attn.k_proj", vec![1024, 4096]),
        ("self_attn.v_proj", vec![1024, 4096]),
        ("self_attn.o_proj", vec![4096, 4096]),
        ("self_attn.q_norm", vec![128]),
        ("self_attn.k_norm", vec![128]),
        ("post_attention_layernorm", vec![4096]),
        ("mlp.gate_proj", vec![12288, 4096]),
        ("mlp.up_proj", vec![12288, 4096]),
        ("mlp.down_proj", vec![4096, 12288]),
    ];
    let mut results = Vec::new();
    for layer in 0..layers {
        let mut buffers = Vec::new();
        let mut tensors = Vec::new();
        let mut hashes = BTreeMap::new();
        for (suffix, shape) in &suffixes {
            let name = format!("model.language_model.layers.{layer}.{suffix}.weight");
            let (bytes, shape) = read_tensor(&component, &index, &name, shape)?;
            hashes.insert(name, sha(&bytes));
            let buffer = Owned::upload(&gpu, &bytes)?;
            tensors.push(WeightTensor {
                ptr: buffer.ptr,
                shape,
                dtype: WeightDtype::BF16,
            });
            buffers.push(buffer);
        }
        let mut block = DiagnosticTextBlock::new(
            &gpu,
            tokens,
            TextBlockWeights(std::array::from_fn(|i| &tensors[i])),
        )?;
        let result = block.forward(input.ptr, cos.ptr, sin.ptr, stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(result, &mut hidden)?;
        ensure!(
            hidden
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
            "nonfinite encoder output"
        );
        std::fs::write(out.join(format!("layer-{layer:02}.bf16")), &hidden)?;
        results.push(
            serde_json::json!({"layer":layer,"weight_sha256":hashes,"output_sha256":sha(&hidden)}),
        );
        gpu.copy_h2d(&hidden, input.ptr)?;
    }
    std::fs::write(out.join("pre-final-norm.bf16"), &hidden)?;
    let receipt = serde_json::json!({"checkpoint_revision":"d26bb61231c349cf6b7896fa83353113880e1ba3","module_manifest_sha256":sha(&source),"embedding_weight_sha256":embedding_hash,"input_ids":ids,"results":results,"scope":"dense text-only encoder diagnostic, no final norm, no vision/tokenizer/pipeline qualification","qualified":false});
    std::fs::write(
        out.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
