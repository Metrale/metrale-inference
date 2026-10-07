// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Actual native Rust visual-block/stack diagnostic, not image serving.
//! MODEL MODULES_JSON FIXTURE_DIR NEW_OUT [LAYERS=1]. Uses fixed saved 2x3 inputs.
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::{cuda_backend::MetraleCudaBackend, gpu::GpuBackend};
use metrale_model_arch::qwen_image21::{block::DiagnosticImageBlock, rope::ImageRopeLayout};
use metrale_model_weights::{
    qwen_image21::{Block, Config},
    weights::{WeightDtype, WeightTensor},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
mod qwen_support;
use qwen_support::{Modules, Owned, read_tensor, sha};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("model directory")?);
    let manifest = PathBuf::from(args.next().context("modules JSON")?);
    let fixture = PathBuf::from(args.next().context("fixture directory")?);
    let out = PathBuf::from(args.next().context("new output directory")?);
    let layers: usize = args.next().unwrap_or_else(|| "1".into()).parse()?;
    ensure!((1..=32).contains(&layers), "layers must be 1..32");
    std::fs::create_dir(&out)?;
    let source = std::fs::read(&manifest)?;
    let modules: Modules = serde_json::from_slice(&source)?;
    let required = [
        "nllb_encoder",
        "image_modulation",
        "dense_gemm_bf16",
        "embed_from_argmax",
        "rms_norm_vanilla",
    ];
    ensure!(
        modules.modules.len() == required.len(),
        "module count differs"
    );
    let mut ptx = Vec::new();
    for expected in required {
        let m = modules
            .modules
            .iter()
            .find(|m| m.name == expected)
            .context("missing native module")?;
        ensure!(
            Path::new(&m.ptx).components().count() == 1 && !Path::new(&m.ptx).is_absolute(),
            "unsafe PTX path"
        );
        let bytes = std::fs::read(manifest.parent().context("module parent")?.join(&m.ptx))?;
        ensure!(sha(&bytes) == m.ptx_sha256, "native PTX SHA differs");
        ptx.push((expected, &*Box::leak(bytes.into_boxed_slice())));
    }
    let gpu = MetraleCudaBackend::new(0, &ptx)?;
    let stream = gpu.default_stream();
    // One layer is loaded at a time, below 512 MiB; reserve 1 GiB for all buffers.
    let total = gpu.total_memory()?;
    let used = total.saturating_sub(gpu.device_free_memory()?);
    ensure!(
        used.saturating_add(1 << 30) <= total / 100 * 85,
        "GPU memory would exceed 85 percent"
    );
    let component = model.join("transformer");
    let config = Config::parse(&std::fs::read_to_string(component.join("config.json"))?)?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        component.join("diffusion_pytorch_model.safetensors.index.json"),
    )?)?;
    let index: BTreeMap<String, String> = serde_json::from_value(index["weight_map"].clone())?;
    let mut hidden = std::fs::read(fixture.join("input.bf16"))?;
    let modulation = std::fs::read(fixture.join("modulation.bf16"))?;
    ensure!(
        hidden.len() == 2 * 3 * 4096 * 2 && modulation.len() == 3 * 16384 * 2,
        "fixture shape differs"
    );
    let input_hash = sha(&hidden);
    let modulation_hash = sha(&modulation);
    let input = Owned::upload(&gpu, &hidden)?;
    let mod_input = Owned::upload(&gpu, &modulation)?;
    let rope = ImageRopeLayout::new(&[false, true, false], &[[1, 1, 1]])?;
    let suffixes = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.norm_q",
        "attn.norm_k",
        "img_mlp.gate_layer",
        "img_mlp.proj",
        "img_mlp.out",
    ];
    let mut results = Vec::new();
    for layer in 0..layers {
        let mut buffers = Vec::new();
        let mut tensors = Vec::new();
        let mut hashes = BTreeMap::new();
        for suffix in suffixes {
            let name = format!("transformer_blocks.{layer}.{suffix}.weight");
            let expected_shape: &[usize] = match suffix {
                "attn.norm_q" | "attn.norm_k" => &[128],
                "img_mlp.gate_layer" | "img_mlp.proj" => &[12288, 4096],
                "img_mlp.out" => &[4096, 12288],
                _ => &[4096, 4096],
            };
            let (bytes, shape) = read_tensor(&component, &index, &name, expected_shape)?;
            hashes.insert(name, sha(&bytes));
            let buffer = Owned::upload(&gpu, &bytes)?;
            tensors.push(WeightTensor {
                ptr: buffer.ptr,
                shape,
                dtype: WeightDtype::BF16,
            });
            buffers.push(buffer);
        }
        let weights = Block {
            q: &tensors[0],
            k: &tensors[1],
            v: &tensors[2],
            o: &tensors[3],
            q_norm: &tensors[4],
            k_norm: &tensors[5],
            gate: &tensors[6],
            up: &tensors[7],
            down: &tensors[8],
        };
        let mut block = DiagnosticImageBlock::new(
            &config,
            &weights,
            &gpu,
            2,
            &[-1, 0, -1],
            &[true; 6],
            Some(&[false, true, true]),
        )?;
        let started = std::time::Instant::now();
        let output = block.forward(input.ptr, mod_input.ptr, &rope, stream)?;
        gpu.synchronize(stream)?;
        let elapsed = started.elapsed().as_secs_f64();
        gpu.copy_d2h(output, &mut hidden)?;
        ensure!(
            hidden
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
            "nonfinite block output"
        );
        std::fs::write(out.join(format!("layer-{layer:02}.bf16")), &hidden)?;
        results.push(serde_json::json!({"layer":layer,"weight_sha256":hashes,"output_sha256":sha(&hidden),"synchronized_forward_seconds":elapsed}));
        gpu.copy_h2d(&hidden, input.ptr)?;
    }
    let receipt = serde_json::json!({"checkpoint_revision":"d26bb61231c349cf6b7896fa83353113880e1ba3","module_manifest_sha256":sha(&source),"input_sha256":input_hash,"modulation_sha256":modulation_hash,"results":results,"scope":"native Rust diagnostic visual blocks only; no encoder/VAE/denoising or qualification","qualified":false});
    std::fs::write(
        out.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
