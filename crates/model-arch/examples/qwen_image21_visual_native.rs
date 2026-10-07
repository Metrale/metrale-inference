// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Actual native Rust visual-block/stack diagnostic, not image serving.
//! MODEL MODULES_JSON FIXTURE_DIR NEW_OUT [LAYERS=32]. External encoder fixture.
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::{cuda_backend::MetraleCudaBackend, gpu::GpuBackend};
use metrale_model_arch::qwen_image21::{
    block::DiagnosticImageBlock,
    conditioning::{ConditioningWeights, DiagnosticConditioning},
    io::{DiagnosticVisualIo, ProjectionWeights},
    layout::JointLayout,
};
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
    let layers: usize = args.next().unwrap_or_else(|| "32".into()).parse()?;
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
        "rms_norm",
        "gelu",
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
        used.saturating_add(2 << 30) <= total / 100 * 85,
        "GPU memory would exceed 85 percent"
    );
    let component = model.join("transformer");
    let config = Config::parse(&std::fs::read_to_string(component.join("config.json"))?)?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        component.join("diffusion_pytorch_model.safetensors.index.json"),
    )?)?;
    let index: BTreeMap<String, String> = serde_json::from_value(index["weight_map"].clone())?;
    let layout = JointLayout::new(
        2,
        3,
        &[false, true, false, true],
        &[[1, 2, 2], [1, 2, 2]],
        &[true, true, false, false, true, true],
    )?;
    let mut global_buffers = Vec::new();
    let mut global_tensors = Vec::new();
    let mut global_hashes = BTreeMap::new();
    for (name, shape) in [
        (
            "time_text_embed.timestep_embedder.linear_1",
            vec![4096, 256],
        ),
        (
            "time_text_embed.timestep_embedder.linear_2",
            vec![4096, 4096],
        ),
        ("modulation.1", vec![16384, 4096]),
        ("norm_out.linear", vec![4096, 4096]),
        ("img_in", vec![4096, 64]),
        ("txt_in.text_norm", vec![4096]),
        ("txt_in.in_layer", vec![4096, 4096]),
        ("txt_in.out_layer", vec![4096, 4096]),
        ("proj_out", vec![64, 4096]),
    ] {
        let name = format!("{name}.weight");
        let (bytes, shape) = read_tensor(&component, &index, &name, &shape)?;
        global_hashes.insert(name, sha(&bytes));
        let buffer = Owned::upload(&gpu, &bytes)?;
        global_tensors.push(WeightTensor {
            ptr: buffer.ptr,
            shape,
            dtype: WeightDtype::BF16,
        });
        global_buffers.push(buffer);
    }
    let mut conditioning = DiagnosticConditioning::new(
        &gpu,
        2,
        ConditioningWeights {
            time_in: &global_tensors[0],
            time_out: &global_tensors[1],
            modulation: &global_tensors[2],
            final_scale: &global_tensors[3],
        },
    )?;
    let mut io = DiagnosticVisualIo::new(
        &gpu,
        &layout,
        ProjectionWeights {
            image_in: &global_tensors[4],
            text_norm: &global_tensors[5],
            text_in: &global_tensors[6],
            text_out: &global_tensors[7],
            image_out: &global_tensors[8],
        },
    )?;
    let image = std::fs::read(fixture.join("image-input.bf16"))?;
    let text = std::fs::read(fixture.join("text-input.bf16"))?;
    ensure!(
        image.len() == 2 * 8 * 64 * 2 && text.len() == 2 * 3 * 4096 * 2,
        "IO fixture shape differs"
    );
    let fixture_hashes = serde_json::json!({"image":sha(&image),"text":sha(&text)});
    let image = Owned::upload(&gpu, &image)?;
    let text = Owned::upload(&gpu, &text)?;
    let conditioning_output = conditioning.forward(&[0.731, 0.019], stream)?;
    let initial = io.input_project(image.ptr, text.ptr, stream)?;
    gpu.synchronize(stream)?;
    let mut hidden = vec![0u8; 2 * 10 * 4096 * 2];
    gpu.copy_d2h(initial, &mut hidden)?;
    std::fs::write(out.join("joint-input.bf16"), &hidden)?;
    let input = Owned::upload(&gpu, &hidden)?;
    let input_hash = sha(&hidden);
    let mut modulation = vec![0u8; 3 * 16384 * 2];
    gpu.copy_d2h(conditioning_output.modulation, &mut modulation)?;
    std::fs::write(out.join("modulation.bf16"), &modulation)?;
    let modulation_hash = sha(&modulation);
    let mut final_scale = vec![0u8; 3 * 4096 * 2];
    gpu.copy_d2h(conditioning_output.final_scale, &mut final_scale)?;
    std::fs::write(out.join("final-scale.bf16"), &final_scale)?;
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
            layout.image_ids(),
            layout.key_valid(),
            Some(layout.target_mask()),
        )?;
        let started = std::time::Instant::now();
        let output = block.forward(
            input.ptr,
            conditioning_output.modulation,
            layout.rope(),
            stream,
        )?;
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
    let target = io.output_project(input.ptr, conditioning_output.final_scale, stream)?;
    gpu.synchronize(stream)?;
    let mut target_bytes = vec![0u8; 2 * 4 * 64 * 2];
    gpu.copy_d2h(target, &mut target_bytes)?;
    ensure!(
        target_bytes
            .chunks_exact(2)
            .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
        "nonfinite target output"
    );
    std::fs::write(out.join("target-latents.bf16"), &target_bytes)?;
    let receipt = serde_json::json!({"checkpoint_revision":"d26bb61231c349cf6b7896fa83353113880e1ba3","module_manifest_sha256":sha(&source),"input_sha256":input_hash,"modulation_sha256":modulation_hash,"results":results,"global_weight_sha256":global_hashes,"fixture_sha256":fixture_hashes,"target_sha256":sha(&target_bytes),"scope":"native Rust visual transformer with external encoder inputs; no encoder/VAE/denoising or qualification","qualified":false});
    std::fs::write(
        out.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
