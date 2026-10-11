// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Single-frame original-FP32 decoder diagnostic, all native kernels.
//! MODEL MODULES INPUT_F32 NEW_OUT H W. Input is de-normalized VAE latent NCHW.
//! Slow host synchronization/copies deliberately preserve inspectable stages.
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::{
    cuda_backend::MetraleCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use metrale_model_arch::qwen_image21::{
    vae::{DiagnosticVaeConv, ImageShape, channel_norm_f32, silu_f32},
    vae_attention::attention_f32,
    vae_layout::{upsample_f32, upsample_shape},
    vae_residual::{DiagnosticVaeResidual, ResidualWeights, add_f32},
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};
mod qwen_support;
use qwen_support::{Modules, Owned, read_tensor_f32, sha};
struct Image<'a> {
    data: Owned<'a>,
    shape: ImageShape,
}
struct Weights<'a> {
    tensors: BTreeMap<String, WeightTensor>,
    _owned: Vec<Owned<'a>>,
    hashes: BTreeMap<String, String>,
}
impl Weights<'_> {
    fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing VAE weight {name}"))
    }
}
fn bytes(gpu: &dyn GpuBackend, p: DevicePtr, shape: ImageShape) -> Result<Vec<u8>> {
    gpu.synchronize(gpu.default_stream())?;
    let mut data = vec![0; shape.elements() as usize * 4];
    gpu.copy_d2h(p, &mut data)?;
    ensure!(
        data.chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        "nonfinite native VAE stage"
    );
    Ok(data)
}
fn retain<'a>(gpu: &'a dyn GpuBackend, p: DevicePtr, shape: ImageShape) -> Result<Image<'a>> {
    Ok(Image {
        data: Owned::upload(gpu, &bytes(gpu, p, shape)?)?,
        shape,
    })
}
fn zeros<'a>(gpu: &'a dyn GpuBackend, shape: ImageShape) -> Result<Image<'a>> {
    Ok(Image {
        data: Owned::upload(gpu, &vec![0; shape.elements() as usize * 4])?,
        shape,
    })
}
fn conv<'a>(
    gpu: &'a dyn GpuBackend,
    w: &Weights<'_>,
    name: &str,
    x: &Image<'_>,
) -> Result<Image<'a>> {
    let mut c = DiagnosticVaeConv::new(
        gpu,
        x.shape,
        w.get(&format!("{name}.weight"))?,
        w.get(&format!("{name}.bias"))?,
    )?;
    let p = c.forward(x.data.ptr, gpu.default_stream())?;
    retain(gpu, p, c.output_shape())
}
fn norm<'a>(
    gpu: &'a dyn GpuBackend,
    w: &Weights<'_>,
    name: &str,
    x: &Image<'_>,
) -> Result<Image<'a>> {
    let y = zeros(gpu, x.shape)?;
    channel_norm_f32(
        gpu,
        x.shape,
        x.data.ptr,
        w.get(&format!("{name}.gamma"))?,
        y.data.ptr,
        gpu.default_stream(),
    )?;
    Ok(y)
}
fn residual<'a>(
    gpu: &'a dyn GpuBackend,
    w: &Weights<'_>,
    name: &str,
    x: &Image<'_>,
) -> Result<Image<'a>> {
    let get = |suffix: &str| w.get(&format!("{name}.{suffix}"));
    let shortcut = if w
        .tensors
        .contains_key(&format!("{name}.conv_shortcut.weight"))
    {
        Some((get("conv_shortcut.weight")?, get("conv_shortcut.bias")?))
    } else {
        None
    };
    let mut block = DiagnosticVaeResidual::new(
        gpu,
        x.shape,
        ResidualWeights {
            norm1: get("norm1.gamma")?,
            conv1: get("conv1.weight")?,
            bias1: get("conv1.bias")?,
            norm2: get("norm2.gamma")?,
            conv2: get("conv2.weight")?,
            bias2: get("conv2.bias")?,
            shortcut,
        },
    )?;
    let p = block.forward(x.data.ptr, gpu.default_stream())?;
    retain(gpu, p, block.output_shape())
}
fn upsample<'a>(
    gpu: &'a dyn GpuBackend,
    x: &Image<'_>,
    channels: u32,
    ft: u32,
) -> Result<Image<'a>> {
    let shape = upsample_shape(x.shape, channels, ft)?;
    let y = zeros(gpu, shape)?;
    upsample_f32(
        gpu,
        x.shape,
        channels,
        ft,
        x.data.ptr,
        y.data.ptr,
        gpu.default_stream(),
    )?;
    Ok(y)
}
fn save(
    gpu: &dyn GpuBackend,
    x: &Image<'_>,
    out: &Path,
    name: &str,
    stages: &mut Vec<serde_json::Value>,
) -> Result<()> {
    let data = bytes(gpu, x.data.ptr, x.shape)?;
    std::fs::write(out.join(format!("{name}.f32")), &data)?;
    stages.push(serde_json::json!({"name":name,"shape":x.shape.dimensions(),"sha256":sha(&data),"all_finite":true}));
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("model")?);
    let manifest = PathBuf::from(args.next().context("modules")?);
    let input = PathBuf::from(args.next().context("input")?);
    let out = PathBuf::from(args.next().context("new output")?);
    let h: u32 = args.next().context("height")?.parse()?;
    let width: u32 = args.next().context("width")?.parse()?;
    ensure!(
        (1..=32).contains(&h) && (1..=32).contains(&width),
        "diagnostic latent shape outside1..32"
    );
    let shape = ImageShape::new(64, h, width)?;
    let data = std::fs::read(&input)?;
    ensure!(
        data.len() == shape.elements() as usize * 4,
        "latent byte size differs"
    );
    ensure!(
        data.chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        "nonfinite latent"
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model.join("vae/config.json"))?)?;
    ensure!(
        config["_class_name"] == "AutoencoderKLQwenImage21"
            && config["decoder_base_dim"] == 144
            && config["z_dim"] == 64
            && config["dim_mult"] == serde_json::json!([1, 2, 4, 8, 8])
            && config["num_res_blocks"] == 2
            && config["out_channels"] == 4
            && config["is_residual"] == true
            && config["dropout"] == 0.0
            && config["temperal_downsample"] == serde_json::json!([false, true, true, true]),
        "VAE config differs"
    );
    std::fs::create_dir(&out)?;
    let manifest_bytes = std::fs::read(&manifest)?;
    let modules: Modules = serde_json::from_slice(&manifest_bytes)?;
    let mut ptx = Vec::new();
    ensure!(modules.modules.len() == 3, "module count differs");
    for name in ["image_vae", "image_vae_attention", "nllb_encoder"] {
        let m = modules
            .modules
            .iter()
            .find(|m| m.name == name)
            .context("missing module")?;
        ensure!(
            Path::new(&m.ptx).components().count() == 1 && !Path::new(&m.ptx).is_absolute(),
            "unsafe PTX path"
        );
        let b = std::fs::read(manifest.parent().context("manifest parent")?.join(&m.ptx))?;
        ensure!(sha(&b) == m.ptx_sha256, "PTX SHA differs");
        ptx.push((name, &*Box::leak(b.into_boxed_slice())));
    }
    let gpu = MetraleCudaBackend::new(0, &ptx)?;
    let stream = gpu.default_stream();
    let total = gpu.total_memory()?;
    ensure!(
        total
            .saturating_sub(gpu.device_free_memory()?)
            .saturating_add(16 << 30)
            <= total / 100 * 85,
        "VAE memory exceeds85percent guard"
    );
    let component = model.join("vae");
    let file = "diffusion_pytorch_model.safetensors";
    let mut f = std::fs::File::open(component.join(file))?;
    let mut prefix = [0; 8];
    f.read_exact(&mut prefix)?;
    let n = u64::from_le_bytes(prefix);
    ensure!(n <= 16 * 1024 * 1024, "header too large");
    let mut header = vec![0; n as usize];
    f.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    let mut w = Weights {
        tensors: BTreeMap::new(),
        _owned: Vec::new(),
        hashes: BTreeMap::new(),
    };
    let mut loaded_bytes = 0usize;
    let used_bytes = total.saturating_sub(gpu.device_free_memory()?);
    for (name, t) in header.as_object().context("header object")? {
        if !(name.starts_with("decoder.") || name.starts_with("post_quant_conv.")) {
            continue;
        }
        let dimensions: Vec<usize> = serde_json::from_value(t["shape"].clone())?;
        let tensor_bytes = dimensions
            .iter()
            .try_fold(4usize, |n, d| n.checked_mul(*d))
            .context("VAE tensor byte overflow")?;
        loaded_bytes = loaded_bytes
            .checked_add(tensor_bytes)
            .context("VAE weight size overflow")?;
        ensure!(
            used_bytes
                .saturating_add(loaded_bytes)
                .saturating_add(2 << 30)
                <= total / 100 * 85,
            "VAE actual weights exceed memory budget"
        );
        let index = BTreeMap::from([(name.clone(), file.to_string())]);
        let (b, dimensions) = read_tensor_f32(&component, &index, name, &dimensions)?;
        w.hashes.insert(name.clone(), sha(&b));
        let owned = Owned::upload(&gpu, &b)?;
        w.tensors.insert(
            name.clone(),
            WeightTensor {
                ptr: owned.ptr,
                shape: dimensions,
                dtype: WeightDtype::FP32,
            },
        );
        w._owned.push(owned);
    }
    let mut stages = Vec::new();
    let mut x = Image {
        data: Owned::upload(&gpu, &data)?,
        shape,
    };
    x = conv(&gpu, &w, "post_quant_conv", &x)?;
    save(&gpu, &x, &out, "post_quant", &mut stages)?;
    x = conv(&gpu, &w, "decoder.conv_in", &x)?;
    save(&gpu, &x, &out, "conv_in", &mut stages)?;
    x = residual(&gpu, &w, "decoder.mid_block.resnets.0", &x)?;
    save(&gpu, &x, &out, "mid_res0", &mut stages)?;
    let a = norm(&gpu, &w, "decoder.mid_block.attentions.0.norm", &x)?;
    let qkv = conv(&gpu, &w, "decoder.mid_block.attentions.0.to_qkv", &a)?;
    let a = zeros(&gpu, x.shape)?;
    attention_f32(&gpu, qkv.data.ptr, a.data.ptr, 1152, h * width, stream)?;
    let a = conv(&gpu, &w, "decoder.mid_block.attentions.0.proj", &a)?;
    add_f32(&gpu, x.shape, a.data.ptr, x.data.ptr, stream)?;
    x = a;
    save(&gpu, &x, &out, "mid_attn", &mut stages)?;
    x = residual(&gpu, &w, "decoder.mid_block.resnets.1", &x)?;
    save(&gpu, &x, &out, "mid_res1", &mut stages)?;
    for block in 0..5 {
        let original = retain(&gpu, x.data.ptr, x.shape)?;
        for layer in 0..3 {
            x = residual(
                &gpu,
                &w,
                &format!("decoder.up_blocks.{block}.resnets.{layer}"),
                &x,
            )?;
        }
        if block < 4 {
            x = upsample(&gpu, &x, x.shape.dimensions()[0], 1)?;
            x = conv(
                &gpu,
                &w,
                &format!("decoder.up_blocks.{block}.upsampler.resample.1"),
                &x,
            )?;
            let shortcut = upsample(
                &gpu,
                &original,
                x.shape.dimensions()[0],
                if block < 3 { 2 } else { 1 },
            )?;
            add_f32(&gpu, x.shape, x.data.ptr, shortcut.data.ptr, stream)?;
            gpu.synchronize(stream)?;
        }
        save(&gpu, &x, &out, &format!("up{block}"), &mut stages)?;
    }
    x = norm(&gpu, &w, "decoder.norm_out", &x)?;
    silu_f32(&gpu, x.shape, x.data.ptr, x.data.ptr, stream)?;
    x = conv(&gpu, &w, "decoder.conv_out", &x)?;
    save(&gpu, &x, &out, "decoded", &mut stages)?;
    let decoded = bytes(&gpu, x.data.ptr, x.shape)?;
    let clamped: Vec<u8> = decoded
        .chunks_exact(4)
        .flat_map(|b| {
            f32::from_le_bytes(b.try_into().unwrap())
                .clamp(-1., 1.)
                .to_le_bytes()
        })
        .collect();
    std::fs::write(out.join("clamped.f32"), &clamped)?;
    let receipt = serde_json::json!({"scope":"native single-frame FP32 VAE decoder; diagnostic, not qualified image generation","precision":"original checkpoint FP32","input_sha256":sha(&data),"modules_sha256":sha(&manifest_bytes),"weight_sha256":w.hashes,"stages":stages,"clamped_sha256":sha(&clamped)});
    std::fs::write(
        out.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{}", receipt);
    Ok(())
}
