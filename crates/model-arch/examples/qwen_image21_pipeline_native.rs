// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native component orchestration; intentionally slow diagnostic.
//! JOB_JSON NEW_OUT. Runs verified native binaries sequentially, never Python.
//! No model registration, numerical admission, or production performance claim.
use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_model_arch::qwen_image21::{
    prompt::TextPromptEncoder,
    scheduler::{Config as SchedulerConfig, Schedule, step_bf16_host},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    model: PathBuf,
    encoder_binary: PathBuf,
    visual_binary: PathBuf,
    vae_binary: PathBuf,
    encoder_modules: PathBuf,
    visual_modules: PathBuf,
    vae_modules: PathBuf,
    prompt: String,
    negative_prompt: String,
    width: u32,
    height: u32,
    steps: usize,
    seed: u64,
    guidance: f32,
}
fn sha(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn write_bf16(path: &Path, v: &[bf16]) -> Result<()> {
    std::fs::write(
        path,
        v.iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
fn read_bf16(path: &Path, count: usize) -> Result<Vec<bf16>> {
    let b = std::fs::read(path)?;
    ensure!(b.len() == count * 2, "BF16 component output size differs");
    let v: Vec<_> = b
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
        .collect();
    ensure!(
        v.iter().all(|v| v.is_finite()),
        "nonfinite component output"
    );
    Ok(v)
}
fn run(binary: &Path, args: &[PathBuf], log: &Path) -> Result<()> {
    let file = std::fs::File::create(log)?;
    let status = Command::new(binary)
        .args(args)
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file))
        .status()
        .with_context(|| format!("launch {}", binary.display()))?;
    ensure!(
        status.success(),
        "native component failed; see {}",
        log.display()
    );
    Ok(())
}
fn encode(
    job: &Job,
    encoder: &TextPromptEncoder,
    prompt: &str,
    out: &Path,
    name: &str,
) -> Result<(Vec<u8>, usize)> {
    let p = encoder.encode(prompt, 4096)?;
    let ids = out.join(format!("{name}-ids.json"));
    std::fs::write(&ids, serde_json::to_vec(&p.input_ids)?)?;
    let folder = out.join(format!("{name}-encoder"));
    run(
        &job.encoder_binary,
        &[
            job.model.clone(),
            job.encoder_modules.clone(),
            ids,
            folder.clone(),
        ],
        &out.join(format!("{name}-encoder.log")),
    )?;
    let hidden = std::fs::read(folder.join("pre-final-norm.bf16"))?;
    ensure!(
        hidden.len() == p.input_ids.len() * 4096 * 2,
        "encoder output length differs"
    );
    let kept = p.input_ids.len() - p.drop_prefix;
    ensure!(kept > 0, "empty prompt embeddings");
    let bytes = hidden[p.drop_prefix * 4096 * 2..].to_vec();
    std::fs::write(out.join(format!("{name}-embedding.bf16")), &bytes)?;
    Ok((bytes, kept))
}
fn predict(
    job: &Job,
    out: &Path,
    name: &str,
    step: usize,
    embedding: &[u8],
    text: usize,
    latent: &[bf16],
    time: f32,
) -> Result<Vec<bf16>> {
    let fixture = out.join(format!("{name}-step-{step:03}-input"));
    std::fs::create_dir(&fixture)?;
    let lh = job.height as usize / 16;
    let lw = job.width as usize / 16;
    let pixels = lh * lw;
    let slots: Vec<bool> = (0..text + pixels / 4).map(|i| i >= text).collect();
    let spec = serde_json::json!({"samples":1,"text_tokens":text,"image_slots":slots,"image_shapes":[[1,lh,lw]],"text_key_valid":vec![true;text],"timesteps":[time]});
    std::fs::write(fixture.join("fixture.json"), serde_json::to_vec(&spec)?)?;
    std::fs::write(fixture.join("text-input.bf16"), embedding)?;
    write_bf16(&fixture.join("image-input.bf16"), latent)?;
    let target = out.join(format!("{name}-step-{step:03}"));
    run(
        &job.visual_binary,
        &[
            job.model.clone(),
            job.visual_modules.clone(),
            fixture,
            target.clone(),
        ],
        &out.join(format!("{name}-step-{step:03}.log")),
    )?;
    read_bf16(&target.join("target-latents.bf16"), pixels * 64)
}
// SplitMix64 + Box-Muller in FP64, cast once to BF16. The exact noise is saved;
// this is a native reproducible fixture RNG, not a claim of Torch seed identity.
fn noise(seed: u64, count: usize) -> Vec<bf16> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / 9007199254740992.0
    };
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let r = (-2.0 * uniform().ln()).sqrt();
        let phase = std::f64::consts::TAU * uniform();
        for x in [r * phase.cos(), r * phase.sin()] {
            if values.len() < count {
                values.push(bf16::from_f32(x as f32));
            }
        }
    }
    values
}
fn guided(positive: &[bf16], negative: &[bf16], scale: f32) -> Result<Vec<bf16>> {
    ensure!(positive.len() == negative.len(), "guidance shapes differ");
    let scale = bf16::from_f32(scale).to_f32();
    Ok(positive
        .iter()
        .zip(negative)
        .map(|(p, n)| {
            let delta = bf16::from_f32(p.to_f32() - n.to_f32());
            let product = bf16::from_f32(scale * delta.to_f32());
            bf16::from_f32(n.to_f32() + product.to_f32())
        })
        .collect())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let job_path = PathBuf::from(args.next().context("job JSON")?);
    let out = PathBuf::from(args.next().context("new output")?);
    let job_bytes = std::fs::read(job_path)?;
    let job: Job = serde_json::from_slice(&job_bytes)?;
    ensure!(
        (32..=128).contains(&job.width)
            && (32..=128).contains(&job.height)
            && job.width.is_multiple_of(32)
            && job.height.is_multiple_of(32),
        "diagnostic image dimensions must be32..128 in32pixel increments"
    );
    ensure!(
        job.guidance.is_finite() && (1.0..=10.0).contains(&job.guidance),
        "guidance outside1..10"
    );
    ensure!(
        (2..=50).contains(&job.steps),
        "diagnostic steps outside2..50"
    );
    std::fs::create_dir(&out)?;
    std::fs::write(out.join("job.json"), &job_bytes)?;
    let mut binary_hashes = serde_json::Map::new();
    for (name, p) in [
        ("encoder", &job.encoder_binary),
        ("visual", &job.visual_binary),
        ("vae", &job.vae_binary),
    ] {
        binary_hashes.insert(name.into(), sha(&std::fs::read(p)?).into());
    }
    let encoder = TextPromptEncoder::from_bytes(
        &std::fs::read(job.model.join("processor/tokenizer.json"))?,
        &std::fs::read(job.model.join("processor/chat_template.jinja"))?,
    )?;
    let positive = encode(&job, &encoder, &job.prompt, &out, "positive")?;
    let negative = if job.guidance > 1.0 {
        Some(encode(
            &job,
            &encoder,
            &job.negative_prompt,
            &out,
            "negative",
        )?)
    } else {
        None
    };
    let pixels = job.width as usize / 16 * (job.height as usize / 16);
    let config = SchedulerConfig::from_value(serde_json::from_slice(&std::fs::read(
        job.model.join("scheduler/scheduler_config.json"),
    )?)?)?;
    let schedule = Schedule::new(&config, job.steps, pixels)?;
    let mut latent = noise(job.seed, pixels * 64);
    write_bf16(&out.join("initial-latents.bf16"), &latent)?;
    let mut steps = Vec::new();
    for i in 0..job.steps {
        let time = schedule.model_timestep(i)?.to_f32();
        let mut prediction = predict(
            &job,
            &out,
            "positive",
            i,
            &positive.0,
            positive.1,
            &latent,
            time,
        )?;
        if let Some((embedding, text)) = &negative {
            let n = predict(&job, &out, "negative", i, embedding, *text, &latent, time)?;
            prediction = guided(&prediction, &n, job.guidance)?;
        }
        write_bf16(&out.join(format!("prediction-{i:03}.bf16")), &prediction)?;
        latent = step_bf16_host(&latent, &prediction, schedule.delta(i)?)?;
        write_bf16(&out.join(format!("latent-{i:03}.bf16")), &latent)?;
        steps.push(serde_json::json!({"index":i,"scheduler_time":schedule.timestep(i)?,"model_time":time,"delta":schedule.delta(i)?}));
    }
    let vae: serde_json::Value =
        serde_json::from_slice(&std::fs::read(job.model.join("vae/config.json"))?)?;
    let mean: Vec<f32> = serde_json::from_value(vae["latents_mean"].clone())?;
    let std: Vec<f32> = serde_json::from_value(vae["latents_std"].clone())?;
    ensure!(
        mean.len() == 64 && std.len() == 64 && mean.iter().chain(&std).all(|v| v.is_finite()),
        "invalid VAE normalization"
    );
    let mut decoded_input = Vec::with_capacity(latent.len() * 4);
    for c in 0..64 {
        for pixel in 0..pixels {
            let value = latent[pixel * 64 + c].to_f32() * std[c] + mean[c];
            decoded_input.extend(value.to_le_bytes());
        }
    }
    let input = out.join("vae-input.f32");
    std::fs::write(&input, &decoded_input)?;
    let decoded = out.join("vae");
    run(
        &job.vae_binary,
        &[
            job.model.clone(),
            job.vae_modules.clone(),
            input,
            decoded.clone(),
            (job.height / 16).to_string().into(),
            (job.width / 16).to_string().into(),
        ],
        &out.join("vae.log"),
    )?;
    let values = std::fs::read(decoded.join("clamped.f32"))?;
    let pixels = job.width as usize * job.height as usize;
    ensure!(values.len() == pixels * 4 * 4, "decoded RGBA size differs");
    let mut rgba = Vec::with_capacity(pixels * 4);
    for i in 0..pixels {
        for c in 0..4 {
            let offset = (c * pixels + i) * 4;
            let x = f32::from_le_bytes(values[offset..offset + 4].try_into().unwrap());
            ensure!(x.is_finite(), "nonfinite decodedpixel");
            rgba.push(((x / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8);
        }
    }
    let mut png = png::Encoder::new(
        std::fs::File::create(out.join("image.png"))?,
        job.width,
        job.height,
    );
    png.set_color(png::ColorType::Rgba);
    png.set_depth(png::BitDepth::Eight);
    png.write_header()?.write_image_data(&rgba)?;
    let receipt = serde_json::json!({"checkpoint_revision":"d26bb61231c349cf6b7896fa83353113880e1ba3","job_sha256":sha(&job_bytes),"binary_sha256":binary_hashes,"steps":steps,"rgba_sha256":sha(&rgba),"scope":"native diagnostic orchestration with component numerical failures retained; not qualified image support or speed","precision":"BF16 encoder/visual/scheduler, original FP32 VAE","rng":"SplitMix64 BoxMuller FP64 to BF16; saved exact initial noise","qualified":false});
    std::fs::write(
        out.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_noise_and_staged_guidance_controls() {
        assert_eq!(noise(7, 33), noise(7, 33));
        assert_ne!(noise(7, 33), noise(8, 33));
        assert!(noise(7, 4096).iter().all(|v| v.is_finite()));
        let p = [bf16::from_f32(2.0)];
        let n = [bf16::from_f32(1.0)];
        assert_eq!(guided(&p, &n, 4.0).unwrap(), [bf16::from_f32(5.0)]);
        assert!(guided(&p, &[], 4.0).is_err());
        // Separate subtraction, scalar BF16 cast, product, then addition.
        // A fused FP32 expression would round to 5.03125 instead.
        assert_eq!(
            guided(&[bf16::from_f32(-4.25)], &[bf16::from_f32(-7.6875)], 3.7).unwrap(),
            [bf16::from_f32(5.0625)]
        );
    }
}
