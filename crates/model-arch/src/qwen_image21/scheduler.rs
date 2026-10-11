// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Explicit deterministic flow schedule and staged BF16 Euler update.
//! Pure host logic; callers own device transfers. No stochastic/custom schedule fallback.
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    _diffusers_version: String,
    base_image_seq_len: usize,
    max_image_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
    num_train_timesteps: u32,
    shift: f64,
    shift_terminal: f32,
    time_shift_type: String,
    use_dynamic_shifting: bool,
    invert_sigmas: bool,
    stochastic_sampling: bool,
    use_beta_sigmas: bool,
    use_exponential_sigmas: bool,
    use_karras_sigmas: bool,
}
impl Config {
    pub fn from_value(value: serde_json::Value) -> Result<Self> {
        let config: Self = serde_json::from_value(value)?;
        ensure!(
            config.class_name == "FlowMatchEulerDiscreteScheduler",
            "wrong scheduler class"
        );
        ensure!(
            config.use_dynamic_shifting && config.time_shift_type == "exponential",
            "only dynamic exponential shifting is implemented"
        );
        ensure!(
            !config.invert_sigmas
                && !config.stochastic_sampling
                && !config.use_beta_sigmas
                && !config.use_exponential_sigmas
                && !config.use_karras_sigmas,
            "unsupported scheduler policy"
        );
        ensure!(
            config.base_image_seq_len > 0 && config.max_image_seq_len > config.base_image_seq_len,
            "invalid image sequence range"
        );
        ensure!(
            config.base_shift.is_finite()
                && config.max_shift.is_finite()
                && config.shift.is_finite()
                && config.shift > 0.0
                && config.shift_terminal > 0.0
                && config.shift_terminal < 1.0
                && config.num_train_timesteps == 1000,
            "unsupported scheduler parameters"
        );
        Ok(config)
    }
}

pub struct Schedule {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}
impl Schedule {
    pub fn new(config: &Config, steps: usize, image_tokens: usize) -> Result<Self> {
        ensure!(
            (2..=10_000).contains(&steps),
            "steps must be in 2..=10000; one-step stretch is undefined"
        );
        ensure!(image_tokens > 0, "empty image sequence");
        let slope = (config.max_shift - config.base_shift)
            / (config.max_image_seq_len - config.base_image_seq_len) as f64;
        let mu = image_tokens as f64 * slope
            + (config.base_shift - slope * config.base_image_seq_len as f64);
        let shift = mu.exp() as f32;
        ensure!(
            shift.is_finite() && shift > 0.0,
            "unrepresentable dynamic shift"
        );
        // 2026-10-06: NumPy constructs linspace in FP64, casts to FP32, then stages
        // each array operation in FP32 under its scalar-promotion rules.
        let end = 1.0 / steps as f64;
        let stride = (end - 1.0) / (steps - 1) as f64;
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| {
                let t = if i + 1 == steps {
                    end
                } else {
                    1.0 + i as f64 * stride
                } as f32;
                shift / (shift + (1.0 / t - 1.0))
            })
            .collect();
        let scale = (1.0 - sigmas[steps - 1]) / (1.0 - config.shift_terminal);
        ensure!(scale.is_finite() && scale > 0.0, "invalid terminal stretch");
        for sigma in &mut sigmas {
            *sigma = 1.0 - (1.0 - *sigma) / scale;
        }
        ensure!(
            sigmas
                .iter()
                .all(|s| s.is_finite() && *s > 0.0 && *s <= 1.0)
                && sigmas.windows(2).all(|s| s[1] < s[0]),
            "non-decreasing or invalid schedule"
        );
        let timesteps = sigmas
            .iter()
            .map(|s| s * config.num_train_timesteps as f32)
            .collect();
        sigmas.push(0.0);
        Ok(Self { sigmas, timesteps })
    }
    pub fn len(&self) -> usize {
        self.timesteps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    pub fn timestep(&self, index: usize) -> Result<f32> {
        self.timesteps
            .get(index)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("step index out of range"))
    }
    pub fn model_timestep(&self, index: usize) -> Result<bf16> {
        Ok(bf16::from_f32(
            bf16::from_f32(self.timestep(index)?).to_f32() / 1000.0,
        ))
    }
    pub fn delta(&self, index: usize) -> Result<f32> {
        ensure!(index < self.len(), "step index out of range");
        Ok(self.sigmas[index + 1] - self.sigmas[index])
    }
}

/// 2026-10-06: Pinned Torch scalar promotion first casts the zero-dimensional
/// FP32 delta to BF16, multiplies and rounds to BF16, then adds to FP32 sample.
/// This host implementation is a correctness path, not GPU throughput evidence.
pub fn step_bf16_host(sample: &[bf16], prediction: &[bf16], delta: f32) -> Result<Vec<bf16>> {
    ensure!(
        !sample.is_empty() && sample.len() == prediction.len(),
        "invalid latent shapes"
    );
    ensure!(
        delta.is_finite() && (-1.0..0.0).contains(&delta),
        "invalid denoising delta"
    );
    ensure!(
        sample.iter().chain(prediction).all(|x| x.is_finite()),
        "nonfinite latent or prediction"
    );
    let scalar = bf16::from_f32(delta).to_f32();
    let result: Vec<_> = sample
        .iter()
        .zip(prediction)
        .map(|(s, p)| {
            let product = bf16::from_f32(scalar * p.to_f32());
            bf16::from_f32(s.to_f32() + product.to_f32())
        })
        .collect();
    ensure!(
        result.iter().all(|x| x.is_finite()),
        "nonfinite denoising result"
    );
    Ok(result)
}
