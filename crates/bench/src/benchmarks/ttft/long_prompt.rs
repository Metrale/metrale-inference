// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The high-ISL TTFT prompt source: one committed long prompt
//! (`prompts/long-32k.txt`, see `prompts/NOTICE.md`) behind a cold or warm
//! tag, with the server's prompt-token count checked on every measured sample.
//!
//! Owner: bench, ttft.
//! Invariants:
//! - A cold tag and the warm tag have the same shape: `COLD_TAG_PREFIX` or the
//!   warm tag's prefix, then `NONCE_DIGITS` decimal digits. The Qwen tokenizers
//!   split digits one token each, so every sample of either gate renders to the
//!   same token count (`scripts/make_long_prompt.py` asserts it).
//! - A measured sample whose response carries no `usage.prompt_tokens`, or
//!   fewer than `min_prompt_tokens`, ends the run with an error, so no record
//!   and no baseline come from a truncated or refused prompt.
//! - `scripts/make_long_prompt.py` reads `TASK_LINE`, `WARM_TAG`,
//!   `COLD_TAG_PREFIX` and `NONCE_DIGITS` from this file; `content` is the
//!   layout it mirrors, pinned by a digest in `long_prompt_tests.rs`.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use serde_json::{Value, json};

use super::{Mode, TtftGate};
use crate::benchmark::BenchmarkDescriptor;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};

/// 2026-09-27: The `prompt` choice naming the committed 32k fixture.
pub(crate) const LONG_32K: &str = "long-32k";
/// 2026-09-27: Every `prompt` choice, in the order the parameter lists them.
const PROMPT_CHOICES: &[&str] = &[LONG_32K];
const LONG_32K_TEXT: &str = include_str!("prompts/long-32k.txt");
/// 2026-09-27: The token count `prompts/long-32k.txt` was cut to, through the
/// Qwen3.6-35B-A3B tokenizer and the chat template this engine serves it with
/// (`prompts/NOTICE.md`); the default `min_prompt_tokens`.
pub(crate) const LONG_32K_TOKENS: usize = 32_768;

/// 2026-09-27: The instruction after the text; the reply is capped at 8 tokens,
/// so it only has to be a plausible question.
pub(crate) const TASK_LINE: &str = "In one sentence, what is the passage above about?";
/// 2026-09-27: The warm gate's fixed tag, the same shape as a cold tag.
pub(crate) const WARM_TAG: &str = "warm-32k-0000000000000000";
pub(crate) const COLD_TAG_PREFIX: &str = "cold-32k-";
pub(crate) const NONCE_DIGITS: usize = 16;
/// 2026-09-27: The run salt fills the nonce's leading digits and the sample
/// index its last `SAMPLE_DIGITS`, so a sample index below 1000 (the `repeats`
/// spec caps it at 200) can never collide with another within a run.
const SAMPLE_DIGITS: usize = 3;
const SALT_MODULUS: u64 = 10u64.pow((NONCE_DIGITS - SAMPLE_DIGITS) as u32);

/// 2026-09-27: The fixture text for a `prompt` choice, matched case-insensitively
/// as `ParamKind::Choice` matches.
pub(crate) fn fixture_text(name: &str) -> Option<&'static str> {
    name.eq_ignore_ascii_case(LONG_32K).then_some(LONG_32K_TEXT)
}

/// 2026-09-27: The user message: the tag first, so no two cold samples share a
/// prefix, then the text (which ends with a newline) and the task line.
pub(crate) fn content(text: &str, tag: &str) -> String {
    format!("[{tag}] {text}\n{TASK_LINE}")
}

/// 2026-09-27: The tag of one sample: the fixed warm tag, or a cold tag unique
/// to this run's `salt` and the sample index.
pub(crate) fn tag(mode: Mode, salt: u64, sample: usize) -> String {
    match mode {
        Mode::Warm => WARM_TAG.to_string(),
        Mode::Cold => format!(
            "{COLD_TAG_PREFIX}{:0salt_w$}{:0sample_w$}",
            salt % SALT_MODULUS,
            sample % 10usize.pow(SAMPLE_DIGITS as u32),
            salt_w = NONCE_DIGITS - SAMPLE_DIGITS,
            sample_w = SAMPLE_DIGITS,
        ),
    }
}

/// 2026-09-27: A salt for one configured run, from the wall clock and the pid,
/// so a server that outlives this process never sees a cold tag again.
pub(crate) fn fresh_salt() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    // 2026-09-27: splitmix64's finaliser, so nearby clocks give distant salts.
    let mut z = nanos ^ (u64::from(std::process::id()) << 32);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 2026-09-27: What a long-prompt request adds to the synthetic gates' body:
/// usage on the stream, which `vLLM` sends only when asked, so every sample's
/// prompt size can be checked; and thinking disabled, so each engine renders
/// its chat template the same defined way (`prompts/NOTICE.md` has the counts).
pub(crate) fn extend_request(body: &mut Value) {
    body["stream_options"] = json!({"include_usage": true});
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
}

impl TtftGate {
    /// 2026-09-27: A high-ISL gate: the TTFT state machine and verdict, with
    /// its own descriptor, and so its own stored baseline, over a long prompt.
    pub(crate) fn high_isl(
        mode: Mode,
        descriptor: &'static BenchmarkDescriptor,
        metadata: &'static PluginMetadata,
    ) -> Self {
        Self::build(mode, descriptor, metadata, true)
    }

    /// 2026-09-27: The smallest server-reported prompt size so far; `None` for
    /// a synthetic gate or before the first measured sample.
    pub(super) fn observed_prompt_tokens(&self) -> Option<usize> {
        self.long.as_ref().and_then(|long| long.prompt_tokens)
    }
}

/// 2026-09-27: The configured long-prompt source of a high-ISL gate.
pub(crate) struct LongPrompt {
    pub text: &'static str,
    pub min_prompt_tokens: usize,
    pub salt: u64,
    /// 2026-09-27: The smallest server-reported prompt size among the measured
    /// samples so far; the `prompt_tokens` metric.
    pub prompt_tokens: Option<usize>,
}

impl LongPrompt {
    /// 2026-09-27: Read the source's parameters.
    pub(crate) fn configure(values: &ParamValues) -> Result<Self> {
        let name = values.text("prompt")?;
        let Some(text) = fixture_text(name) else {
            bail!("unknown prompt {name:?}; expected one of {PROMPT_CHOICES:?}");
        };
        Ok(Self {
            text,
            min_prompt_tokens: values.usize("min_prompt_tokens")?,
            salt: fresh_salt(),
            prompt_tokens: None,
        })
    }

    /// 2026-09-27: The user message of one sample.
    pub(crate) fn prompt(&self, mode: Mode, sample: usize) -> String {
        content(self.text, &tag(mode, self.salt, sample))
    }

    /// 2026-09-27: Accept one measured sample's server-reported prompt size, or
    /// end the run. `0` is what `http::ChatOutcome` holds when the response
    /// carried no `usage.prompt_tokens`.
    pub(crate) fn admit(&mut self, sample: usize, prompt_tokens: usize) -> Result<()> {
        if prompt_tokens == 0 {
            bail!(
                "sample {sample}: the response carried no usage.prompt_tokens, so the prompt \
                 size this gate claims cannot be verified; the run is invalid"
            );
        }
        if prompt_tokens < self.min_prompt_tokens {
            bail!(
                "sample {sample}: the server counted {prompt_tokens} prompt tokens, below \
                 min_prompt_tokens {}; the prompt was truncated or mis-rendered and the run is \
                 invalid",
                self.min_prompt_tokens
            );
        }
        self.prompt_tokens = Some(
            self.prompt_tokens
                .map_or(prompt_tokens, |seen| seen.min(prompt_tokens)),
        );
        Ok(())
    }
}

/// 2026-09-27: The source parameters of a high-ISL gate, in place of the
/// synthetic gates' `prompt_lengths`.
pub(crate) fn source_parameters() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new(
            "prompt",
            "Prompt",
            "The committed long prompt to send (crates/bench/src/benchmarks/ttft/prompts).",
            ParamKind::Choice(PROMPT_CHOICES),
            ParamValue::Text(LONG_32K.to_string()),
        ),
        ParamSpec::new(
            "min_prompt_tokens",
            "Minimum prompt tokens",
            "The run is invalid if the server reports fewer prompt tokens than this for any sample.",
            ParamKind::Int {
                min: 1,
                max: 1_048_576,
            },
            ParamValue::Int(LONG_32K_TOKENS as i64),
        ),
    ]
}

#[cfg(test)]
#[path = "long_prompt_tests.rs"]
mod tests;
