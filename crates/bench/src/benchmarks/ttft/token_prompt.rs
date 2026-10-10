// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Token-exact TTFT prompts. With a `tokenizer` parameter, a TTFT
//! gate cuts every user message to exactly the requested number of tokens of
//! the TARGET model's own tokenizer, so "32k ISL" means 32,768 tokens of the
//! model being measured, whatever its vocabulary.
//!
//! Owner: bench, ttft.
//! Invariants:
//! - The user message is `long_prompt::content(body, tag)`: the tag, a prefix
//!   of the gate's body text (the committed fixture for a high-ISL gate, the
//!   synthetic filler otherwise), then the task line. It encodes, without
//!   special tokens, to exactly the target count, or `prompt` errors; it is
//!   never sent at another size.
//! - A cold tag starts with a 16-hex-digit nonce unique to the run's salt and
//!   the sample, so two cold prompts differ from their first tag token and no
//!   prefix-cache block (vLLM or Metrale, both hash-chained from the first
//!   block) can be shared. The warm tag is fixed per target count.
//! - The pure search ([`ExactPrompts::prompt`]) reads the tokenizer only
//!   through [`Codec`]; [`load`] is the one place a tokenizer file is read.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::Mode;
use super::long_prompt::content;

/// 2026-10-10: The `tokenizer` value that keeps a gate's committed prompts (a
/// word, since `ParamKind::Text` refuses an empty value).
pub(crate) const NO_TOKENIZER: &str = "none";
/// 2026-10-10: How far the exact search walks from the binary-search point
/// before it gives up: the counts of neighbouring cuts differ by one token
/// except where a merge across the cut moves two at once.
const NEIGHBOURHOOD: usize = 16;
/// 2026-10-10: Words appended to the tag when no cut hits the target: a tokenizer can merge
/// the end of a cut with the newline after it so that neighbouring cuts skip the target by
/// two (seen with GLM-5.3's tokenizer on the synthetic filler at 4096), and a suffix that adds
/// one token shifts every count by one. The first entry is no suffix.
const TAG_SUFFIXES: &[&str] = &["", " a", " a b", " a b c"];
/// 2026-10-10: Spare body tokens beyond the largest target, so the tag and
/// the task line never leave the search without room.
const SPARE: usize = 64;

/// 2026-10-10: Text to token ids and back, without special tokens.
pub(crate) trait Codec {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}

impl Codec for tokenizers::Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let e = std::ops::Deref::deref(self)
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encoding: {e}"))?;
        Ok(e.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        std::ops::Deref::deref(self)
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decoding: {e}"))
    }
}

/// 2026-10-10: The `tokenizer.json` a `tokenizer` value names: the file
/// itself, or a checkpoint directory holding one.
pub fn tokenizer_file(value: &str) -> Result<PathBuf> {
    let p = Path::new(value);
    if p.is_dir() {
        let f = p.join("tokenizer.json");
        if !f.is_file() {
            bail!("tokenizer {value}: the directory holds no tokenizer.json");
        }
        return Ok(f);
    }
    if !p.is_file() {
        bail!(
            "tokenizer {value}: no such file or directory (pass a tokenizer.json or its checkpoint directory)"
        );
    }
    Ok(p.to_path_buf())
}

/// 2026-10-10: Read the tokenizer a `tokenizer` value names.
pub(crate) fn load(value: &str) -> Result<tokenizers::Tokenizer> {
    let file = tokenizer_file(value)?;
    tokenizers::Tokenizer::from_file(&file)
        .map_err(|e| anyhow::anyhow!("loading the tokenizer {}: {e}", file.display()))
}

/// 2026-10-10: A sample's tag: the fixed warm tag of a target count, or a cold
/// tag whose leading nonce is unique to `salt` and `sample`.
pub(crate) fn tag(mode: Mode, salt: u64, target: usize, sample: usize) -> String {
    match mode {
        Mode::Warm => format!("warm-{target}"),
        Mode::Cold => format!("{:016x}-cold-{target}", nonce(salt, sample)),
    }
}

/// 2026-10-10: splitmix64 over the salt and the sample index, so the nonces of
/// one run's samples are distinct (a bijection of `salt + sample`) and far apart.
fn nonce(salt: u64, sample: usize) -> u64 {
    let mut z = salt.wrapping_add((sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 2026-10-10: The token-exact prompt source of one configured run.
pub(crate) struct ExactPrompts<C: Codec> {
    codec: C,
    /// 2026-10-10: The body text's ids, repeated until they cover the largest
    /// target plus [`SPARE`].
    body: Vec<u32>,
    pub(crate) salt: u64,
    /// 2026-10-10: The tag suffix and body cut that last hit each target, tried first next time.
    hints: BTreeMap<usize, (usize, usize)>,
    /// 2026-10-10: The smallest server-reported prompt size per target.
    reported: BTreeMap<usize, usize>,
}

impl<C: Codec> ExactPrompts<C> {
    /// 2026-10-10: Encode `text` once, repeating it (newline-joined) until the
    /// ids cover `max_target` plus [`SPARE`].
    pub(crate) fn new(codec: C, text: &str, max_target: usize, salt: u64) -> Result<Self> {
        let unit = codec.encode(text)?;
        if unit.is_empty() {
            bail!("the prompt body encodes to no tokens");
        }
        let need = max_target + SPARE;
        let mut joined = text.to_string();
        let mut body = unit;
        while body.len() < need {
            joined.push('\n');
            joined.push_str(text);
            body = codec.encode(&joined)?;
        }
        Ok(Self {
            codec,
            body,
            salt,
            hints: BTreeMap::new(),
            reported: BTreeMap::new(),
        })
    }

    fn count(&self, cut: usize, tag: &str) -> Result<(usize, String)> {
        let text = content(&self.codec.decode(&self.body[..cut])?, tag);
        Ok((self.codec.encode(&text)?.len(), text))
    }

    /// 2026-10-10: The user message of one sample, exactly `target` tokens.
    pub(crate) fn prompt(&mut self, mode: Mode, target: usize, sample: usize) -> Result<String> {
        let base = tag(mode, self.salt, target, sample);
        if let Some(&(suffix, cut)) = self.hints.get(&target) {
            let (n, text) = self.count(cut, &format!("{base}{}", TAG_SUFFIXES[suffix]))?;
            if n == target {
                return Ok(text);
            }
        }
        let mut closest = Vec::new();
        for (suffix, pad) in TAG_SUFFIXES.iter().enumerate() {
            let tag = format!("{base}{pad}");
            if let Some((cut, text)) = self.search(&tag, target, &mut closest)? {
                self.hints.insert(target, (suffix, cut));
                return Ok(text);
            }
        }
        closest.sort_unstable();
        closest.dedup();
        bail!(
            "no cut of the prompt body encodes to exactly {target} tokens with tag {base:?} and \
             any of its suffixes {TAG_SUFFIXES:?} (counts near the cut: {closest:?}); the target \
             is below the tag and task line, or the tokenizer merges across every nearby cut"
        )
    }

    /// 2026-10-10: The cut, near the smallest one whose message reaches `target`, that encodes
    /// to exactly `target` with `tag`; the counts it saw go to `closest`.
    fn search(
        &self,
        tag: &str,
        target: usize,
        closest: &mut Vec<usize>,
    ) -> Result<Option<(usize, String)>> {
        let (mut lo, mut hi) = (0usize, self.body.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.count(mid, tag)?.0 >= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        for d in 0..=NEIGHBOURHOOD {
            for cut in [lo.checked_sub(d), lo.checked_add(d)].into_iter().flatten() {
                if cut > self.body.len() {
                    continue;
                }
                let (n, text) = self.count(cut, tag)?;
                if n == target {
                    return Ok(Some((cut, text)));
                }
                closest.push(n);
            }
        }
        Ok(None)
    }

    /// 2026-10-10: Accept one measured sample's server-reported prompt size:
    /// the template adds tokens to the message, so a count below `target`
    /// means the prompt was truncated or mis-rendered. `0` is a response
    /// without `usage.prompt_tokens`.
    pub(crate) fn admit(&mut self, target: usize, sample: usize, reported: usize) -> Result<()> {
        if reported == 0 {
            bail!(
                "sample {sample}: the response carried no usage.prompt_tokens, so the \
                 {target}-token prompt cannot be verified; the run is invalid"
            );
        }
        if reported < target {
            bail!(
                "sample {sample}: the server counted {reported} prompt tokens for a message of \
                 exactly {target} tokens of the target tokenizer; the prompt was truncated, or \
                 the tokenizer is not the served model's, and the run is invalid"
            );
        }
        let seen = self.reported.entry(target).or_insert(reported);
        *seen = (*seen).min(reported);
        Ok(())
    }

    /// 2026-10-10: The smallest server-reported size for `target`, once seen.
    pub(crate) fn reported(&self, target: usize) -> Option<usize> {
        self.reported.get(&target).copied()
    }

    /// 2026-10-10: The smallest server-reported size over every target.
    pub(crate) fn smallest_reported(&self) -> Option<usize> {
        self.reported.values().copied().min()
    }
}

/// 2026-10-10: Read a configured `tokenizer` value: `None` for
/// [`NO_TOKENIZER`], else the token-exact source over `text`.
pub(crate) fn configure(
    value: &str,
    text: &str,
    max_target: usize,
    salt: u64,
) -> Result<Option<ExactPrompts<tokenizers::Tokenizer>>> {
    if value == NO_TOKENIZER {
        return Ok(None);
    }
    let tokenizer = load(value)?;
    ExactPrompts::new(tokenizer, text, max_target, salt)
        .map(Some)
        .with_context(|| format!("building token-exact prompts with {value}"))
}

#[cfg(test)]
#[path = "token_prompt_tests.rs"]
mod tests;
