// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Pinned text-only prompt tokenization; no generic chat-template rewrite.
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
const SYSTEM: &str = "Comprehend and analyze the provided prompt.";

pub struct TextPromptEncoder {
    tokenizer: Tokenizer,
    prefix_ids: Vec<u32>,
    vision_ids: Vec<u32>,
}
pub struct EncodedPrompt {
    pub input_ids: Vec<u32>,
    pub drop_prefix: usize,
}
fn system_prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n")
}
fn render(prompt: &str) -> String {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}
fn verify_asset(path: &str, bytes: &[u8]) -> Result<()> {
    // 2026-10-06: The checked-in checkpoint manifest is the sole asset-hash source.
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/model-manifests/qwen-image-2.1.json"
    ))?;
    let expected = manifest["files"]
        .as_array()
        .and_then(|files| files.iter().find(|entry| entry["path"] == path))
        .and_then(|entry| entry["sha256"].as_str())
        .ok_or_else(|| anyhow::anyhow!("missing pinned asset hash"))?;
    let actual: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    ensure!(
        actual == expected,
        "processor asset differs from pinned checkpoint: {path}"
    );
    Ok(())
}
impl TextPromptEncoder {
    /// 2026-10-06: Caller supplies verified asset bytes through its I/O boundary.
    /// The pinned system-only chat template was independently proven identical
    /// to this raw prefix; derive its token count rather than hardcode 14.
    pub fn from_bytes(tokenizer_json: &[u8], chat_template: &[u8]) -> Result<Self> {
        verify_asset("processor/tokenizer.json", tokenizer_json)?;
        verify_asset("processor/chat_template.jinja", chat_template)?;
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_json).map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            tokenizer.get_padding().is_none() && tokenizer.get_truncation().is_none(),
            "text-only encoder requires unpadded, untruncated tokenizer"
        );
        let prefix_ids = tokenizer
            .encode(system_prefix(), true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        ensure!(!prefix_ids.is_empty(), "empty system prefix");
        let vision_ids = [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        .iter()
        .map(|name| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| anyhow::anyhow!("missing vision token"))
        })
        .collect::<Result<_>>()?;
        Ok(Self {
            tokenizer,
            prefix_ids,
            vision_ids,
        })
    }
    pub fn encode(&self, prompt: &str, max_tokens: usize) -> Result<EncodedPrompt> {
        ensure!(
            max_tokens > self.prefix_ids.len(),
            "token capacity cannot fit prefix"
        );
        let encoding = self
            .tokenizer
            .encode(render(prompt), true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = encoding.get_ids();
        ensure!(
            ids.len() <= max_tokens && ids.len() > self.prefix_ids.len(),
            "prompt exceeds token capacity"
        );
        ensure!(
            ids.starts_with(&self.prefix_ids),
            "tokenized system prefix changed"
        );
        ensure!(
            !ids.iter().any(|id| self.vision_ids.contains(id)),
            "image/video-conditioned prompts are not implemented"
        );
        Ok(EncodedPrompt {
            input_ids: ids.to_vec(),
            drop_prefix: self.prefix_ids.len(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unpinned_assets_without_parsing_or_fallback() {
        assert!(TextPromptEncoder::from_bytes(b"{}", b"template").is_err());
        assert!(verify_asset("processor/chat_template.jinja", b"alternate").is_err());
    }
    #[test]
    fn raw_layout_preserves_empty_space_unicode_and_newlines() {
        assert_eq!(render(""), render(" "));
        assert_ne!(render(""), render("  "));
        let text = render("café 日本 😀\nnext");
        assert!(text.ends_with("user\ncafé 日本 😀\nnext<|im_end|>\n<|im_start|>assistant\n"));
        assert_eq!(text.matches("<|im_start|>").count(), 3);
        assert!(!text.contains("<think>"));
    }
}
