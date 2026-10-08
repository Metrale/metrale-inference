// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Fail-closed blocking text adapter; analysis is never returned as content.
use super::{Ending, stream::ByteTokenizer};

pub struct TextResponse {
    pub content: String,
    /// 2026-10-07: Generated analysis-body token IDs, excluding protocol headers/delimiters.
    pub reasoning_tokens: u32,
}

pub fn text_response(
    tokenizer: &ByteTokenizer,
    prompt: &[u32],
    output: &[u32],
) -> Result<TextResponse, &'static str> {
    let mut stream = tokenizer.assistant_stream(prompt)?;
    let mut final_text = None;
    for id in output {
        if let Some(message) = stream.push(*id)? {
            match (message.channel.as_deref(), message.ending) {
                (Some("analysis"), Ending::Message) => {}
                (None | Some("final"), Ending::Turn) if message.recipient.is_none() => {
                    final_text = Some(message.body);
                }
                _ => return Err("unsupported Harmony response channel or ending"),
            }
        }
    }
    stream.finish()?;
    Ok(TextResponse {
        content: final_text.ok_or("missing final Harmony answer")?,
        reasoning_tokens: stream.reasoning_tokens(),
    })
}
