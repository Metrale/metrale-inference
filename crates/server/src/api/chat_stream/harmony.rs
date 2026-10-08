// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Token-aware streaming branch, bypassing generic decoded-text sanitizers.
use super::ctx::StreamCtx;
use crate::{api::inference_types::StreamEvent, harmony::text_stream::TextStream, ir::StreamDelta};
use futures::StreamExt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(super) fn adapt(
    events: impl futures::Stream<Item = StreamEvent> + Send + 'static,
    mut parser: TextStream,
    ctx: StreamCtx,
    cancel: Arc<AtomicBool>,
) -> crate::ir::DeltaStream {
    let guard = CancelOnDrop(cancel.clone());
    let mut failed = false;
    Box::pin(events.flat_map(move |event| {
        let _keep_guard = &guard;
        if failed {
            return futures::stream::iter(Vec::new());
        }
        let mut out = Vec::new();
        let result: Result<(), String> = (|| {
            match event {
                StreamEvent::Token(id) | StreamEvent::TokenWithLogprobs(id, _) => {
                    let text = parser.push(id).map_err(str::to_owned)?;
                    if !text.is_empty() {
                        out.push(StreamDelta::Content {
                            text,
                            token_ids: vec![],
                        });
                    }
                }
                StreamEvent::PromptLogprobs(_) => {}
                StreamEvent::Error(message) => return Err(message),
                StreamEvent::Done {
                    terminal_token_id,
                    finish_reason,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens: _,
                    cached_prompt_tokens,
                    accepted_prediction_tokens,
                    guard_stop,
                    ..
                } => {
                    let text =
                        finish_parser(&mut parser, terminal_token_id, &finish_reason, guard_stop)?;
                    if !text.is_empty() {
                        out.push(StreamDelta::Content {
                            text,
                            token_ids: vec![],
                        });
                    }
                    let usage = crate::ir::Usage {
                        prompt_tokens: ctx.prompt_len,
                        completion_tokens,
                        cached_prompt_tokens: cached_prompt_tokens as usize,
                        reasoning_tokens: parser.reasoning_tokens() as usize,
                        accepted_prediction_tokens,
                        time_to_first_token_ms,
                        decode_time_ms,
                        response_tokens_per_second: crate::ir::Usage::decode_rate_tok_s(
                            completion_tokens,
                            decode_time_ms,
                        ),
                    };
                    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
                    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
                    if let Some(ref rctx) = ctx.req_ctx {
                        ctx.state.rate_limiter.refund_tokens(
                            &rctx.identity,
                            rctx.reserved_tokens
                                .saturating_sub((ctx.prompt_len + completion_tokens) as u64),
                        );
                    }
                    let reason = if let Some(call) = parser.take_tool_call() {
                        out.push(StreamDelta::ToolCallStart {
                            index: 0,
                            id: format!("call_{}", crate::ids::uuid_v4()),
                            name: call.name,
                        });
                        out.push(StreamDelta::ToolCallArgs {
                            index: 0,
                            fragment: call.arguments.to_string(),
                            token_ids: vec![],
                        });
                        crate::ir::FinishReason::ToolCalls
                    } else {
                        crate::ir::FinishReason::Stop
                    };
                    out.push(StreamDelta::Finish {
                        reason,
                        usage,
                        token_ids: vec![],
                    });
                }
            }
            Ok(())
        })();
        if let Err(message) = result {
            failed = true;
            cancel.store(true, Ordering::Release);
            out.clear();
            // 2026-10-07: Reuse the wire-ready error envelope and reservation refund.
            out.extend(super::handle_error::handle_error(&ctx, message));
        }
        futures::stream::iter(out)
    }))
}

// 2026-10-07: Completion requires both scheduler evidence and the protocol terminal.
fn finish_parser(
    parser: &mut TextStream,
    terminal: Option<u32>,
    reason: &str,
    guard: Option<&str>,
) -> Result<String, String> {
    if reason != "stop" || guard.is_some() {
        return Err("Harmony generation ended before a verified final turn".into());
    }
    let text = parser
        .push(terminal.ok_or("missing Harmony terminal token")?)
        .map_err(str::to_owned)?;
    parser.finish().map_err(str::to_owned)?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parser() -> TextStream {
        let tokenizer = Arc::new(
            crate::harmony::stream::ByteTokenizer::from_tokenizer_json(include_str!(
                "../../harmony/fixtures/gpt-oss-byte-vocab.json"
            ))
            .unwrap(),
        );
        let prompt: Vec<_> = [
            vec![200006],
            "assistant".bytes().map(|b| u32::from(b - 33)).collect(),
        ]
        .concat();
        let mut p = TextStream::new(tokenizer, &prompt).unwrap();
        p.push(200008).unwrap();
        assert_eq!(p.push(u32::from(b'x' - 33)).unwrap(), "x");
        p
    }
    #[test]
    fn completion_requires_terminal_and_rejects_timeout_truncation_and_duplicates() {
        for (id, reason, guard) in [
            (None, "stop", None),
            (Some(200002), "length", None),
            (Some(200002), "timeout", None),
            (Some(200002), "stop", Some("cancel")),
            (Some(200007), "stop", None),
            (Some(200012), "stop", None),
        ] {
            assert!(finish_parser(&mut parser(), id, reason, guard).is_err());
        }
        let mut p = parser();
        assert_eq!(
            finish_parser(&mut p, Some(200002), "stop", None).unwrap(),
            ""
        );
        assert!(finish_parser(&mut p, Some(200002), "stop", None).is_err());
    }
    #[test]
    fn dropping_unpolled_or_partial_stream_cancels_scheduler() {
        let flag = Arc::new(AtomicBool::new(false));
        let guard = CancelOnDrop(flag.clone());
        assert!(!flag.load(Ordering::Acquire));
        drop(guard);
        assert!(flag.load(Ordering::Acquire));
    }
}
