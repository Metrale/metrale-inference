// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Encoder from `ir::StreamDelta`s to OpenAI
//! `chat.completion.chunk` SSE events.
//!
//! Owner: server (OpenAI adapter).
//! Invariants:
//! - Every `chat.completion.chunk` of one response carries the same id.
//! - The role chunk comes first and `[DONE]` last.

use axum::response::sse::Event;

use crate::ir::StreamDelta;
use crate::reasoning_parser::ReasoningWire;

use super::{ChatCompletionChunk, Usage};

/// 2026-09-26: Encode one delta into its OpenAI SSE event(s).
/// `include_usage` is the request's `stream_options.include_usage`: it
/// chooses between a usage-only chunk followed by a finish chunk without
/// usage, and one finish chunk carrying usage.
pub(crate) fn delta_to_chunk_events(
    d: &StreamDelta,
    model: &str,
    id: &str,
    include_usage: bool,
    reasoning_wire: ReasoningWire,
) -> Vec<Event> {
    delta_to_payloads(d, model, id, include_usage, reasoning_wire)
        .into_iter()
        .map(|payload| Event::default().data(payload))
        .collect()
}

/// 2026-09-26: The `data:` payloads for one delta, in emission order. A
/// separate function so tests can read the chunk JSON; axum's `Event`
/// cannot be read back.
fn delta_to_payloads(
    d: &StreamDelta,
    model: &str,
    id: &str,
    include_usage: bool,
    reasoning_wire: ReasoningWire,
) -> Vec<String> {
    match d {
        // 2026-10-20: The OpenAI stream has no prompt-count event; the final
        // chunk's usage carries it.
        StreamDelta::Prompt { .. } => vec![],
        StreamDelta::Content { text, token_ids } => vec![chunk_json(
            ChatCompletionChunk::content_chunk(model, id, text.clone())
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::Reasoning { text, token_ids } => vec![chunk_json(
            ChatCompletionChunk::reasoning_chunk(model, id, reasoning_wire, text.clone())
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::ToolCallStart {
            index,
            id: call_id,
            name,
        } => {
            // 2026-09-26: `tool_call_start_chunk` takes a whole `ToolCall`;
            // the delta carries only id and name, and the start chunk's
            // arguments are empty.
            let tc = crate::tool_parser::ToolCall {
                id: call_id.clone(),
                call_type: "function".to_string(),
                function: crate::tool_parser::FunctionCall {
                    name: name.clone(),
                    arguments: String::new(),
                },
            };
            vec![chunk_json(ChatCompletionChunk::tool_call_start_chunk(
                model, id, &tc, *index,
            ))]
        }
        StreamDelta::ToolCallArgs {
            index,
            fragment,
            token_ids,
        } => vec![chunk_json(
            ChatCompletionChunk::tool_call_args_fragment(model, id, *index, fragment)
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::Refusal { text } => vec![chunk_json(ChatCompletionChunk::refusal_chunk(
            model,
            id,
            text.clone(),
        ))],
        StreamDelta::Finish {
            reason,
            usage,
            token_ids,
        } => {
            let wire_usage = wire_usage(usage);
            if include_usage {
                // 2026-09-26: The usage-only chunk (`choices: []`) comes
                // first, then the finish chunk without usage. The residual
                // token ids go on the finish chunk: the usage-only chunk
                // has no choice to carry them.
                vec![
                    chunk_json(ChatCompletionChunk::usage_only_chunk(model, id, wire_usage)),
                    chunk_json(
                        ChatCompletionChunk::final_chunk_no_usage(model, id, reason.as_wire())
                            .with_token_ids(token_ids.clone()),
                    ),
                ]
            } else {
                vec![chunk_json(
                    ChatCompletionChunk::done_chunk(model, id, reason.as_wire(), wire_usage)
                        .with_token_ids(token_ids.clone()),
                )]
            }
        }
        // 2026-09-26: The error payload is already wire JSON; it is sent
        // unchanged.
        StreamDelta::Error { message } => vec![message.clone()],
    }
}

fn chunk_json(chunk: ChatCompletionChunk) -> String {
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// 2026-09-26: The same `From<&ir::Usage>` mapping the blocking encoder
/// uses.
fn wire_usage(u: &crate::ir::Usage) -> Usage {
    Usage::from(u)
}

/// 2026-09-26: The streamed `/v1/chat/completions` response: the role
/// chunk, the encoded deltas, then `[DONE]`, with SSE keep-alive. It mints
/// the chunk id that every chunk of the response shares.
pub(crate) fn encode_sse_response(
    deltas: crate::ir::DeltaStream,
    model: String,
    include_usage: bool,
    reasoning_wire: ReasoningWire,
    store: Option<std::sync::Arc<crate::response_store::ResponseStore>>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures::StreamExt;

    let chunk_id = super::new_chunk_id();
    let role_chunk = super::ChatCompletionChunk::role_chunk(&model, &chunk_id);
    let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();
    let role_event = futures::stream::once(async move {
        Ok::<_, std::convert::Infallible>(Event::default().data(role_json))
    });
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    // 2026-09-26: Logged once, when the first delta reaches this encoder;
    // compare with `handle_token`'s first-delta log line.
    let first_event_logged = std::cell::Cell::new(false);
    // 2026-10-20: `store: true` on a streaming request stores the assembled
    // response, matching the blocking path: accumulate content, tool calls,
    // finish reason and usage as the deltas go by, and insert on stream end.
    let acc = store
        .as_ref()
        .map(|_| std::sync::Arc::new(std::sync::Mutex::new(StoredAccum::default())));
    let acc_for_stream = acc.clone();
    let stored_id = chunk_id.clone();
    let stored_model = model.clone();
    let token_stream = deltas.flat_map(move |d| {
        if !first_event_logged.replace(true) {
            tracing::debug!("stream: first delta reaches the SSE encoder");
        }
        if let Some(acc) = acc_for_stream.as_ref() {
            acc.lock().unwrap().observe(&d);
        }
        let events: Vec<Result<Event, std::convert::Infallible>> =
            delta_to_chunk_events(&d, &model, &chunk_id, include_usage, reasoning_wire)
                .into_iter()
                .map(Ok)
                .collect();
        futures::stream::iter(events)
    });
    let stored = acc.as_ref().map(|acc| {
        let acc = acc.clone();
        let id = stored_id;
        let model = stored_model;
        let store = store.clone().expect("store set iff accumulator set");
        futures::stream::once(async move {
            let body = acc.lock().unwrap().to_json(&id, &model);
            store.insert(crate::response_store::StoredEntry {
                id,
                kind: crate::response_store::StoredKind::ChatCompletion,
                model,
                created_at: crate::ids::unix_timestamp(),
                messages: Vec::new(),
                body,
                last_access: std::time::Instant::now(),
            });
        })
        .map(|()| unreachable!("the store stream yields no events"))
    });
    let token_stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = match stored {
        Some(stored) => Box::pin(token_stream.chain(stored)),
        None => Box::pin(token_stream),
    };
    let full_stream = role_event.chain(token_stream).chain(done_event);
    Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// 2026-10-20: Accumulates a streamed response into the same
/// `chat.completion` JSON the blocking path stores, so `GET
/// /v1/chat/completions/{id}` answers for a streamed `store: true` request
/// too.
#[derive(Default)]
struct StoredAccum {
    content: String,
    tool_calls: Vec<(usize, String, String, String)>, // (index, id, name, args)
    finish: Option<String>,
    usage: Option<crate::ir::Usage>,
}

impl StoredAccum {
    fn observe(&mut self, d: &StreamDelta) {
        match d {
            StreamDelta::Content { text, .. } => self.content.push_str(text),
            StreamDelta::ToolCallStart {
                index,
                id: call_id,
                name,
            } => self.tool_calls.push((*index, call_id.clone(), name.clone(), String::new())),
            StreamDelta::ToolCallArgs { index, fragment, .. } => {
                if let Some(entry) = self
                    .tool_calls
                    .iter_mut()
                    .find(|(i, ..)| i == index)
                {
                    entry.3.push_str(fragment);
                }
            }
            StreamDelta::Finish { reason, usage, .. } => {
                self.finish = Some(reason.as_wire().to_string());
                self.usage = Some(*usage);
            }
            _ => {}
        }
    }

    fn to_json(&self, id: &str, model: &str) -> serde_json::Value {
        let tool_calls: Vec<serde_json::Value> = self
            .tool_calls
            .iter()
            .map(|(index, id, name, args)| {
                serde_json::json!({
                    "index": index,
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            })
            .collect();
        let mut message = serde_json::json!({"role": "assistant"});
        if !self.content.is_empty() {
            message["content"] = serde_json::Value::String(self.content.clone());
        }
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = serde_json::Value::Array(tool_calls);
        }
        let mut body = serde_json::json!({
            "id": id,
            "object": "chat.completion",
            "created": crate::ids::unix_timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish.clone().unwrap_or_else(|| "stop".into()),
                "logprobs": null,
            }],
        });
        if let Some(u) = &self.usage {
            body["usage"] = serde_json::to_value(Usage::from(u)).unwrap_or_default();
        }
        body
    }
}

#[cfg(test)]
#[path = "encode_stream_tests.rs"]
mod tests;
