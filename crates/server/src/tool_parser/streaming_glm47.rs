// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The streaming detector under [`CallPolicy::FailClosed`]
//! (`set_fail_closed`): only `<tool_call>` envelopes are calls, a call's
//! arguments are held until it closes, and each closed call is judged once
//! (`glm47::judge`) and emitted as a `CheckedToolCall`. The header goes out
//! early, as a `ToolCallStart`, once the name is complete and is an offered
//! tool, so the client sees the call start and the keep-alive chunks
//! (`api/chat_stream/keepalive.rs`) have a call to attach to.
//!
//! Owner: server (tool parser).
//! Invariants:
//! - A call emits at most one `ToolCallStart` and exactly one
//!   `CheckedToolCall`, whose `header_sent` says whether the start went out.
//! - Text outside envelopes is emitted as content, except a trailing prefix of
//!   `<tool_call>` and a whitespace-only tail, which wait for more text.

use super::glm47::{ARG_KEY, TOOL_CALL, TOOL_CALL_END, judge, split_body};
use super::*;

impl StreamingToolDetector {
    /// 2026-10-08: Use the fail-closed GLM-4.7 handling for every later
    /// `process` and `flush` call. `reset` keeps the setting.
    pub fn set_fail_closed(&mut self, on: bool) {
        self.fail_closed = on;
    }

    /// 2026-10-08: The index of the call whose arguments are being held after
    /// its header went out, under the fail-closed handling; `None` otherwise.
    pub fn held_call_index(&self) -> Option<usize> {
        (self.fail_closed && self.inside_tag && self.current_tc_name.is_some())
            .then_some(self.call_counter as usize)
    }

    /// 2026-10-08: `process` under the fail-closed handling; `new_text` is
    /// already in the buffer.
    pub(super) fn process_fail_closed(&mut self, outputs: &mut Vec<DetectorOutput>) {
        loop {
            if self.inside_tag {
                if let Some(end) = self.buffer.find(TOOL_CALL_END) {
                    let body = self.buffer[..end].to_string();
                    self.buffer.drain(..end + TOOL_CALL_END.len());
                    self.inside_tag = false;
                    outputs.push(self.checked_call(&body));
                    continue;
                }
                self.maybe_start_header(outputs);
                return;
            }
            if let Some(start) = self.buffer.find(TOOL_CALL) {
                if start > 0 {
                    outputs.push(DetectorOutput::Content(self.buffer[..start].to_string()));
                }
                self.buffer.drain(..start + TOOL_CALL.len());
                self.inside_tag = true;
                continue;
            }
            if self.buffer.trim().is_empty() {
                return;
            }
            let keep = trailing_opener_prefix_len(&self.buffer);
            let emit_to = self.buffer.len() - keep;
            if emit_to > 0 {
                outputs.push(DetectorOutput::Content(self.buffer[..emit_to].to_string()));
                self.buffer.drain(..emit_to);
            }
            return;
        }
    }

    /// 2026-10-08: `flush` under the fail-closed handling: a call the output
    /// cut short is judged on what was written; any other text is content.
    pub(super) fn flush_fail_closed(&mut self) -> Vec<DetectorOutput> {
        let text = std::mem::take(&mut self.buffer);
        if std::mem::take(&mut self.inside_tag) {
            return vec![self.checked_call(&text)];
        }
        if text.is_empty() {
            Vec::new()
        } else {
            vec![DetectorOutput::Content(text)]
        }
    }

    /// 2026-10-08: Send the open call's header once its name is complete
    /// (`<arg_key>` follows it) and is exactly an offered tool. Decided once
    /// per call: a name that is not offered gets no early header, and the
    /// verdict at the close carries it.
    fn maybe_start_header(&mut self, outputs: &mut Vec<DetectorOutput>) {
        if self.header_decided || !self.buffer.contains(ARG_KEY) {
            return;
        }
        self.header_decided = true;
        let (name, _) = split_body(&self.buffer);
        if !self.tools.iter().any(|t| t.function.name == name) {
            return;
        }
        let id = next_tool_call_id();
        outputs.push(DetectorOutput::ToolCallStart {
            id: id.clone(),
            name: name.to_string(),
            idx: self.call_counter as usize,
        });
        self.current_tc_name = Some(name.to_string());
        self.current_tc_id = Some(id);
    }

    /// 2026-10-08: Judge the call `body` and emit it under the header's id when
    /// a header went out, else under a new id.
    fn checked_call(&mut self, body: &str) -> DetectorOutput {
        let verdict = judge(body, &self.tools);
        let idx = self.call_counter as usize;
        let header_id = self.current_tc_id.take();
        let header_sent = header_id.is_some();
        let call = ToolCall {
            id: header_id.unwrap_or_else(next_tool_call_id),
            call_type: "function".into(),
            function: FunctionCall {
                name: verdict.name().to_string(),
                arguments: verdict.arguments().to_string(),
            },
        };
        self.call_counter += 1;
        self.emitted_tool_calls = true;
        self.reset_call_state();
        DetectorOutput::CheckedToolCall {
            call,
            idx,
            header_sent,
            refused: matches!(verdict, super::glm47::Verdict::Refuse { .. }),
        }
    }
}

/// 2026-10-08: Length of the longest suffix of `buf` that is a proper prefix
/// of `<tool_call>`; that much waits for the next delta.
fn trailing_opener_prefix_len(buf: &str) -> usize {
    (1..TOOL_CALL.len())
        .rev()
        .find(|&n| {
            buf.len() >= n && buf.is_char_boundary(buf.len() - n) && buf.ends_with(&TOOL_CALL[..n])
        })
        .unwrap_or(0)
}
