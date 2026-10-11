// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native checkpoint tool/history preparation and exact blocking IR mapping.
use crate::{harmony::tool_schema::ToolSchema, ir, tool_parser::ToolChoice};
use std::sync::Arc;

fn schemas(req: &ir::ChatRequest) -> Result<Vec<ToolSchema>, &'static str> {
    if req.tools.len() > 32 {
        return Err("at most32 Harmony tools are supported");
    }
    let mut names = std::collections::BTreeSet::new();
    req.tools.iter().map(|tool| {
        if tool.tool_type!="function" || !names.insert(tool.function.name.as_str()) {return Err("only uniquely named function tools are supported");}
        ToolSchema::new(&tool.function.name,tool.function.parameters.clone().unwrap_or_else(||serde_json::json!({"type":"object","properties":{},"additionalProperties":false})))
    }).collect()
}

// 2026-10-07: Validate before infallible OpenAI lowering can replace bad JSON with {}.
pub(crate) fn validate_wire(
    req: &crate::openai::ChatCompletionRequest,
    body: &[u8],
) -> Result<(), &'static str> {
    crate::harmony::strict_json::parse(
        std::str::from_utf8(body).map_err(|_| "invalid request UTF8")?,
    )?;
    for message in &req.messages {
        for call in message.tool_calls.iter().flatten() {
            crate::harmony::strict_json::parse(&call.function.arguments)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_request(req: &ir::ChatRequest) -> Result<(), &'static str> {
    if let Some(choice) = &req.tool_choice
        && !matches!(choice,ToolChoice::Mode(mode) if mode=="auto" || mode=="none")
    {
        return Err("Harmony tool_choice supports only auto or none");
    }
    let tools = schemas(req)?;
    let mut pending: Option<(&str, &str)> = None;
    let mut call_ids = std::collections::BTreeSet::new();
    for (index, message) in req.messages.iter().enumerate() {
        if index != 0 && matches!(message.role.as_wire(), "system" | "developer") {
            return Err("Harmony supports one leading system/developer message");
        }
        if !matches!(
            message.role.as_wire(),
            "system" | "developer" | "user" | "assistant" | "tool"
        ) || message.tool_error
        {
            return Err("unsupported Harmony message role or tool-error flag");
        }
        if message
            .content
            .iter()
            .any(|part| !matches!(part, ir::ContentPart::Text(_)))
        {
            return Err("Harmony supports text messages only");
        }
        if message.reasoning.is_some() {
            return Err("Harmony reasoning-history input is not supported");
        }
        if !message.tool_calls.is_empty() {
            if message.role != ir::Role::Assistant
                || message.tool_calls.len() != 1
                || pending.is_some()
                || !message.text().is_empty()
            {
                return Err(
                    "Harmony requires one outstanding tool call and empty assistant content",
                );
            }
            let call = &message.tool_calls[0];
            if call.id.is_empty() || !call_ids.insert(call.id.as_str()) {
                return Err("tool call history requires an ID");
            }
            let tool = tools
                .iter()
                .find(|t| t.name == call.name)
                .ok_or("undeclared historical tool")?;
            tool.validate(&call.arguments)?;
            pending = Some((&call.id, &call.name));
        } else if message.role == ir::Role::Tool {
            let (id, name) = pending.take().ok_or("orphan tool result")?;
            if message.tool_call_id.as_deref() != Some(id)
                || message.name.as_deref().is_some_and(|n| n != name)
            {
                return Err("tool result ID or name does not match preceding call");
            }
        } else if pending.is_some() {
            return Err("tool call must be followed by its tool result");
        }
    }
    if pending.is_some() {
        return Err("missing tool result in conversation history");
    }
    Ok(())
}

// 2026-10-07: Match the existing prompt-admission Response error contract.
#[allow(clippy::result_large_err)]
pub(crate) fn prepare(
    state: &Arc<crate::AppState>,
    req: &ir::ChatRequest,
) -> Result<super::prepare::PreparedChat, axum::response::Response> {
    let result = (|| -> anyhow::Result<_> {
        validate_request(req).map_err(anyhow::Error::msg)?;
        let active =
            !req.tools.is_empty() && !req.tool_choice.as_ref().is_some_and(ToolChoice::is_none);
        let tools = active
            .then(|| {
                req.tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let messages:Vec<_>=req.messages.iter().map(|m|{
            let mut value=serde_json::json!({"role":m.role.as_wire(),"content":m.text()});
            if !m.tool_calls.is_empty() {value["tool_calls"]=serde_json::Value::Array(m.tool_calls.iter().map(|call|serde_json::json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}})).collect());}
            value
        }).collect();
        let effort = req
            .reasoning_effort
            .or(state.default_reasoning_effort)
            .map(ir::ReasoningEffort::as_str);
        let prompt_tokens = state.tokenizer.apply_chat_template_openai_with_effort(
            &messages,
            tools.as_deref(),
            false,
            false,
            effort,
            None,
        )?;
        Ok(super::prepare::PreparedChat {
            tools_active: active,
            cwd_hint: None,
            image_pixels: vec![],
            prompt_tokens,
            enable_thinking: false,
            thinking_budget: None,
        })
    })();
    result.map_err(|e| {
        super::super::compact::openai_error_response(
            axum::http::StatusCode::BAD_REQUEST,
            format!("Harmony request: {e}"),
        )
    })
}

pub(crate) fn choice(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    req: &ir::ChatRequest,
    prompt: &[u32],
    output: &[u32],
    index: usize,
) -> Result<(ir::Choice, u32), &'static str> {
    let tools = if req.tool_choice.as_ref().is_some_and(ToolChoice::is_none) {
        vec![]
    } else {
        schemas(req)?
    };
    let parsed = crate::harmony::tool_response::response(
        tokenizer.harmony().ok_or("missing Harmony tokenizer")?,
        prompt,
        output,
        &tools,
    )?;
    let calls = parsed
        .tool_call
        .map(|call| ir::message::ToolCall {
            id: format!("call_{}", crate::ids::uuid_v4()),
            name: call.name,
            arguments: call.arguments,
        })
        .into_iter()
        .collect::<Vec<_>>();
    let reason = if calls.is_empty() {
        ir::FinishReason::Stop
    } else {
        ir::FinishReason::ToolCalls
    };
    Ok((
        ir::Choice {
            index,
            content: parsed.content,
            reasoning: None,
            tool_calls: calls,
            refusal: None,
            finish_reason: reason,
            matched_stop: None,
            logprobs: None,
        },
        parsed.reasoning_tokens,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wire(messages: serde_json::Value) -> crate::openai::ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({"model":"gpt","messages":messages,"tools":[{"type":"function","function":{"name":"lookup_part","parameters":{"type":"object","properties":{"part_id":{"type":"string"}},"required":["part_id"],"additionalProperties":false}}}]})).unwrap()
    }
    #[test]
    fn history_requires_matching_id_declared_name_and_single_outstanding_call() {
        let valid = serde_json::json!([
            {"role":"user","content":"Look up A-42."},
            {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup_part","arguments":"{\"part_id\":\"A-42\"}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"available"}
        ]);
        assert!(validate_request(&wire(valid.clone()).into()).is_ok());
        for (field, value) in [("tool_call_id", "wrong"), ("name", "wrong")] {
            let mut bad = valid.clone();
            bad[2][field] = value.into();
            assert!(validate_request(&wire(bad).into()).is_err());
        }
        let mut bad = valid.clone();
        bad[1]["tool_calls"][0]["function"]["name"] = "LOOKUP_PART".into();
        assert!(validate_request(&wire(bad).into()).is_err());
        assert!(validate_request(&wire(serde_json::json!([valid[2]])).into()).is_err());
        assert!(validate_request(&wire(serde_json::json!([valid[0], valid[1]])).into()).is_err());
    }
    #[test]
    fn malformed_and_duplicate_history_arguments_refuse_before_lossy_lowering() {
        for arguments in ["not JSON", r#"{"part_id":"A","part_id":"B"}"#] {
            let req = wire(
                serde_json::json!([{"role":"assistant","tool_calls":[{"id":"call_1","function":{"name":"lookup_part","arguments":arguments}}]}]),
            );
            assert!(validate_wire(&req, b"{}").is_err());
        }
        let req = wire(serde_json::json!([{"role":"user","content":"hello"}]));
        assert!(validate_wire(&req, br#"{"model":"a","model":"b"}"#).is_err());
    }
}
