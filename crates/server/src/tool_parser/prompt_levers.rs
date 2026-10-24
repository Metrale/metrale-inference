// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`PromptLevers`]: a model's settings that change how its tool
//! prompt is rendered. They come from the model's MODEL.toml `[behavior]`
//! table and (2026-10-17) the load's resolved settings (`api/chat/levers.rs`
//! `ChatLevers::resolve`), and each parser receives them as an argument.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

/// 2026-09-26: Prompt-rendering settings for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PromptLevers {
    /// 2026-09-26: `[behavior].tscg`: render tool schemas as compact TSCG
    /// function signatures instead of JSON. False when MODEL.toml omits it
    /// (`crates/kernels/build_parse_behavior.rs`).
    pub tscg: bool,
    /// 2026-10-17: `METRALE_OFFICIAL_TOOL_PROMPT`: qwen3_coder renders the four-bullet
    /// `<IMPORTANT>` reminder of `jinja-templates/qwen3_5_moe.jinja` instead of its
    /// `<IMMEDIATE_TOOL_USE>` block. Off when unset.
    pub official_tool_prompt: bool,
}

impl PromptLevers {
    /// 2026-09-26: TSCG off: tool schemas render as JSON. `ChatLevers::OFF`
    /// and the parser tests use it.
    pub const OFF: Self = Self {
        tscg: false,
        official_tool_prompt: false,
    };

    pub fn new(tscg: bool, official_tool_prompt: bool) -> Self {
        Self {
            tscg,
            official_tool_prompt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_the_default_and_the_json_path() {
        assert_eq!(PromptLevers::default(), PromptLevers::OFF);
        const { assert!(!PromptLevers::OFF.tscg) };
    }

    #[test]
    fn two_models_can_disagree_within_one_process() {
        let a = PromptLevers::new(true, false);
        let b = PromptLevers::new(false, false);
        assert!(a.tscg && !b.tscg);
    }
}
