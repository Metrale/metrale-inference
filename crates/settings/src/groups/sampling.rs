// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `sampling` lever group: the `sampling.*` levers a `met` process reads at run time.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: The `sampling` group, resolved by `crate::resolve()`.
    pub struct SamplingSettings { group: "sampling", class: Runtime }
    agentic_preserve_thinking: Switch = "off", None, "METRALE_AGENTIC_PRESERVE_THINKING", "crates/bench/src/benchmarks/agentic/agent_history.rs", "`METRALE_AGENTIC_PRESERVE_THINKING=1`: echo each turn's non-empty reasoning as `reasoning_content` in its assistant message, and send `preserve_thinking: true` in `chat_template_kwargs`. `1` turns it on.";
    disable_fast_greedy: Switch = "off", None, "METRALE_DISABLE_FAST_GREEDY", "crates/server/src/scheduler/levers.rs", "Ships ON; `METRALE_DISABLE_FAST_GREEDY=1` opts out. `1` turns it on.";
    disable_fast_masked: Switch = "off", None, "METRALE_DISABLE_FAST_MASKED", "crates/server/src/scheduler/levers.rs", "Ships ON; `METRALE_DISABLE_FAST_MASKED=1` opts out. `1` turns it on.";
    disable_forced_token: Switch = "off", None, "METRALE_DISABLE_FORCED_TOKEN", "crates/server/src/scheduler/levers.rs", "Ships ON; `METRALE_DISABLE_FORCED_TOKEN=1` opts out. `1` turns it on.";
    disable_refusal_detection: Switch = "off", None, "METRALE_DISABLE_REFUSAL_DETECTION", "crates/server/src/api/chat/levers.rs", "Set `METRALE_DISABLE_REFUSAL_DETECTION=1` to force `refusal: None` on all responses, matching pre-PR-4 behavior byte-for-byte. `1` turns it on.";
    disable_rne: Presence = "off", None, "METRALE_DISABLE_RNE", "crates/core/src/numeric.rs", "`METRALE_DISABLE_RNE` is a bisect escape hatch that reverts to truncation. Any value turns it on, `0` included.";
    force_temp_zero: Switch = "off", None, "METRALE_FORCE_TEMP_ZERO", "crates/server/src/api/chat/sampling_setup.rs", "METRALE_FORCE_TEMP_ZERO=1 — diagnostic override that forces fully greedy deterministic decoding, ignoring client params AND MODEL.toml presets. `1` or `true` turns it on.";
    grammar_allow_empty_value: Switch = "off", None, "METRALE_GRAMMAR_ALLOW_EMPTY_VALUE", "crates/server/src/grammar/compile_tools.rs", "P1-1 (2026-07-09, env `METRALE_GRAMMAR_ALLOW_EMPTY_VALUE=1`): the non-empty guard makes an EMPTY parameter value unrepresentable, so a model intending `<parameter=content></parameter>` is. `1` turns it on.";
    grammar_async_prewarm: Text = "unset", None, "METRALE_GRAMMAR_ASYNC_PREWARM", "crates/server/src/grammar/prewarm.rs", "Kill switch: `METRALE_GRAMMAR_ASYNC_PREWARM=0` restores the synchronous prewarm.";
    grammar_budget_close: Switch = "on", None, "METRALE_GRAMMAR_BUDGET_CLOSE", "crates/server/src/scheduler/levers.rs", "`METRALE_GRAMMAR_BUDGET_CLOSE` (default on). `0` turns it off.";
    grammar_cache: Text = "unset", None, "METRALE_GRAMMAR_CACHE", "crates/server/src/grammar/mask_cache.rs", "Controls: `METRALE_GRAMMAR_CACHE=0` disables persistence entirely; `METRALE_GRAMMAR_CACHE_DIR` relocates the file (for a read-only model directory, or to share one warm cache across several model...";
    grammar_cache_dir: Path = "unset", None, "METRALE_GRAMMAR_CACHE_DIR", "crates/server/src/grammar/mask_cache.rs", "Controls: `METRALE_GRAMMAR_CACHE=0` disables persistence entirely; `METRALE_GRAMMAR_CACHE_DIR` relocates the file (for a read-only model directory, or to share one warm cache across several model...";
    grammar_force_close: Switch = "off", None, "METRALE_GRAMMAR_FORCE_CLOSE", "crates/server/src/grammar/compile_tools.rs", "P1-2 (2026-07-09, env `METRALE_GRAMMAR_FORCE_CLOSE=1`): with `force_close`, the DEEPEST arm (`\"</parameter\" [^>]` for qwen3_coder) is omitted, so once the value contains the full close-prefix the. `1` turns it on.";
    grammar_value_harden: Switch = "off", None, "METRALE_GRAMMAR_VALUE_HARDEN", "crates/server/src/grammar/compile_tools.rs", "F2-2a (2026-06-02): structural ceiling on a parameter VALUE's `rest` repetition, applied ONLY when the `METRALE_GRAMMAR_VALUE_HARDEN` kill-switch is on. `1` turns it on.";
    hw_temp_gate: Text = "unset", None, "METRALE_HW_TEMP_GATE", "crates/bench/src/hardware/policy.rs", "Env var that opts in to the absolute-temperature ceilings.";
    inthink_tool_leak_openers: Text = "unset", None, "METRALE_INTHINK_TOOL_LEAK_OPENERS", "crates/server/src/api/chat/levers.rs", "`METRALE_INTHINK_TOOL_LEAK_OPENERS=N` — how many tool-call openers (`<tool_call>`, `<function=`, …) the in-think leak scanner tolerates in a stream's reasoning before cancelling the sequence.";
    kernel_set_hash: Text = "unset", None, "METRALE_KERNEL_SET_HASH", "crates/kernels/src/lib.rs", "Content fingerprint of the generated kernel set, emitted by `build.rs` as a `rustc-env`.";
    nllb_device_topk_min_rows: Int = "unset", None, "METRALE_NLLB_DEVICE_TOPK_MIN_ROWS", "crates/model-engine/src/model/nllb/beam_multi.rs", "So gate on the batch row count (METRALE_NLLB_DEVICE_TOPK_MIN_ROWS, default 64); METRALE_NLLB_HOST_TOPK=1 forces the host path (A/B + >16-beam requests).";
    nllb_host_topk: Switch = "off", None, "METRALE_NLLB_HOST_TOPK", "crates/model-engine/src/model/nllb/beam_multi.rs", "So gate on the batch row count (METRALE_NLLB_DEVICE_TOPK_MIN_ROWS, default 64); METRALE_NLLB_HOST_TOPK=1 forces the host path (A/B + >16-beam requests). `1` or `true` turns it on.";
    no_fast_greedy_chat: Switch = "off", None, "METRALE_NO_FAST_GREEDY_CHAT", "crates/server/src/scheduler/levers.rs", "Ships ON; `METRALE_NO_FAST_GREEDY_CHAT=1` restores the per-seq `[K,vocab]`-D2H slow path (the byte-invariant tie-breaking arm). `1` turns it on.";
    no_thinkended_gpu_argmax: Switch = "off", None, "METRALE_NO_THINKENDED_GPU_ARGMAX", "crates/server/src/scheduler/levers.rs", "`think_ended` rows may take the GPU argmax fast path; kill switch `METRALE_NO_THINKENDED_GPU_ARGMAX=1`. `1` turns it on.";
    numeric_rne_child: Text = "unset", None, "METRALE_NUMERIC_RNE_CHILD", "crates/core/src/numeric.rs", "Read in `crates/core/src/numeric.rs`.";
    parallel_sample: Switch = "on", None, "METRALE_PARALLEL_SAMPLE", "crates/server/src/scheduler/levers.rs", "`METRALE_PARALLEL_SAMPLE` (default on; `0` disables). `0` turns it off.";
    tool_eos_escape: Switch = "on", None, "METRALE_TOOL_EOS_ESCAPE", "crates/server/src/scheduler/levers.rs", "`METRALE_TOOL_EOS_ESCAPE` (default on). `0` turns it off.";
    tool_response_stop: Switch = "on", None, "METRALE_TOOL_RESPONSE_STOP", "crates/server/src/scheduler/levers.rs", "Ships ON. `0` turns it off.";
}
