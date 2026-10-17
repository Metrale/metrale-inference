// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `tool` lever group: levers read only by an operator script or the site build.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: Tool levers; no `met` process resolves them.
    pub specs ToolLevers { group: "tool", class: Tool }
    agent_shell: Text = "unset", None, "METRALE_AGENT_SHELL", "bench/fp8_dgx2_drift/harness/cargo-shim/cargo", "Marker the agentic benchmark exports into the agent's shell; the cargo shim keys on it.";
    base_url: Text = "unset", None, "METRALE_BASE_URL", "examples/python/chat_completions.py", "Server base URL for the Python example client.";
    bin: Text = "unset", None, "METRALE_BIN", "scripts/mlperf-edge/run_golden_e2e.sh", "Path to the `met` binary a script runs.";
    container: Text = "unset", None, "METRALE_CONTAINER", "bench/agentic/conc_harness.py", "Container whose log the concurrency harness reads.";
    diag: Text = "unset", None, "METRALE_DIAG", "scripts/start-deepseek-ep2-redhat.sh", "Turns the DeepSeek EP2 launcher's per-layer diagnostics on.";
    enable_ep4: Text = "unset", None, "METRALE_ENABLE_EP4", "tests/run_all_models.py", "Opts the all-models suite into its EP4 rounds.";
    gate_strict: Text = "unset", None, "METRALE_GATE_STRICT", "scripts/hopper_ptx_gate.sh", "Makes the Hopper PTX gate script strict.";
    git_sha: Text = "unset", None, "METRALE_GIT_SHA", "docker/gb10/Dockerfile", "Image build argument: the revision label.";
    head_ip: Text = "unset", None, "METRALE_HEAD_IP", "tests/run_all_models.py", "Head node address for the all-models suite.";
    head_port: Text = "unset", None, "METRALE_HEAD_PORT", "tests/run_all_models.py", "Head node port for the all-models suite.";
    hf_cache_head: Text = "unset", None, "METRALE_HF_CACHE_HEAD", "tests/run_all_models.py", "Head node Hugging Face cache for the all-models suite.";
    hf_cache_worker: Text = "unset", None, "METRALE_HF_CACHE_WORKER", "tests/run_all_models.py", "Worker node Hugging Face cache for the all-models suite.";
    hf_hub: Text = "unset", None, "METRALE_HF_HUB", "scripts/dev/run_all_final.py", "Hugging Face hub directory the dev runner mounts.";
    image: Text = "unset", None, "METRALE_IMAGE", "tests/run_all_models.py", "Container image the all-models suite runs.";
    model: Text = "unset", None, "METRALE_MODEL", "bench/agentic/two_wave.py", "Served model name the bench scripts request.";
    opencode_output_cap: Text = "unset", None, "METRALE_OPENCODE_OUTPUT_CAP", "bench/fp8_dgx2_drift/harness/run_tier.sh", "Output-token cap the drift harness writes into the opencode config.";
    rate_limit_rpm_observed: Text = "unset", None, "METRALE_RATE_LIMIT_RPM_OBSERVED", "tests/test_openai_compat_v2.py", "The rate limit the compat test expects the server to enforce.";
    real_cargo: Text = "unset", None, "METRALE_REAL_CARGO", "bench/fp8_dgx2_drift/harness/cargo-shim/cargo", "The cargo the shim forwards to.";
    url: Text = "unset", None, "METRALE_URL", "bench/agentic/coherence_check.py", "Chat-completions URL the bench scripts call.";
    vision_test_image: Text = "unset", None, "METRALE_VISION_TEST_IMAGE", "tests/single_gpu_suite.py", "Image the single-GPU vision test sends.";
    vision_test_keywords: Text = "unset", None, "METRALE_VISION_TEST_KEYWORDS", "tests/single_gpu_suite.py", "Keywords the single-GPU vision test expects.";
    worker_ip: Text = "unset", None, "METRALE_WORKER_IP", "tests/run_all_models.py", "Worker node address for the all-models suite.";
    worker_port: Text = "unset", None, "METRALE_WORKER_PORT", "tests/run_all_models.py", "Worker node port for the all-models suite.";
    worker_url: Text = "unset", None, "METRALE_WORKER_URL", "bench/prefix_cache_contamination.py", "Worker URL the prefix-cache contamination probes call.";
    ws_build_timeout: Text = "unset", None, "METRALE_WS_BUILD_TIMEOUT", "bench/fp8_dgx2_drift/harness/score_run.py", "Build timeout the webserver scorer allows.";
}
