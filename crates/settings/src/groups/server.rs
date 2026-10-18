// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `server` lever group: the `server.*` levers a `met` process reads at run time.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: The `server` group, resolved by `crate::resolve()`.
    pub struct ServerSettings { group: "server", class: Runtime }
    agentic_sampling: Text = "unset", None, "METRALE_AGENTIC_SAMPLING", "crates/host-probe/src/levers.rs", "METRALE_AGENTIC_SAMPLING=model-card: omit the greedy pins so the server's card presets own sampling (A/B arm; the GATE default stays pinned-greedy per the rationale above TEMPERATURE).";
    d2h_spin_sync: Switch = "off", None, "METRALE_D2H_SPIN_SYNC", "crates/gpu-runtime/src/cuda_backend/gpu_copy.rs", "METRALE_D2H_SPIN_SYNC=1 (read once) busy-polls `cuStreamQuery` until it stops returning CUDA_ERROR_NOT_READY, instead of blocking in `cuStreamSynchronize`. `1` turns it on.";
    hermetic: Text = "unset", "--hermetic", "METRALE_HERMETIC", "crates/gpu-runtime/src/lib.rs", "So absent leaves `METRALE_HERMETIC` live, and passing the flag wins.";
    host_transpose: Switch = "off", None, "METRALE_HOST_TRANSPOSE", "crates/model-layers/src/weight_map/quantized/transpose.rs", "`None` when the target's kernel set lacks it, or when `METRALE_HOST_TRANSPOSE=1` forces the host path (parity/debug kill switch). `1` turns it on.";
    marconi_exact: Switch = "off", None, "METRALE_MARCONI_EXACT", "crates/model-engine/src/model/trait_impl/prefill_b/exact_leaf.rs", "`METRALE_MARCONI_EXACT=1` re-enables the exact full-prompt shortcut. `1` turns it on.";
    marconi_min_tokens: Int = "unset", "--marconi-min-tokens", "METRALE_MARCONI_MIN_TOKENS", "crates/model-layers/src/mtp_carry.rs", "`METRALE_MARCONI_MIN_TOKENS=<n>` overrides; 0 restores the previous always-restore behaviour.";
    mixed_slice_tokens: Int = "0", None, "METRALE_MIXED_SLICE_TOKENS", "crates/server/src/scheduler/levers.rs", "`METRALE_MIXED_SLICE_TOKENS` (0 = policy default).";
    no_tui: Switch = "off", "--no-tui", "METRALE_NO_TUI", "crates/server/src/main.rs", "Start without the dashboard: `--no-tui` or this switch; a non-interactive stdout OR stdin (docker `-t` without `-i` therefore stays plain) or `TERM=dumb` also starts plain. `1` turns it on.";
    rate_limit_burst_rpm: Text = "unset", None, "METRALE_RATE_LIMIT_BURST_RPM", "crates/server/src/rate_limiter.rs", "Env configuration (all default 0 = disabled, pure passthrough): METRALE_RATE_LIMIT_RPM — requests per minute cap METRALE_RATE_LIMIT_TPM — tokens per minute cap METRALE_RATE_LIMIT_BURST_RPM — max...";
    rate_limit_burst_tpm: Text = "unset", None, "METRALE_RATE_LIMIT_BURST_TPM", "crates/server/src/rate_limiter.rs", "Env configuration (all default 0 = disabled, pure passthrough): METRALE_RATE_LIMIT_RPM — requests per minute cap METRALE_RATE_LIMIT_TPM — tokens per minute cap METRALE_RATE_LIMIT_BURST_RPM — max...";
    rate_limit_rpm: Text = "unset", None, "METRALE_RATE_LIMIT_RPM", "crates/server/src/rate_limiter.rs", "Env configuration (all default 0 = disabled, pure passthrough): METRALE_RATE_LIMIT_RPM — requests per minute cap METRALE_RATE_LIMIT_TPM — tokens per minute cap METRALE_RATE_LIMIT_BURST_RPM — max...";
    rate_limit_tpm: Text = "unset", None, "METRALE_RATE_LIMIT_TPM", "crates/server/src/rate_limiter.rs", "Env configuration (all default 0 = disabled, pure passthrough): METRALE_RATE_LIMIT_RPM — requests per minute cap METRALE_RATE_LIMIT_TPM — tokens per minute cap METRALE_RATE_LIMIT_BURST_RPM — max...";
    store_max_entries: Text = "unset", None, "METRALE_STORE_MAX_ENTRIES", "crates/server/src/response_store/store_impl.rs", "Build from env. - `METRALE_STORE_MAX_ENTRIES` (default 10 000) - `METRALE_STORE_TTL_SECONDS` (default 86 400) - `METRALE_STORE_DIR` — when set, enable filesystem persistence and replay any...";
    store_ttl_seconds: Text = "unset", None, "METRALE_STORE_TTL_SECONDS", "crates/server/src/response_store/store_impl.rs", "Build from env. - `METRALE_STORE_MAX_ENTRIES` (default 10 000) - `METRALE_STORE_TTL_SECONDS` (default 86 400) - `METRALE_STORE_DIR` — when set, enable filesystem persistence and replay any...";
}
