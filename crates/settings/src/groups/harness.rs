// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `harness` lever group: levers that place, log or build for the harness.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: The `harness` group; the harness resolves it with `HarnessSettings::resolve`, never `crate::resolve()`.
    pub struct HarnessSettings { group: "harness", class: Harness }
    harness_port: Text = "unset", None, "METRALE_HARNESS_PORT", "crates/bench/src/benchmarks/agentic/warm.rs", "The agentic benchmark's harness port; read by the process driving requests, never by the server.";
    hipcc: Text = "unset", None, "METRALE_HIPCC", "crates/kernels/build.rs", "The HIP compiler the AMD kernel build invokes.";
    home: Text = "unset", None, "METRALE_HOME", "crates/host-probe/src/levers.rs", "The artifact store root (records, runs and the serve lease), `$HOME/.metrale` when unset.";
    skip_build: Text = "unset", None, "METRALE_SKIP_BUILD", "crates/kernels/build.rs", "Skip the kernel build (CI, `cargo check`): the binary carries no PTX.";
    target_hw: Presence = "off", None, "METRALE_TARGET_HW", "crates/kernels/build.rs", "Kernel target hardware class to compile (`kernels/<hw>`). Any value turns it on, `0` included.";
    target_model: Text = "unset", None, "METRALE_TARGET_MODEL", "crates/kernels/build.rs", "Kernel target model directory to compile (`kernels/<hw>/<model>`).";
    target_quant: Text = "unset", None, "METRALE_TARGET_QUANT", "crates/kernels/build.rs", "Kernel target quantization to compile.";
    tui_log_file: Path = "unset", None, "METRALE_TUI_LOG_FILE", "crates/server/src/main.rs", "Where the dashboard tees its log.";
    warm_template_dir: Path = "unset", None, "METRALE_WARM_TEMPLATE_DIR", "crates/bench/src/benchmarks/agentic/warm.rs", "Where the agentic benchmark keeps its warm workspace template.";
}
