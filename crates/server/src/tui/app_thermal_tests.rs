// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests that an [`App`] starts the hardware sampler only when its caller hands it
//! one. Before `App::new` took the probe, every App a test built leaked a `metrale-thermal`
//! thread that exec'd `nvidia-smi` every 2 s for the rest of the test process. On a GPU host
//! that was a fork storm of about 180 samplers; on CPU CI `nvidia-smi` is absent, so nothing
//! showed.
//!
//! Owner: server tui.
//! Invariants:
//! - The thread is found by its name in `/proc/self/task/*/comm` (Linux only).
//! - Thread names are kernel-truncated to 15 bytes; `metrale-thermal` is exactly 15.

use std::time::{Duration, Instant};

use super::App;
use crate::tui::data::thermal::ThermalProbe;

/// 2026-10-16: The sampler thread's name, as `ThermalProbe::spawn` sets it.
const SAMPLER: &str = "metrale-thermal";

/// 2026-10-16: The threads of this process named `SAMPLER`. Another test's sampler counts too:
/// with the fix no test starts one, except the ignored control below.
fn samplers() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task is readable on Linux")
        .filter_map(|e| std::fs::read_to_string(e.ok()?.path().join("comm")).ok())
        .filter(|comm| comm.trim_end() == SAMPLER)
        .count()
}

/// 2026-10-16: A spawned thread names itself once it runs, so a sampler can appear a moment
/// after `App::new` returns; watch for `window` before concluding either way.
fn sampler_seen_within(window: Duration) -> bool {
    let end = Instant::now() + window;
    loop {
        if samplers() > 0 {
            return true;
        }
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn args() -> crate::cli::ServeArgs {
    clap::Parser::parse_from(["met", "org/m"])
}

/// 2026-10-16: An App built with the inert probe starts no sampler thread.
#[test]
fn an_app_with_the_inert_probe_starts_no_sampler() {
    let app = App::new(args(), ThermalProbe::default());
    assert!(
        !sampler_seen_within(Duration::from_millis(500)),
        "an App built with the inert probe started a {SAMPLER} thread"
    );
    assert!(
        !app.thermal.snapshot().have_data,
        "the inert probe never samples"
    );
}

/// 2026-10-16: CONTROL for the test above: an App handed `ThermalProbe::spawn()` does show the
/// thread, so `samplers()` can see one. Ignored on CI because the sampler execs `nvidia-smi`
/// and the thread is never stopped; run it by hand, alone:
/// `cargo test -p metrale-server --bin met -- --ignored --exact
/// tui::app::thermal_tests::control_an_app_handed_a_spawned_probe_shows_the_sampler`.
#[test]
#[ignore = "execs nvidia-smi every 2 s for the rest of the process; run by hand"]
fn control_an_app_handed_a_spawned_probe_shows_the_sampler() {
    let _app = App::new(args(), ThermalProbe::spawn());
    assert!(
        sampler_seen_within(Duration::from_secs(5)),
        "no {SAMPLER} thread appeared, so the check above cannot fire"
    );
}
