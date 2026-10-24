// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `met` process: parse the CLI, refuse undeclared `METRALE_*` variables, install logging, and run the subcommand.
//!
//! 2026-10-24: Moved from the binary's `main.rs` into the library; the `met` binary
//! (crates/met) builds the runtime and calls [`serve_main`] with what it embeds.
//!
//! Owner: server.
//! Invariants:
//! - `dump-serve-options`, `sync-recipes` and `doctor` return or exit before any
//!   logging subscriber or dashboard is installed.
//! - A `serve` run that returns while the fault latch is set exits through
//!   `metrale_core::fault::exit_code`, never with status 0.
//! - [`crate::embedded`] is installed before anything else runs.

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::serve;
use crate::{cli, main_modules, tui};

/// 2026-10-24: Run `met` with the binary's embedded kernel registry and kernel tree.
pub async fn serve_main(embedded: crate::embedded::Embedded) -> Result<()> {
    crate::embedded::install(embedded);

    // 2026-09-26: Parsed before the subscriber is installed, because the TUI
    // gate (`tui::plain_mode`) needs `--no-tui`.
    let cli = Cli::parse();
    // 2026-10-24: `met embed` becomes the serve of its blueprint here, with the stdio transport
    // beside it, so every later step is the serve path unchanged.
    let (cli, embed) = cli::embed_args::rewrite(cli)?;

    // 2026-09-26: Every `METRALE_*` variable in the environment must be a
    // declared lever (`metrale_config::levers::check`). An undeclared one is a
    // typo or a removed lever, so the run is refused before anything starts
    // rather than served under a configuration that does not apply.
    metrale_config::levers::check(
        std::env::vars_os().map(|(name, _)| name.to_string_lossy().into_owned()),
    )?;

    // 2026-10-17: A build without the `bench` feature refuses `benchmark` and `doctor` here,
    // with a message naming the feature, before anything is installed.
    #[cfg(not(feature = "bench"))]
    cli::no_bench::refuse_if_bench_only(&cli.command);

    // 2026-09-26: Prints the serve-option manifest as JSON and returns before
    // any subscriber, TUI or GPU exists; a dashboard would garble that output.
    if matches!(cli.command, Command::DumpServeOptions) {
        println!(
            "{}",
            serde_json::to_string_pretty(&cli::manifest::build())
                .expect("the manifest is plain data and always serialises")
        );
        return Ok(());
    }

    // 2026-09-26: Fetches the recipe index and prints two lines; no
    // subscriber, TUI or GPU.
    if matches!(cli.command, Command::SyncRecipes) {
        return cli::sync_recipes::run(&metrale_host_probe::levers::BenchLevers::resolve(
            &metrale_io_host_std::ProcessEnv,
        )?);
    }

    // 2026-09-26: `doctor` prints its findings and exits with status 1 when it
    // found a problem; no subscriber, TUI or GPU.
    #[cfg(feature = "bench")]
    if matches!(cli.command, Command::Doctor) {
        let code = cli::doctor::dispatch()?;
        std::process::exit(code);
    }

    // 2026-09-28: `circuit` plans and prints from the embedded circuit files; no
    // subscriber, TUI or GPU.
    if let Command::Circuit(args) = cli.command {
        return cli::circuit::dispatch(args);
    }

    // 2026-10-22: `blueprint` builds, signs and verifies files; no subscriber, TUI or GPU.
    if let Command::Blueprint(args) = cli.command {
        return cli::blueprint::dispatch(args);
    }

    // 2026-10-17: A serve's levers resolve here, once, before the dashboard decision reads
    // them: the serve settings, the harness group and the harness projection `serve` hands to
    // the dashboard and the auto-swap catalogue. A malformed lever stops the boot before any
    // subscriber or dashboard exists.
    let serve_levers = match &cli.command {
        Command::Serve(args) => {
            let env = metrale_io_host_std::ProcessEnv;
            let settings = main_modules::serve_settings::resolve(&env, args)?;
            let harness = metrale_host_probe::levers::resolve_harness(&env)?;
            let bench =
                metrale_host_probe::levers::BenchLevers::from_settings(&settings, &harness, &env)?;
            Some((settings, harness, std::sync::Arc::new(bench)))
        }
        _ => None,
    };
    let no_tui = match &cli.command {
        // 2026-09-26: `--check-kernels` prints a report and one JSON line on
        // stdout and exits, so it runs without a dashboard.
        Command::Serve(args) => {
            serve_levers
                .as_ref()
                .is_some_and(|(s, _, _)| s.server.no_tui)
                || args.rank > 0
                || args.check_kernels
        }
        // 2026-09-26: Benchmarks always log in plain mode.
        Command::Benchmark(_) => true,
        Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => true,
    };

    // 2026-09-26: `benchmark certify --json` writes JSON lines on stdout, so
    // the log moves to stderr for that command only.
    #[cfg(feature = "bench")]
    let logs_to_stderr = matches!(
        &cli.command,
        Command::Benchmark(b) if b.json_stdout()
    );
    // 2026-10-17: Without `bench` the only JSON-stdout command was refused above.
    #[cfg(not(feature = "bench"))]
    let logs_to_stderr = false;
    let tui_channels = if tui::plain_mode(no_tui) {
        // 2026-10-24: `met embed` logs plain text: its stderr is a host's log file.
        let fmt = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .with_ansi(embed.is_none());
        if logs_to_stderr {
            fmt.with_writer(std::io::stderr).init();
        } else {
            fmt.init();
        }
        None
    } else {
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let Some((_, harness, _)) = &serve_levers else {
            unreachable!("only a serve runs the dashboard")
        };
        tui::init::install_tty_subscriber(progress_tx, harness.tui_log_file.as_deref());
        Some(progress_rx)
    };

    // 2026-09-26: Race the server against a shutdown during startup. Pinning
    // `serve()` without a spawn is enough, because its blocking startup runs
    // under `spawn_blocking` and the future yields while it does.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<&'static str>();
    tui::shutdown::arm_startup_escape(shutdown_tx);
    let result = match cli.command {
        // 2026-09-26: Returned above. An explicit arm, not a wildcard, so a new
        // subcommand fails to compile here until it is routed.
        Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => {
            unreachable!("handled before initialisation")
        }
        #[cfg(feature = "bench")]
        Command::Benchmark(args) => {
            // 2026-09-26: Benchmarks load no model and do not use the startup
            // escape.
            drop(shutdown_rx);
            let levers =
                metrale_host_probe::levers::BenchLevers::resolve(&metrale_io_host_std::ProcessEnv)?;
            cli::bench_run::dispatch(args, std::sync::Arc::new(levers)).await
        }
        // 2026-10-17: Refused before initialisation (`cli::no_bench`).
        #[cfg(not(feature = "bench"))]
        Command::Benchmark(_) => unreachable!("refused before initialisation"),
        Command::Serve(args) => {
            let Some((settings, _, bench_levers)) = serve_levers else {
                unreachable!("resolved above for every serve")
            };
            let serving = serve(args, settings, tui_channels, bench_levers, embed);
            tokio::pin!(serving);
            // 2026-09-26: Only a send means shutdown. The sender stays parked in
            // `tui::shutdown` for the life of the process, and a closed channel
            // waits forever instead of stopping a healthy server.
            let shutdown_signal = async {
                match shutdown_rx.await {
                    Ok(reason) => reason,
                    Err(_) => std::future::pending::<&'static str>().await,
                }
            };
            tokio::pin!(shutdown_signal);
            tokio::select! {
                res = &mut serving => res,
                reason = &mut shutdown_signal => {
                    // 2026-09-26: Cancelled before the server came up: nothing is
                    // in flight, and the startup task is abandoned.
                    tracing::info!(
                        "Shutdown requested ({reason}) during startup — exiting before the server came up"
                    );
                    // 2026-09-26: The same cleanup as the normal tail, then exit
                    // without returning: the startup task on the blocking pool
                    // cannot be aborted, and dropping the runtime would wait for it.
                    tui::stop_and_join(std::time::Duration::from_secs(2));
                    tui::terminal_guard::restore();
                    tui::init::flush_tee();
                    // 2026-09-26: A failed kernel launch or memset can latch the
                    // fault before the server is up, so this exit maps the status
                    // the same way as the normal tail.
                    std::process::exit(metrale_core::fault::exit_code(
                        true,
                        metrale_core::fault::global().fault(),
                    ));
                }
            }
        }
    };
    // 2026-09-26: Stop the dashboard thread and wait up to 2 s for it to drop
    // its terminal guard before an error prints. `restore()` then covers a
    // thread that had not entered raw mode yet or did not stop in time.
    tui::stop_and_join(std::time::Duration::from_secs(2));
    tui::terminal_guard::restore();
    tui::init::flush_tee();

    // 2026-09-26: A GPU fault requests the same drain as `SIGTERM`, so
    // `serve()` can return `Ok`; the latch is what makes the exit status
    // nonzero. A healthy run returns its own result.
    match metrale_core::fault::global().fault() {
        Some(reason) => {
            if let Err(e) = &result {
                tracing::error!("{e:#}");
            }
            tracing::error!(
                "Exiting after a fatal GPU fault ({reason}). The CUDA context is \
                 destroyed and cannot be recovered in-process; restart the server."
            );
            std::process::exit(metrale_core::fault::exit_code(result.is_ok(), Some(reason)));
        }
        None => result,
    }
}

#[cfg(test)]
#[path = "main_exit_tests.rs"]
mod main_exit_tests;
