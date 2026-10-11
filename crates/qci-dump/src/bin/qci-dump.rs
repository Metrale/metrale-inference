// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The dump entry. Same functions the chat handler calls.
//! 2026-10-11: `--receipt` turns a finished run's spec into a CompatibilityReceipt.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use clap::Parser;
use metrale_qci_dump::contract::canonical::sha256;
use metrale_qci_dump::contract::hardware::SoftwareVersion;
use metrale_qci_dump::{DumpInput, gate, model_for_case, receipt_spec, render_dump};

#[derive(Parser, Debug)]
#[command(
    name = "qci-dump",
    about = "Print a QCI receipt, or admit one chat request, without a GPU"
)]
struct Args {
    /// Campaign id: gemma-4-26b-a4b or nemotron-3.5-lightning-30b-a3b.
    #[arg(long)]
    case: String,
    /// Topology id. Required for a receipt. Examples: rtx-3090, strix-halo.
    #[arg(long)]
    hardware: Option<String>,
    /// Starting-artifact path. A missing path is a named blocker, not an error.
    #[arg(long)]
    weights: Option<String>,
    /// Vision-component path. A missing path is a named blocker.
    #[arg(long)]
    vision_weights: Option<String>,
    /// Engine commit, when a binary was actually built. Omitted means unmeasured.
    #[arg(long)]
    runtime_commit: Option<String>,
    /// Chat-completions JSON file. Runs the admission path instead of the receipt.
    #[arg(long)]
    request: Option<String>,
    /// Run-spec JSON of a finished qualification run. Prints a CompatibilityReceipt.
    #[arg(long, conflicts_with = "request")]
    receipt: Option<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Some(path) = &args.receipt {
        return match receipt(&args.case, path) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::from(2)
            }
        };
    }
    if let Some(path) = &args.request {
        let model = match model_for_case(&args.case) {
            Some(model) => model,
            None => {
                eprintln!("unknown case {}", args.case);
                return ExitCode::from(2);
            }
        };
        let body = match std::fs::read(path) {
            Ok(body) => body,
            Err(err) => {
                eprintln!("could not read {path}: {err}");
                return ExitCode::from(2);
            }
        };
        match gate(&model, &body) {
            metrale_qci_dump::Gate::Pass => {
                eprintln!("case {} did not select a campaign model", args.case);
                return ExitCode::from(2);
            }
            metrale_qci_dump::Gate::Respond { body, .. } => {
                println!("{}", serde_json::to_string(&body).expect("json"));
                return ExitCode::SUCCESS;
            }
        }
    }
    let Some(hardware) = args.hardware.as_deref() else {
        eprintln!("--hardware is required when --request is absent");
        return ExitCode::from(2);
    };
    let (weights_present, weights_bytes) = probe(args.weights.as_deref());
    let (vision_present, vision_bytes) = probe(args.vision_weights.as_deref());
    let text = render_dump(&DumpInput {
        case_id: &args.case,
        hardware,
        weights_present,
        weights_bytes,
        vision_present,
        vision_bytes,
        runtime_commit: args.runtime_commit.as_deref(),
    });
    print!("{text}");
    ExitCode::SUCCESS
}

/// 2026-10-11: Read the spec, the fingerprint and every artifact it names, then build.
fn receipt(case: &str, spec_path: &str) -> Result<String, String> {
    let read = |path: &str| std::fs::read(path).map_err(|e| format!("could not read {path}: {e}"));
    let spec = receipt_spec::parse(&read(spec_path)?)?;
    let fingerprint = read(&spec.hardware_fingerprint)?;
    let mut artifacts = BTreeMap::new();
    for check in &spec.checks {
        artifacts.insert(check.artifact.clone(), sha256(&read(&check.artifact)?));
    }
    let runner = SoftwareVersion {
        name: "qci-dump".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let receipt = receipt_spec::build(case, spec, &fingerprint, &artifacts, runner)?;
    serde_json::to_string(&receipt).map_err(|e| format!("receipt JSON: {e}"))
}

fn probe(path: Option<&str>) -> (bool, Option<u64>) {
    let Some(path) = path else {
        return (false, None);
    };
    match std::fs::metadata(Path::new(path)) {
        Ok(meta) if meta.is_file() => (true, Some(meta.len())),
        _ => (false, None),
    }
}
